import streamlit as st
import pandas as pd
import openai
import os
from apify_client import ApifyClient
from openai.embeddings_utils import get_embedding
import tiktoken
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime

def main():
    st.title("Political Campaign Analytics")

    # Select the data source
    data_source = st.selectbox("Select data source", ["Twitter", "Polling", "Funding"])

    if data_source == "Twitter":
        st.subheader("Twitter Analytics")

        # Input fields
        search_option = st.selectbox("Search by", ["Profile", "Keyword"])
        if search_option == "Profile":
            handle = st.text_input("Enter the Twitter handle (without @)")
        else:
            keyword = st.text_input("Enter the search keyword")

        n = st.number_input("Enter the number of tweets/posts to fetch:", min_value=1, value=1000, step=1)

        from_date = st.date_input("From date:")
        to_date = st.date_input("To date:")

        if st.button("Fetch Data"):
            try:
                if search_option == "Profile":
                    st.write(f"Fetching {n} tweets for @{handle}")
                    handles = [handle]
                    df = fetch_twitter_data(handles, n, from_date=from_date, to_date=to_date)
                else:
                    st.write(f"Fetching {n} tweets containing '{keyword}'")
                    df = fetch_twitter_data_by_keyword(keyword, n, from_date=from_date, to_date=to_date)

                st.write("Data fetched successfully.")
                st.write(df.head())

                st.write("Generating embeddings...")
                embeddings_matrix = generate_embeddings(df)
                st.write("Embeddings generated.")

                # Select the analysis type
                analysis_option = st.selectbox("Select analysis type", ["Clustering", "Similarity Search", "Sentiment Analysis"])

                if analysis_option == "Clustering":
                    st.write("Clustering the data...")
                    scaler = StandardScaler()
                    scaled_embeddings = scaler.fit_transform(embeddings_matrix)
                    optimal_k = 3  # Replace this with the optimal k value you found from the plot
                    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
                    clusters = kmeans.fit_predict(scaled_embeddings)

                    pca = PCA(n_components=2)
                    embeddings_2d = pca.fit_transform(scaled_embeddings)

                    st.write("Visualizing the clusters...")
                    plot_clusters(embeddings_2d, clusters)

                    num_examples = 30  # Adjust this value based on your needs
                    for cluster_idx in range(optimal_k):
                        cluster_data = np.array(scaled_embeddings)[clusters == cluster_idx]
                        distances = [np.linalg.norm(embedding - kmeans.cluster_centers_[cluster_idx]) for embedding in cluster_data]
                        closest_examples_idx = np.argsort(distances)[:num_examples]

                        st.write(f"\nCluster {cluster_idx}:")
                        for idx in closest_examples_idx:
                            st.write(f"- {df.loc[idx, 'Author']} + {df.loc[idx, 'Text']}")

                elif analysis_option == "Similarity Search":
                    st.write("Computing similarity scores...")
                    phrase = st.text_input("Enter a phrase to compute similarity scores:")
                    if phrase:
                        phrase_embedding = get_embedding(phrase)
                        df['Similarity'] = df.apply(lambda row: cosine_similarity(phrase_embedding, row['embedding']), axis=1)
                        st.write(df.head())

                        st.write("Visualizing similarity over time...")
                        plot_similarity(df)

                elif analysis_option == "Sentiment Analysis":
                    # Implement sentiment analysis here

                    pass

            except Exception as e:
                st.write("An error occurred while fetching or processing data.")
                st.write(e)

