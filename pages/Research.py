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

st.set_option('deprecation.showPyplotGlobalUse', False)

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

openai.api_key = "your_openai_api_key"
client = ApifyClient("your_apify_api_key")

encoding = tiktoken.get_encoding(embedding_encoding)

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

def fetch_twitter_data(handles, n):
    run_input = {
        "handle": handles,
        "tweetsDesired": n,
        "profilesDesired": len(handles),
    }
    run = client.actor("quacker/twitter-scraper").call(run_input=run_input)
    tweet_data = [(item['created_at'], item['full_text'], item['user']['screen_name']) for item in client.dataset(run["defaultDatasetId"]).iterate_items()]
    df = pd.DataFrame(tweet_data, columns=['Date', 'Text', 'Author'])
    return df

def generate_embeddings(df):
    df['embedding'] = df.Text.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
    embeddings = df['embedding'].apply(lambda x: np.array(x))
    embeddings_matrix = np.vstack(embeddings)
    return embeddings_matrix

def plot_clusters(embeddings_2d, clusters):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(embeddings_2d[:, 0], embeddings_2d[:, 1], hue=clusters, palette='viridis', legend="full", s=100, alpha=0.7)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('Cluster Visualization of OpenAI Embeddings')
    st.pyplot()

def plot_similarity(df):
    df['Date'] = pd.to_datetime(df['Date'])
    plt.scatter(df['Date'], df['Similarity'])
    plt.xlabel('Date')
    plt.ylabel('Similarity')
    plt.title('Similarity Over Time')
    st.pyplot()

st.title("Twitter Analytics")
st.sidebar.title("Settings")

handle = st.text_input("Enter a Twitter handle:")
n = st.sidebar.slider("Number of tweets to analyze:", 100, 500, 1000)

if st.button("Fetch Data"):
    try:
        st.write(f"Fetching {n} tweets for @{handle}")
        handles = [handle]
        df = fetch_twitter_data(handles, n)
        st.write("Data fetched successfully.")
        st.write(df.head())

        st.write("Generating embeddings...")
        embeddings_matrix = generate_embeddings(df)
        st.write("Embeddings generated.")

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

        # Computing similarity scores
        st.write("Computing similarity scores...")
        phrase = st.text_input("Enter a phrase to compute similarity scores:")
        if phrase:
            phrase_embedding = get_embedding(phrase)
            df['Similarity'] = df.apply(lambda row: cosine_similarity(phrase_embedding, row['embedding']), axis=1)
            st.write(df.head())

            st.write("Visualizing similarity over time...")
            plot_similarity(df)

    except Exception as e:
        st.write("An error occurred while fetching or processing data.")
        st.write(e)

