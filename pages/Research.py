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
