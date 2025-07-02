import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Title
st.title("🎬 Netflix Content Recommendation System (K-Means Based)")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("D:\\guvi\\netflix project\\netflix_clustered.csv", thousands=",")
    df['release_year'] = df['release_year'].astype(str)  # ✅ Convert to string to avoid comma display
    return df

df = load_data()

# User input
st.sidebar.header("📌 Filter Options")
genre_input = st.sidebar.selectbox("Select Genre", sorted(set(df['listed_in'].str.split(', ').sum())))
rating_input = st.sidebar.selectbox("Select Rating", sorted(df.columns[df.columns.str.startswith("rating_")]))

# Filter based on input
st.subheader("🔎 Content Matching Your Preference")
filtered_df = df[df['listed_in'].str.contains(genre_input, case=False)]
filtered_df = filtered_df[filtered_df[rating_input] == 1]

if filtered_df.empty:
    st.warning("No content found for this combination.")
else:
    st.dataframe(filtered_df[['title', 'type', 'listed_in', 'release_year']].reset_index(drop=True))

# Clustering and Recommendation
st.subheader("🤖 Recommended for You (Based on Similar Cluster)")

# Choose first match
if not filtered_df.empty:
    selected = filtered_df.iloc[0]
    cluster_id = selected['cluster_kmeans']
    
    recommendations = df[df['cluster_kmeans'] == cluster_id]
    recommendations = recommendations[recommendations['title'] != selected['title']]
    recommendations = recommendations.sample(min(10, len(recommendations)))

    st.dataframe(recommendations[['title', 'type', 'listed_in', 'release_year']].reset_index(drop=True))