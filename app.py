
import streamlit as st
import pandas as pd
import numpy as np
import re
import wikipedia
from sklearn.metrics.pairwise import cosine_similarity
from model_utils import preprocess, get_movie_vector, load_fasttext_model

# Load everything
@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv("movies.csv")
    model = load_fasttext_model("fasttext_model.bin")
    df['processed'] = df['combined'].apply(preprocess)
    df['vector'] = df['processed'].apply(lambda tokens: get_movie_vector(tokens, model))
    return df, model

movies, model = load_data()

# Helper
def interpret_score(value):
    if value >= 0.85:
        return "Very High"
    elif value >= 0.70:
        return "High"
    elif value >= 0.50:
        return "Moderate"
    else:
        return "Low"

def get_recommendations_from_text(query_text, df, model, top_n=5):
    tokens = preprocess(query_text)
    vector = get_movie_vector(tokens, model).reshape(1, -1)
    all_vectors = np.stack(df['vector'].values)
    similarities = cosine_similarity(vector, all_vectors)[0]
    top_indices = similarities.argsort()[::-1][:top_n]

    results = df.iloc[top_indices][['title', 'release_date', 'genres', 'overview']].copy()
    results['Similarity Score'] = [round(similarities[i], 4) for i in top_indices]
    results['Interpretation'] = results['Similarity Score'].apply(interpret_score)
    return results.reset_index(drop=True)

# Streamlit UI
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 FastText Movie Recommender")
st.write("Get recommendations based on plot (Wikipedia or custom).")

mode = st.radio("Input Mode", ["Wikipedia Title", "Custom Description"])

if mode == "Wikipedia Title":
    title = st.text_input("Enter movie/show title:")
    if st.button("Recommend"):
        if title:
            try:
                page = wikipedia.page(title)
                plot = page.content[:1000]  # fallback: first 1000 chars
                st.info(plot)
                results = get_recommendations_from_text(plot, movies, model)
                st.dataframe(results)
            except:
                st.error("Could not fetch Wikipedia page.")
else:
    plot = st.text_area("Describe your movie/story:")
    if st.button("Recommend"):
        if plot:
            results = get_recommendations_from_text(plot, movies, model)
            st.dataframe(results)
