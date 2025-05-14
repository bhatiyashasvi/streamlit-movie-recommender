
import pandas as pd
import numpy as np
from gensim.models import FastText
import re

# Load model
def load_fasttext_model(path='fasttext_model.bin'):
    return FastText.load(path)

# Preprocessing
def preprocess(text):
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [
        token.lemma_.lower().strip()
        for token in doc
        if not token.is_stop and not token.is_punct and token.is_alpha
    ]

# Generate movie vector
def get_movie_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.random.rand(model.vector_size) * 1e-6
