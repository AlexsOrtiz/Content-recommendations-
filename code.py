import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/articles.csv", encoding='latin1')
data.head()
articles = data["Article"].tolist()
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(articles)
cosine_sim = cosine_similarity(tfidf_matrix)

def recommend_articles(x):
    return ", ".join(data["Title"].iloc[x.argsort()[-5:-1]])

data["Recommended Articles"] = [recommend_articles(x) for x in cosine_sim]
data.head()

print(data["Recommended Articles"][15])
