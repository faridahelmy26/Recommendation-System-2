import pandas as pd
import numpy as np
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# Load Data
# =========================
content_df = pd.read_csv("content.csv")
interactions_df = pd.read_csv("interactions.csv")
users_df = pd.read_excel("users.xlsx")
# =========================
# Build Text Features
# =========================
content_df['text'] =(
    content_df['category'].astype(str) + " " +
    content_df['level'].astype(str) + " " +
    content_df['description'].astype(str)
    )

# =========================
# Load Model
# =========================
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(content_df['text'].tolist())
cosine_sim = cosine_similarity(embeddings)

# =========================
# Create FastAPI App
# =========================
app = FastAPI(title="Recommendation System API")

# =========================
# Helper Function
# =========================
def recommend_by_title(title, top_n=5):

    # البحث بجزء من الاسم
    matches = content_df[
        content_df['title']
        .str.lower()
        .str.contains(title.lower(), na=False)
    ]

    if len(matches) == 0:
        return {
            "error": f"No course found containing '{title}'"
        }

    # أول كورس مطابق
    idx = matches.index[0]

    selected_course = content_df.iloc[idx]['title']

    # similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )

    sim_scores = sim_scores[1:top_n+1]

    recommendations = []

    for i, score in sim_scores:

        recommendations.append({
            "content_id": int(content_df.iloc[i]['content_id']),
            "title": content_df.iloc[i]['title'],
            "category": content_df.iloc[i]['category'],
            "level": content_df.iloc[i]['level'],
            "score": round(float(score), 4)
        })

    return {
        "matched_course": selected_course,
        "recommendations": recommendations
    }
# =========================
# API Endpoint
# =========================
@app.get("/recommend")
def recommend(title: str, top_n: int = 5):
    return recommend_by_title(title, top_n)
