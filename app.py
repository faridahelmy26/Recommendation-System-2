from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI(title="Recommendation System API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

content_df = pd.read_csv(BASE_DIR / "data" / "content.csv")
interactions_df = pd.read_csv(BASE_DIR / "data" / "interactions.csv")
df = pd.read_csv(BASE_DIR / "data" / "df.csv")

# =========================
# TF-IDF
# =========================
content_df["features"] = (
    content_df["category"].astype(str) + " " +
    content_df["level"].astype(str) + " " +
    content_df["difficulty"].astype(str) + " " +
    content_df["description"].astype(str)
)

vectorizer = TfidfVectorizer(stop_words="english")
feature_matrix = vectorizer.fit_transform(content_df["features"])
cosine_sim = cosine_similarity(feature_matrix)


# =========================
# SAFE title -> content_id
# =========================
def get_content_id(title: str):

    query_vec = vectorizer.transform([title])
    sims = cosine_similarity(query_vec, feature_matrix)

    idx = np.argmax(sims)

    # safety check
    if sims[0][idx] == 0:
        return None

    return int(content_df.iloc[idx]["content_id"])


# =========================
# CONTENT-BASED
# =========================
def recommend_content(title: str):

    cid = get_content_id(title)
    if cid is None:
        return []

    row = content_df[content_df["content_id"] == cid]
    if row.empty:
        return []

    idx = row.index[0]

    scores = cosine_sim[idx]
    rec_idx = np.argsort(scores)[::-1]

    results = []

    for i in rec_idx:
        rec_id = content_df.iloc[i]["content_id"]

        if rec_id != cid:
            results.append({
                "content_id": int(rec_id),
                "title": content_df.iloc[i]["title"]
            })

        if len(results) == 5:
            break

    return results


# =========================
# COLLABORATIVE (stable version)
# =========================
def collaborative_recommend(title: str):

    cid = get_content_id(title)
    if cid is None:
        return []

    users = interactions_df[
        interactions_df["content_id"] == cid
    ]["user_id"].unique()

    if len(users) == 0:
        return []

    recs = interactions_df[
        interactions_df["user_id"].isin(users)
    ]["content_id"].value_counts().head(5).index

    return content_df[
        content_df["content_id"].isin(recs)
    ][["content_id", "title"]].to_dict(orient="records")


# =========================
# ML (popularity-based safe)
# =========================
def ml_recommend(title: str):

    cid = get_content_id(title)
    if cid is None:
        return []

    users = interactions_df[
        interactions_df["content_id"] == cid
    ]["user_id"].values

    if len(users) == 0:
        return []

    top_items = interactions_df[
        interactions_df["user_id"].isin(users)
    ]["content_id"].value_counts().head(5).index

    return content_df[
        content_df["content_id"].isin(top_items)
    ][["content_id", "title"]].to_dict(orient="records")


# =========================
# API
# =========================
@app.get("/recommend")
def recommend(title: str):

    content = recommend_content(title)
    collab = collaborative_recommend(title)
    ml = ml_recommend(title)

    final = content + collab + ml

    seen = set()
    unique = []

    for item in final:
        if item["content_id"] not in seen:
            unique.append(item)
            seen.add(item["content_id"])

    if len(unique) == 0:
        raise HTTPException(status_code=404, detail="No recommendations found")

    return {
        "input": title,
        "recommendations": unique[:10]
    }