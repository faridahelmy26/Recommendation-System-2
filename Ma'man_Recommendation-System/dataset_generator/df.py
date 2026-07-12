import numpy as np
import pandas as pd
from pathlib import Path

# ============================================
# Project Paths
# ============================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

print("Loading data from:", DATA_DIR)

# ============================================
# Load Files
# ============================================

content = pd.read_csv(DATA_DIR / "content.csv")
users = pd.read_excel(DATA_DIR / "users.xlsx")
interactions = pd.read_csv(DATA_DIR / "interactions.csv")

# ============================================
# Rename Columns
# ============================================

content = content.rename(columns={
    "level": "course_level",
    "rating": "course_rating"
})

interactions = interactions.rename(columns={
    "rating": "user_rating"
})

# ============================================
# Merge
# ============================================

merged = interactions.merge(
    users,
    on="user_id",
    how="left"
)

merged = merged.merge(
    content,
    on="content_id",
    how="left"
)

# ============================================
# Feature Engineering
# ============================================

# Age Group
merged["age_group"] = pd.cut(

    merged["age"],

    bins=[0,18,25,40,100],

    labels=[
        "Teen",
        "Young Adult",
        "Adult",
        "Senior"
    ]
)

# Completion Rate

merged["completion_rate"] = (

    merged["time_spent"] /

    (merged["duration"] )

)

merged["completion_rate"] = merged["completion_rate"].clip(0,1)

# Completed

merged["is_completed"] = (

    merged["completion_rate"] >= 0.80

).astype(int)

# Engagement Score

merged["engagement_score"] = (

    0.6 *

    (merged["user_rating"]/5)

    +

    0.4 *

    merged["completion_rate"]

)

merged["engagement_score"] = merged["engagement_score"].round(3)

# Activity Level

interaction_count = (

    merged.groupby("user_id")

    .size()

)

merged["activity_level"] = merged["user_id"].map(

    interaction_count

)

merged["activity_level"] = pd.cut(

    merged["activity_level"],

    bins=[0,20,50,100000],

    labels=[

        "Low",

        "Medium",

        "High"

    ]

)

# Popularity

popularity = (

    merged.groupby("content_id")

    .size()

)

merged["course_popularity"] = merged["content_id"].map(popularity)

# Days Since Interaction

merged["timestamp"] = pd.to_datetime(

    merged["timestamp"]

)

merged["days_since_last_interaction"] = (

    pd.Timestamp.now()

    -

    merged["timestamp"]

).dt.days

# ============================================
# Reorder Columns
# ============================================

merged = merged[[
    "user_id",
    "age",
    "age_group",
    "interest",
    "level",
    "learning_style",
    "activity_level",

    "content_id",
    "title",
    "category",
    "course_level",
    "duration",
    "difficulty",
    "course_rating",

    "user_rating",
    "time_spent",
    "completion_rate",
    "engagement_score",
    "is_completed",

    "course_popularity",
    "days_since_last_interaction",

    "timestamp"
]]

# ============================================
# Save
# ============================================

output_file = DATA_DIR / "df.csv"

merged.to_csv(
    output_file,
    index=False
)

print("\nSaved Successfully!")
print(output_file)

print("="*70)

print("Merged Dataset Created Successfully")

print("="*70)

print("Shape :", merged.shape)

print()

print(merged.head())

print()

print("Saved to data/df.csv")