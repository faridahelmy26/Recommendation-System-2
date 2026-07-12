from pathlib import Path
import random

# ==========================================
# Random Seed
# ==========================================

SEED = 42
random.seed(SEED)


# ==========================================
# Output Folder
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "data"

OUTPUT_DIR.mkdir(exist_ok=True)

# ==========================================
# Dataset Size
# ==========================================

NUM_CONTENT = 2000

NUM_USERS = 5000

NUM_INTERACTIONS = 200000

# ==========================================
# Categories
# ==========================================

CATEGORIES = [

    "Early Childhood Education",

    "Primary Education",

    "Child Development",

    "STEM for Kids",

    "Arts & Creativity",

    "Language & Literacy",

    "Special Needs & Inclusion",

    "Social & Emotional Learning",

    "Physical Education & Health",

    "Parenting & Family"

]

# ==========================================
# Levels
# ==========================================

LEVELS = [

    "Beginner",

    "Intermediate",

    "Advanced"

]

# ==========================================
# Learning Styles
# ==========================================

LEARNING_STYLES = [

    "Visual",

    "Audio",

    "Reading/Writing",

    "Kinesthetic"

]

# ==========================================
# Difficulty Mapping
# ==========================================

LEVEL_TO_DIFFICULTY = {

    "Beginner": 1,

    "Intermediate": 2,

    "Advanced": 3

}

# ==========================================
# Duration
# ==========================================

MIN_DURATION = 20

MAX_DURATION = 240

# ==========================================
# Rating Distribution
# ==========================================

MIN_RATING = 3.5

MAX_RATING = 5.0

# ==========================================
# Age
# ==========================================

MIN_AGE = 16

MAX_AGE = 55

# ==========================================
# Time Spent (Minutes)
# ==========================================

MIN_TIME_SPENT = 20

MAX_TIME_SPENT = 2400

# ==========================================
# Interaction Probabilities
# ==========================================

# 75% نفس الاهتمام
MATCH_INTEREST = 0.75

# 15% مجال قريب
RELATED_INTEREST = 0.15

# 10% عشوائي
RANDOM_INTEREST = 0.10

# ==========================================
# Related Categories
# ==========================================
# FIX: this used to list old tech categories (Programming, Data Science,
# Cyber Security, ...) that don't exist anywhere in CATEGORIES above. Since
# every real `interest` value is one of the education categories, every
# single lookup fell through to the `.get(interest, [interest])` default —
# meaning the "15% related category" path silently always resolved to the
# SAME category as the 75% path. Net effect: interactions were really
# ~90% same-category / 10% random, not 75/15/10, and the model never saw
# any genuine cross-category "related interest" signal.
#
# Updated to actually map each real category to 1-2 genuinely related ones.
RELATED_CATEGORY = {

    "Early Childhood Education": [
        "Child Development",
        "Parenting & Family"
    ],

    "Primary Education": [
        "STEM for Kids",
        "Language & Literacy"
    ],

    "Child Development": [
        "Early Childhood Education",
        "Social & Emotional Learning"
    ],

    "STEM for Kids": [
        "Primary Education"
    ],

    "Arts & Creativity": [
        "Social & Emotional Learning"
    ],

    "Language & Literacy": [
        "Primary Education",
        "Special Needs & Inclusion"
    ],

    "Special Needs & Inclusion": [
        "Child Development",
        "Language & Literacy"
    ],

    "Social & Emotional Learning": [
        "Child Development",
        "Parenting & Family"
    ],

    "Physical Education & Health": [
        "Parenting & Family"
    ],

    "Parenting & Family": [
        "Child Development",
        "Physical Education & Health"
    ]

}

# ==========================================
# Output Files
# ==========================================

CONTENT_FILE = OUTPUT_DIR / "content.csv"

USERS_FILE = OUTPUT_DIR / "users.xlsx"

INTERACTIONS_FILE = OUTPUT_DIR / "interactions.csv"

print("=" * 60)
print("Output Directory :", OUTPUT_DIR)
print("Content File     :", CONTENT_FILE)
print("Users File       :", USERS_FILE)
print("Interactions File:", INTERACTIONS_FILE)
print("=" * 60)