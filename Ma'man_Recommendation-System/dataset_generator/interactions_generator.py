"""
=========================================
Interactions Generator
Generate 200000 Realistic Interactions
=========================================
"""

import random
from datetime import datetime, timedelta

import pandas as pd

from dataset_generator.config import (
    NUM_INTERACTIONS,
    INTERACTIONS_FILE,
    MIN_TIME_SPENT,
    MAX_TIME_SPENT,
    MATCH_INTEREST,
    RELATED_INTEREST,
    RANDOM_INTEREST,
    RELATED_CATEGORY
)


class InteractionsGenerator:

    def __init__(self, users_df, content_df):

        self.users = users_df
        self.content = content_df

        self.interactions = []

        # Cache courses by category
        self.category_courses = {}

        for category in self.content["category"].unique():

            self.category_courses[category] = self.content[
                self.content["category"] == category
            ]

    # =====================================
    # Rating
    # =====================================

    def generate_rating(self, match=True):

        if match:

            return random.choices(

                [3,4,5],

                weights=[20,40,40]

            )[0]

        return random.choices(

            [1,2,3,4,5],

            weights=[20,25,25,20,10]

        )[0]

    # =====================================
    # Time Spent
    # =====================================
    def generate_time_spent(self, rating, duration):
        if rating == 5:
                completion_ratio = random.uniform(0.85, 1.0)
        elif rating == 4:
                completion_ratio = random.uniform(0.6, 0.9)
        elif rating == 3:
                completion_ratio = random.uniform(0.35, 0.65)
        elif rating == 2:
                completion_ratio = random.uniform(0.15, 0.4)
        else:
                completion_ratio = random.uniform(0.05, 0.2)

        return int(duration * completion_ratio)

    # =====================================
    # Timestamp
    # =====================================

    def random_timestamp(self):

        start = datetime(2023,1,1)

        end = datetime(2026,1,1)

        delta = end - start

        random_days = random.randint(0, delta.days)

        random_seconds = random.randint(0,86400)

        return (

            start +

            timedelta(

                days=random_days,

                seconds=random_seconds

            )

        )

    # =====================================
    # Choose Course
    # =====================================

    def choose_course(self, interest):

        r = random.random()

        # 75%
        if r <= MATCH_INTEREST:

            pool = self.category_courses[interest]

        # 15%
        elif r <= MATCH_INTEREST + RELATED_INTEREST:

            related = RELATED_CATEGORY.get(

                interest,

                [interest]

            )

            cat = random.choice(related)

            pool = self.category_courses[cat]

        # 10%
        else:

            pool = self.content

        return pool.sample(1).iloc[0]

    # =====================================
    # Main Generator
    # =====================================

    def generate(self):

        print("Generating interactions...")

        for _ in range(NUM_INTERACTIONS):

            user = self.users.sample(1).iloc[0]

            interest = user["interest"]

            course = self.choose_course(

                interest

            )

            matched = (

                course["category"] == interest

            )

            rating = self.generate_rating(

                matched

            )

            # FIX: generate_time_spent's signature is (rating, duration) —
            # it needs the COURSE's own duration to compute a realistic
            # time_spent (e.g. 90% of a 60-minute course, not a number
            # disconnected from how long the course actually is). The
            # call here was still only passing `rating`, which throws
            # "missing 1 required positional argument: 'duration'"
            # immediately on the very first interaction generated.
            time_spent = self.generate_time_spent(

                rating,

                course["duration"]

            )

            self.interactions.append({

                "user_id":

                    int(user["user_id"]),

                "content_id":

                    int(course["content_id"]),

                "time_spent":

                    time_spent,

                "rating":

                    rating,

                "timestamp":

                    self.random_timestamp()

                    .strftime(

                        "%Y-%m-%d %H:%M:%S"

                    )

            })

        df = pd.DataFrame(

            self.interactions

        )

        df.to_csv(

            INTERACTIONS_FILE,

            index=False

        )

        print(

            f"Saved {len(df)} interactions"

        )

        return df


# =====================================
# Run
# =====================================

if __name__ == "__main__":

    users = pd.read_excel(

        "data/users.xlsx"

    )

    content = pd.read_csv(

        "data/content.csv"

    )

    generator = InteractionsGenerator(

        users,

        content

    )

    generator.generate()