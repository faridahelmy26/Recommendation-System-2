"""
=========================================
Users Generator
Generate 5000 Realistic Users
=========================================
"""

import random
import pandas as pd

from dataset_generator.config import (
    NUM_USERS,
    USERS_FILE,
    MIN_AGE,
    MAX_AGE,
    CATEGORIES,
    LEVELS,
    LEARNING_STYLES
)


class UsersGenerator:

    def __init__(self):

        self.users = []

    # ==========================================
    # Age Distribution
    # ==========================================

    def generate_age(self):

        r = random.random()

        if r < 0.20:
            return random.randint(16, 20)

        elif r < 0.55:
            return random.randint(21, 30)

        elif r < 0.80:
            return random.randint(31, 40)

        else:
            return random.randint(41, 55)

    # ==========================================
    # Learning Level
    # ==========================================

    def generate_level(self, age):

        if age <= 20:

            weights = [60, 30, 10]

        elif age <= 30:

            weights = [35, 45, 20]

        elif age <= 40:

            weights = [20, 50, 30]

        else:

            weights = [15, 40, 45]

        return random.choices(

            LEVELS,

            weights=weights

        )[0]

    # ==========================================
    # Interest
    # ==========================================

    def generate_interest(self):

        return random.choice(CATEGORIES)

    # ==========================================
    # Learning Style
    # ==========================================

    def generate_learning_style(self):

        return random.choices(

            LEARNING_STYLES,

            weights=[35, 25, 20, 20]

        )[0]

    # ==========================================
    # Single User
    # ==========================================

    def create_user(self, user_id):

        age = self.generate_age()

        return {

            "user_id": user_id,

            "age": age,

            "interest": self.generate_interest(),

            "level": self.generate_level(age),

            "learning_style": self.generate_learning_style()

        }

    # ==========================================
    # Dataset
    # ==========================================

    def generate(self):

        print("Generating Users Dataset...")

        for user_id in range(1, NUM_USERS + 1):

            self.users.append(

                self.create_user(user_id)

            )

        df = pd.DataFrame(self.users)

        df.to_excel(

            USERS_FILE,

            index=False

        )

        print(f"Saved {len(df)} users")

        return df


# ==========================================
# Run
# ==========================================

if __name__ == "__main__":

    generator = UsersGenerator()

    generator.generate()