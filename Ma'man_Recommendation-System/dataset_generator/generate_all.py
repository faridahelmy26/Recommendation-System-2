"""
=========================================
Generate Complete Recommendation Dataset
=========================================
"""

import os

from dataset_generator.content_generator import ContentGenerator
from dataset_generator.users_generator import UsersGenerator
from dataset_generator.interactions_generator import InteractionsGenerator

from dataset_generator.config import OUTPUT_DIR


def main():

    print("=" * 60)
    print("Hybrid Recommendation Dataset Generator")
    print("=" * 60)

    # ==========================================
    # Create Folder
    # ==========================================

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ==========================================
    # Generate Content
    # ==========================================

    print("\nGenerating Courses...")

    content_generator = ContentGenerator()

    content_df = content_generator.generate()

    print(f"Content Generated : {len(content_df)}")

    # ==========================================
    # Generate Users
    # ==========================================

    print("\nGenerating Users...")

    users_generator = UsersGenerator()

    users_df = users_generator.generate()

    print(f"Users Generated : {len(users_df)}")

    # ==========================================
    # Generate Interactions
    # ==========================================

    print("\nGenerating Interactions...")

    interaction_generator = InteractionsGenerator(

        users_df,

        content_df

    )

    interactions_df = interaction_generator.generate()

    print(f"Interactions Generated : {len(interactions_df)}")

    # ==========================================
    # Summary
    # ==========================================

    print("\n" + "=" * 60)

    print("Dataset Generated Successfully")

    print("=" * 60)

    print(f"Courses      : {len(content_df)}")

    print(f"Users        : {len(users_df)}")

    print(f"Interactions : {len(interactions_df)}")

    print("\nFiles Created")

    print("data/content.csv")

    print("data/users.xlsx")

    print("data/interactions.csv")

    print("=" * 60)


if __name__ == "__main__":

    main()