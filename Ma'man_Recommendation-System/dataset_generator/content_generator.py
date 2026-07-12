"""
=========================================
Content Generator
Generate 2000 Realistic Courses
=========================================
"""

import random
import pandas as pd
from dataset_generator.course_variants import (
    FORMAT_VARIANTS,
    YEAR_VARIANTS
)

from dataset_generator.config import (
    NUM_CONTENT,
    CATEGORIES,
    LEVELS,
    LEVEL_TO_DIFFICULTY,
    MIN_DURATION,
    MAX_DURATION,
    MIN_RATING,
    MAX_RATING,
    CONTENT_FILE
)

from dataset_generator.titles import COURSES
from dataset_generator.descriptions import generate_description


class ContentGenerator:

    def __init__(self):
        self.content = []

    # ==================================================
    # Generate Rating
    # ==================================================

    def generate_rating(self, level):

        if level == "Beginner":
            mean = 4.5

        elif level == "Intermediate":
            mean = 4.3

        else:
            mean = 4.1

        rating = random.gauss(mean, 0.25)

        rating = max(MIN_RATING, rating)
        rating = min(MAX_RATING, rating)

        return round(rating, 2)

    # ==================================================
    # Generate Duration
    # ==================================================
    def generate_duration(self, level):

        if level == "Beginner":
               return random.choice([30,45,60,75,90])

        elif level == "Intermediate":
               return random.choice([60,90,120,150])

        else:
               return random.choice([120,150,180,210,240])
    # ==================================================
    # Create Single Course
    # ==================================================
    def create_course(self, content_id, category):

        # اسم الكورس الأساسي
        base_title = random.choice(COURSES[category])

        # مستوى الكورس (المصدر الوحيد للمستوى — العنوان والعمود
        # لازم ياخدوا القيمة دي بالظبط، مش قيمة عشوائية تانية)
        level = random.choices(
              LEVELS,
              weights=[40, 35, 25]
        )[0]

        # نسخة مختلفة من الاسم
        format_variant = random.choice(FORMAT_VARIANTS)
        year_variant = random.choice(YEAR_VARIANTS)

        title = base_title

        # FIX: this used to pick an INDEPENDENT random level_variant from
        # LEVEL_VARIANTS (a separate Beginner/Intermediate/Advanced list),
        # completely disconnected from the `level` value above. That meant
        # a course could end up with level="Advanced" in the data column
        # while its title literally read "... - Beginner" — real example
        # seen in testing: title said "beginner" while level column said
        # "Intermediate". The title must always reflect the SAME level
        # that was actually generated, never a second independent draw.
        # Made this optional (not always appended) just like the original
        # intent of "level_variant" being sometimes present, but now it's
        # never contradictory.
        if random.choice([True, False]):
              title += f" - {level}"

        if format_variant:
              title += f" | {format_variant}"

        if year_variant:
              title += f" ({year_variant})"

        difficulty = LEVEL_TO_DIFFICULTY[level]

        duration = self.generate_duration(level)

        rating = self.generate_rating(level)

        description = generate_description(
              base_title,
              category,
              level
        )

        return {

              "content_id": content_id,

              "title": title,

              "description": description,

              "category": category,

              "level": level,

              "duration": duration,

              "difficulty": difficulty,

              "rating": rating

        }

    # ==================================================
    # Generate Dataset
    # ==================================================

    def generate(self):

        print("Generating Content Dataset...")

        content_id = 1

        per_category = NUM_CONTENT // len(CATEGORIES)

        for category in CATEGORIES:

            for _ in range(per_category):

                self.content.append(

                    self.create_course(

                        content_id,

                        category

                    )

                )

                content_id += 1

        # لو العدد مش قابل للقسمة
        while len(self.content) < NUM_CONTENT:

            category = random.choice(CATEGORIES)

            self.content.append(

                self.create_course(

                    content_id,

                    category

                )

            )

            content_id += 1

        df = pd.DataFrame(self.content)

        # ترتيب الأعمدة
        df = df[

            [

                "content_id",

                "title",

                "description",

                "category",

                "level",

                "duration",

                "difficulty",

                "rating"

            ]

        ]

        df.to_csv(

            CONTENT_FILE,

            index=False

        )

        print(f"Saved {len(df)} courses")

        return df


# ==================================================
# Run
# ==================================================

if __name__ == "__main__":

    generator = ContentGenerator()

    generator.generate()