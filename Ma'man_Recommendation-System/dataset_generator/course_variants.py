# NOTE: LEVEL_VARIANTS was removed on purpose. It used to be a second,
# independent Beginner/Intermediate/Advanced list that content_generator.py
# sampled separately from the real `level` field, so a course could end up
# with level="Advanced" in the data while its title said "- Beginner".
# content_generator.py now builds the level text directly from the actual
# `level` value instead, so there's nothing left to keep in sync here.

FORMAT_VARIANTS = [
    "",
    "Bootcamp",
    "Complete Course",
    "Masterclass",
    "Professional Certificate",
    "Practical Workshop",
    "Hands-on Training",
    "Project-Based Learning",
    "Essentials",
    "Fundamentals"
]

YEAR_VARIANTS = [
    "",
    "2026 Edition",
    "Updated Edition",
    "Latest Version"
]

DESCRIPTION_TEMPLATES = [

    "Learn {title} through interactive lessons and practical activities designed for children and educators.",

    "Master {title} with engaging projects, real-life examples, and step-by-step guidance.",

    "Develop essential skills in {title} using fun educational content suitable for young learners.",

    "Explore {title} with hands-on exercises, quizzes, and creative classroom activities.",

    "Build confidence in {title} through practical learning experiences and modern teaching techniques.",

    "Discover the foundations of {title} with easy-to-follow lessons and interactive projects.",

    "Gain real-world experience in {title} through engaging educational activities.",

    "Improve your understanding of {title} with modern learning methods and practical applications."

]