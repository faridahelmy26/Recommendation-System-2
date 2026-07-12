"""
=========================================
Course Description Generator
=========================================
"""

import random

# =====================================================
# Beginner Templates
# =====================================================

BEGINNER = [

"Learn the fundamentals of {title} from scratch with practical examples and step-by-step explanations.",

"Perfect for beginners who want to build a strong foundation in {category}.",

"Start your learning journey in {category} through hands-on projects and real-world examples.",

"Master the basics of {title} using simple explanations and interactive exercises.",

"This beginner-friendly course introduces the essential concepts required to succeed in {category}.",

"Develop your first practical projects while learning the core concepts of {title}.",

"Understand key principles and build confidence through guided exercises and quizzes.",

"Gain practical skills in {title} with real-life case studies and beginner projects."

]

# =====================================================
# Intermediate Templates
# =====================================================

INTERMEDIATE = [

"Strengthen your knowledge of {title} by solving practical industry-focused projects.",

"Expand your experience in {category} using modern tools and best practices.",

"Learn advanced workflows and techniques used by professionals working in {category}.",

"Improve your technical skills with hands-on labs, assignments, and real-world applications.",

"Explore intermediate concepts in {title} while building portfolio-ready projects.",

"Practice real-world scenarios and improve your problem-solving skills in {category}.",

"Develop scalable solutions using modern frameworks and professional development practices.",

"Apply your existing knowledge to build complete projects from start to finish."

]

# =====================================================
# Advanced Templates
# =====================================================

ADVANCED = [

"Master advanced concepts in {title} through real-world enterprise applications.",

"Build production-ready systems while learning professional development techniques.",

"Work with large-scale projects and advanced design strategies in {category}.",

"Explore optimization, scalability, and best engineering practices in {title}.",

"Develop enterprise-grade applications using industry standards and modern architectures.",

"Gain expert-level knowledge through challenging case studies and complex projects.",

"Understand advanced implementation techniques used in professional environments.",

"Prepare for senior-level positions by mastering the latest technologies in {category}."

]

# =====================================================
# Skills
# =====================================================

SKILLS = [

"problem solving",

"critical thinking",

"software development",

"system design",

"project implementation",

"data analysis",

"debugging",

"best practices",

"team collaboration",

"real-world deployment"

]

# =====================================================
# Learning Outcomes
# =====================================================

OUTCOMES = [

"You will build real projects throughout the course.",

"You will gain hands-on experience using industry-standard tools.",

"You will be able to apply your knowledge to practical business scenarios.",

"You will improve your technical and analytical thinking.",

"You will develop portfolio-ready applications.",

"You will understand best practices followed by professional engineers.",

"You will be prepared for technical interviews and real-world challenges.",

"You will confidently work on real software projects."

]

# =====================================================
# Generator
# =====================================================

def generate_description(title, category, level):

    if level == "Beginner":
        template = random.choice(BEGINNER)

    elif level == "Intermediate":
        template = random.choice(INTERMEDIATE)

    else:
        template = random.choice(ADVANCED)

    description = template.format(

        title=title,

        category=category

    )

    description += " "

    description += "Skills covered include "

    description += ", ".join(random.sample(SKILLS, 3))

    description += ". "

    description += random.choice(OUTCOMES)

    return description