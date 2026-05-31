import pandas as pd

class ContentRecommender:
    def __init__(self, path):
        self.df = pd.read_csv(path)

    def recommend_with_input(self, title, k=5):
        row = df[df['title'].str.lower().str.contains(title.lower().strip())]
        if row.empty:
            return None

        # dummy example (عدليه لاحقًا)
        recs = self.df.sample(k)['title'].tolist()

        return {
            "input": title,
            "recommendations": recs
        }