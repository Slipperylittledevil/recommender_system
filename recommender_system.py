# ===== [IMPORTS] =====
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===== [STEP 2] Read the dataset =====
# on_bad_lines='skip' handles the malformed rows this Kaggle file is known for
df = pd.read_csv('books.csv', on_bad_lines='skip')

# Clean up column names (the file has stray spaces, e.g. '  num_pages')
df.columns = df.columns.str.strip()

# Make sure the numeric columns really are numeric
df['average_rating'] = pd.to_numeric(df['average_rating'], errors='coerce')
df['ratings_count'] = pd.to_numeric(df['ratings_count'], errors='coerce')
df = df.dropna(subset=['average_rating', 'ratings_count', 'title', 'authors'])


# ===== [STEP 3] Popularity-based Recommender (IMDB weighted rank) =====
def PopularityRecommender(df, n=10, quantile=0.90):
    C = df['average_rating'].mean()                    # mean rating across all books
    m = df['ratings_count'].quantile(quantile)         # min votes required to qualify

    # Only books with enough votes qualify
    qualified = df[df['ratings_count'] >= m].copy()

    def weighted_rating(row):
        v = row['ratings_count']
        R = row['average_rating']
        return (v / (v + m)) * R + (m / (v + m)) * C

    qualified['weighted_rating'] = qualified.apply(weighted_rating, axis=1)

    recommendations = qualified.sort_values('weighted_rating', ascending=False).head(n)
    return recommendations[['title', 'authors', 'average_rating',
                            'ratings_count', 'weighted_rating']].reset_index(drop=True)


# ===== [STEP 4] Content-based Recommender (TF-IDF on authors + cosine similarity) =====
# Build the TF-IDF matrix from the authors column
tfidf = TfidfVectorizer(stop_words='english')
df['authors'] = df['authors'].fillna('')
tfidf_matrix = tfidf.fit_transform(df['authors'])

# Cosine similarity between every pair of books
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Lookup table: title -> row index
indices = pd.Series(df.index, index=df['title']).drop_duplicates()


def ContentBasedRecommender(title, indices, cosine_sim, n=10):
    id_ = indices[title]

    similarities = list(enumerate(cosine_sim[id_]))
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    similarities = similarities[1:n+1]                 # skip position 0 (the book itself)

    book_indices = [i[0] for i in similarities]
    return df[['title', 'authors']].iloc[book_indices].reset_index(drop=True)


# ===== RUN BOTH RECOMMENDERS =====
print("=== Popularity-based recommendations (IMDB weighted rank) ===")
print(PopularityRecommender(df, n=10))

print("\n=== Content-based recommendations for 'The Hobbit or There and Back Again' ===")
print(ContentBasedRecommender('The Hobbit or There and Back Again', indices, cosine_sim, n=10))