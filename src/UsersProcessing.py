from src.GlobalVariables import USER_BASED_THRESHOLD_PERCENTILE, TOP_N_RECOMMENDATIONS, MIN_NUBMER_OF_RATINGS, MIN_NUBMER_OF_BOOK_RATINGS
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors 
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def preprocess_users(users, ratings):
    users = users.dropna(subset=["User-ID"])
    users = users[users["User-ID"].apply(lambda x: str(x).isdigit())]

    users = users.drop_duplicates(subset=["User-ID"], keep="first")

    users_with_ratings = ratings["User-ID"].unique()
    users = users[users["User-ID"].isin(users_with_ratings)].reset_index(drop=True)

    return users

def filter_active_users_in_ratings(ratings):

    user_ratings_number = ratings.groupby("User-ID").size()
    threshold = max(user_ratings_number.quantile(USER_BASED_THRESHOLD_PERCENTILE), MIN_NUBMER_OF_RATINGS)
    active_users = user_ratings_number[user_ratings_number >= threshold].index
    filtered_ratings = ratings[ratings["User-ID"].isin(active_users)].reset_index(drop=True)
    
    book_counts = filtered_ratings.groupby("ISBN").size()
    popular_books = book_counts[book_counts >= MIN_NUBMER_OF_BOOK_RATINGS].index
    filtered_ratings = filtered_ratings[filtered_ratings["ISBN"].isin(popular_books)].reset_index(drop=True)
    
    return filtered_ratings


def ratings_normalization(ratings):

    user_mean = ratings.groupby("User-ID")["Book-Rating"].mean()
    ratings["Book-Rating-Normalized"] = ratings.apply(
        lambda row: row["Book-Rating"] - user_mean[row["User-ID"]],
        axis=1
    )
    return ratings


def create_user_item_matrix_sparse(ratings):

    user_index = ratings["User-ID"].unique()
    book_index = ratings["ISBN"].unique()

    #create dictionaries for users and books
    user_map = {user: i for i, user in enumerate(user_index)}
    book_map = {book: i for i, book in enumerate(book_index)}
    
    rows = ratings["User-ID"].map(user_map).to_numpy()
    cols = ratings["ISBN"].map(book_map).to_numpy()
    data = ratings["Book-Rating-Normalized"].to_numpy()

    #sparse matrix but only explicit ratings are actually saved, implicit ratings are 0 and they are not saved in order to save memory
    user_item_matrix = csr_matrix((data, (rows, cols)), shape=(len(user_index), len(book_index)))
    
    return user_item_matrix, user_index, book_index


#main function
def ubcf_recommended_books_knn(
    user_id,
    ratings,
    top_n=10,
    k_neighbors=50
):

    filtered_ratings = filter_active_users_in_ratings(ratings)
    normalized_ratings = ratings_normalization(filtered_ratings)

    user_item_matrix, user_index, book_index = create_user_item_matrix_sparse(normalized_ratings)

    if user_id not in user_index:
        raise ValueError(f"User with ID {user_id} does not exists or does not have enough given ratings to recommend book for him.")

    #position in matrix of a user with user_id
    user_pos = np.where(user_index == user_id)[0][0]

    model_knn = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=k_neighbors + 1,
        n_jobs=1
    )
    model_knn.fit(user_item_matrix)

    #distances and indexes of user's neighbors
    distances, indices = model_knn.kneighbors(user_item_matrix[user_pos])
    
    similarities = 1 - distances.flatten()[1:]
    neighbor_indices = indices.flatten()[1:]

    #already rated books of our particular user
    rated_books = set(user_item_matrix[user_pos].indices)

    predicted_scores = defaultdict(lambda: [0.0, 0.0])

    #prediction:
    for sim, neighbor_idx in zip(similarities, neighbor_indices):
        neighbor_books = user_item_matrix[neighbor_idx].indices

        for book_idx in neighbor_books:
            if book_idx in rated_books:
                continue

            rating = user_item_matrix[neighbor_idx, book_idx]
            predicted_scores[book_idx][0] += sim * rating
            predicted_scores[book_idx][1] += abs(sim)

    #final 
    final_predictions = {
        book_index[b]: s[0] / s[1]
        for b, s in predicted_scores.items()
        if s[1] > 0
    }

    #top n books
    top_books = sorted(
        final_predictions.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    return top_books