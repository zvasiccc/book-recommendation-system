from src.UsersProcessing import filter_ratings, ratings_normalization
from src.MatrixCreator import create_user_item_matrix_sparse
from sklearn.neighbors import NearestNeighbors 
from collections import defaultdict
from collections import defaultdict
from sklearn.metrics import mean_squared_error
import numpy as np

def ubcf_recommended_books_knn(
    user_id,
    ratings,
    top_n=10,
    k_neighbors=50
):

    #ratings filtering and normalization
    filtered_ratings = filter_ratings(ratings)
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
        n_jobs=-1
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
    for neighbor_sim, neighbor_index in zip(similarities, neighbor_indices):
        neighbor_books = user_item_matrix[neighbor_index].indices

        for book_idx in neighbor_books:
            if book_idx in rated_books:
                continue

            rating = user_item_matrix[neighbor_index, book_idx]
            predicted_scores[book_idx][0] += neighbor_sim * rating
            predicted_scores[book_idx][1] += abs(neighbor_sim)

    #final predictions
    final_predictions = {}
    for book_idx,score_pair in predicted_scores.items():
        if score_pair[1]>0:
            book_isbn= book_index[book_idx]
            predicted_rating = score_pair[0] / score_pair[1] #weighted average
            final_predictions[book_isbn] = predicted_rating

    #top n books
    top_books = sorted(
        final_predictions.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    return top_books


def ubcf_predict_for_rmse(user_id, ratings, k_neighbors=50):
    filtered_ratings = filter_ratings(ratings)
    normalized_ratings = ratings_normalization(filtered_ratings)
    user_item_matrix, user_index, book_index = create_user_item_matrix_sparse(normalized_ratings)

    if user_id not in user_index:
        raise ValueError(f"User with ID {user_id} does not exists or does not have enough given ratings to recommend book for him.")

    user_pos = np.where(user_index == user_id)[0][0]

    model_knn = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=k_neighbors + 1,
        n_jobs=-1
    )
    model_knn.fit(user_item_matrix)

    distances, indices = model_knn.kneighbors(user_item_matrix[user_pos])
    similarities = 1 - distances.flatten()[1:]
    neighbor_indices = indices.flatten()[1:]

    rated_books = user_item_matrix[user_pos].indices
    predicted_scores = {}

    for book_idx in rated_books:
        numerator = 0.0
        denominator = 0.0
        for neighbor_sim, neighbor_index in zip(similarities, neighbor_indices):
            rating = user_item_matrix[neighbor_index, book_idx]
            numerator += neighbor_sim * rating
            denominator += abs(neighbor_sim)
        if denominator > 0:
            predicted_scores[book_index[book_idx]] = numerator / denominator

    return predicted_scores 

def ubcf_evaluation_rmse(user_id, ratings):
    ubcf_predictions = ubcf_predict_for_rmse(user_id, ratings)
    true_ratings = ratings[ratings['User-ID'] == user_id].set_index('ISBN')['Book-Rating'].to_dict()

    common_books = set(ubcf_predictions.keys()).intersection(true_ratings.keys())
    actual_ratings = [true_ratings[b] for b in common_books]
    predicted_ratings = [ubcf_predictions[b] for b in common_books]

    rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
    return rmse