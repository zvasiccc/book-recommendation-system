from src.UsersProcessing import filter_ratings, ratings_normalization
from src.MatrixCreator import create_item_user_matrix_sparse
from sklearn.neighbors import NearestNeighbors 
from collections import defaultdict
from collections import defaultdict
from sklearn.metrics import mean_squared_error
import numpy as np

def ibcf_recommended_books_knn(user_id, ratings, top_n=10, k_neighbors=50):

    #filtering and normalization
    filtered_ratings = filter_ratings(ratings)
    normalized_ratings = ratings_normalization(filtered_ratings)

    #item-user matrix
    item_user_matrix, book_index, user_index = create_item_user_matrix_sparse(
        normalized_ratings
    )

    if user_id not in user_index:
        raise ValueError(f"User with ID {user_id} does not exists or does not have enough given ratings to recommend book for him.")

    user_pos = np.where(user_index == user_id)[0][0]
    
    rated_books = item_user_matrix[:, user_pos].nonzero()[0]

    if len(rated_books) == 0:
        return []

    model_knn = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=k_neighbors + 1,
        n_jobs=-1
    )
    model_knn.fit(item_user_matrix)

    predicted_scores = defaultdict(lambda: [0.0, 0.0]) 

    for rated_book in rated_books:  
        user_rating = item_user_matrix[rated_book, user_pos]

        distances, indices = model_knn.kneighbors(item_user_matrix[rated_book])

        similarities = 1 - distances.flatten()[1:]
        neighbor_indices = indices.flatten()[1:]

        for neighbor_sim, neighbor_index in zip(similarities, neighbor_indices):

            if neighbor_index in rated_books:
                continue

            if neighbor_sim <= 0:    
                continue

            predicted_scores[neighbor_index][0] += neighbor_sim * user_rating
            predicted_scores[neighbor_index][1] += abs(neighbor_sim)


    final_predictions = {}  

    for neighbor_index, scores in predicted_scores.items():
        numerator = scores[0]    
        denominator = scores[1]  

        #avoid dividing with zero
        if denominator > 0:
            predicted_rating = numerator / denominator 
            final_predictions[book_index[neighbor_index]] = predicted_rating

    return sorted(
        final_predictions.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]
    
def ibcf_predict_for_rmse(user_id, ratings, k_neighbors=10):
    #filtering and normalization
    filtered_ratings = filter_ratings(ratings)
    normalized_ratings = ratings_normalization(filtered_ratings)

    #item-user matrix
    item_user_matrix, book_index, user_index = create_item_user_matrix_sparse(normalized_ratings)

    if user_id not in user_index:
        raise ValueError(f"User with ID {user_id} does not exists or does not have enough given ratings to recommend book for him.")

    user_pos = np.where(user_index == user_id)[0][0]
    
    rated_books = item_user_matrix[:, user_pos].nonzero()[0]
    
    if len(rated_books) == 0:
        return []

    model_knn = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=k_neighbors + 1,
        n_jobs=-1
    )
    model_knn.fit(item_user_matrix)
    
    predicted_scores = {}

    for rated_book in rated_books:
        numerator = 0.0
        denominator = 0.0
        user_rating = item_user_matrix[rated_book, user_pos]

        distances, indices = model_knn.kneighbors(item_user_matrix[rated_book])
        similarities = 1 - distances.flatten()[1:]
        neighbor_indices = indices.flatten()[1:]

        for neighbor_sim, neighbor_index in zip(similarities, neighbor_indices):
            numerator += neighbor_sim * user_rating
            denominator += abs(neighbor_sim)

        if denominator > 0:
            predicted_scores[book_index[rated_book]] = numerator / denominator

    return predicted_scores  

    
def ibcf_evaluation_rmse(user_id,ratings):
    ibcf_predictions = ibcf_predict_for_rmse(user_id, ratings)
    true_ratings = ratings[ratings['User-ID'] == user_id].set_index('ISBN')['Book-Rating'].to_dict()
    
    common_books = set(ibcf_predictions.keys()).intersection(true_ratings.keys())
    actual_ratings = [true_ratings[b] for b in common_books]
    predicted_ratings = [ibcf_predictions[b] for b in common_books]

    rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
    return rmse