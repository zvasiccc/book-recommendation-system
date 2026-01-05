from src.MatrixCreator import create_user_item_matrix_sparse
from sklearn.neighbors import NearestNeighbors 
from collections import defaultdict
from sklearn.metrics import mean_squared_error
import numpy as np

def ubcf_recommended_books_knn(user_id, ratings, top_n=10, k_neighbors=50):

    user_item_matrix, user_index, book_index = create_user_item_matrix_sparse(ratings)

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

            rating_diff = user_item_matrix[neighbor_index, book_idx]
            if rating_diff == 0:
                continue
            predicted_scores[book_idx][0] += neighbor_sim * rating_diff 
            predicted_scores[book_idx][1] += abs(neighbor_sim)

    user_mean = ratings[ratings["User-ID"] == user_id]["user_mean"].iloc[0]
    #final predictions
    final_predictions = {}
    for book_idx,score_pair in predicted_scores.items():
        if score_pair[1]>0:
            book_isbn= book_index[book_idx]
            predicted_rating = score_pair[0] / score_pair[1] + user_mean 
            final_predictions[book_isbn] = max(1.0, min(10.0, predicted_rating))
    #top n books
    top_books = sorted(
        final_predictions.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    return top_books


# def ubcf_predict_for_rmse(user_id, ratings, k_neighbors=50):
#     user_item_matrix, user_index, book_index = create_user_item_matrix_sparse(ratings)

#     if user_id not in user_index:
#         raise ValueError(f"User with ID {user_id} does not exists or does not have enough given ratings to recommend book for him.")

#     user_pos = np.where(user_index == user_id)[0][0]

#     model_knn = NearestNeighbors(
#         metric="cosine",
#         algorithm="brute",
#         n_neighbors=k_neighbors + 1,
#         n_jobs=-1
#     )
#     model_knn.fit(user_item_matrix)

#     distances, indices = model_knn.kneighbors(user_item_matrix[user_pos])
#     similarities = 1 - distances.flatten()[1:]
#     neighbor_indices = indices.flatten()[1:]

#     rated_books = user_item_matrix[user_pos].indices
#     predicted_scores = {}
#     user_mean = ratings[ratings["User-ID"] == user_id]["user_mean"].iloc[0]
#     print(user_mean)
#     for book_idx in rated_books:
#         temp_user_vector = original_user_vector.copy()
#         temp_user_vector[0, book_idx] = 0
#         temp_user_vector.eliminate_zeros()
        
#         numerator = 0.0
#         denominator = 0.0
#         for neighbor_sim, neighbor_index in zip(similarities, neighbor_indices):
#             rating = user_item_matrix[neighbor_index, book_idx]
#             if rating == 0:
#                 continue
#             numerator += neighbor_sim * rating 
#             denominator += abs(neighbor_sim)
#         if denominator > 0:
#             predicted_scores[book_index[book_idx]] = numerator / denominator + user_mean

#     return predicted_scores 

def ubcf_predict_for_rmse(user_id, ratings, k_neighbors=50):
    user_item_matrix, user_index, book_index = create_user_item_matrix_sparse(ratings)
    user_pos = np.where(user_index == user_id)[0][0]
    
    # Originalni vektor korisnika
    original_user_vector = user_item_matrix[user_pos].copy()
    rated_books = original_user_vector.indices
    
    predicted_scores = {}
    user_mean = ratings[ratings["User-ID"] == user_id]["user_mean"].iloc[0]

    # Model fitujemo na CELU matricu
    model_knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=k_neighbors + 1)
    model_knn.fit(user_item_matrix)

    for book_idx in rated_books:
        # --- LOO DEO ---
        # Privremeno ukloni ocenu za knjigu koju predviđamo da ne bi uticala na sličnost
        temp_user_vector = original_user_vector.copy()
        temp_user_vector[0, book_idx] = 0
        temp_user_vector.eliminate_zeros()

        # Nađi komšije na osnovu "okrnjenog" vektora
        distances, indices = model_knn.kneighbors(temp_user_vector)
        similarities = 1 - distances.flatten()[1:]
        neighbor_indices = indices.flatten()[1:]

        numerator = 0.0
        denominator = 0.0

        for neighbor_sim, neighbor_index in zip(similarities, neighbor_indices):
            # Uzmi odstupanje komšije za TU knjigu
            neighbor_diff = user_item_matrix[neighbor_index, book_idx]
            
            if neighbor_diff != 0:
                numerator += neighbor_sim * neighbor_diff
                denominator += abs(neighbor_sim)
        
        if denominator > 0:
            pred = (numerator / denominator) + user_mean
            predicted_scores[book_index[book_idx]] = max(1.0, min(10.0, pred))

    return predicted_scores

def ubcf_evaluation_rmse(user_id, ratings):
    ubcf_predictions = ubcf_predict_for_rmse(user_id, ratings)
    true_ratings = ratings[ratings['User-ID'] == user_id].set_index('ISBN')['Book-Rating'].to_dict()

    common_books = set(ubcf_predictions.keys()).intersection(true_ratings.keys())
    actual_ratings = [true_ratings[b] for b in common_books]
    predicted_ratings = [ubcf_predictions[b] for b in common_books]

    rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
    return rmse