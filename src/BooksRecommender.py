from src.UsersProcessing import filter_active_users_in_ratings, ratings_normalization
from src.MatrixCreator import create_user_item_matrix_sparse, create_item_user_matrix_sparse
from sklearn.neighbors import NearestNeighbors 
from collections import defaultdict
import numpy as np

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

def ibcf_recommended_books_knn(user_id, ratings, top_n=10, k_neighbors=50):

    # 1. filtriranje i normalizacija
    filtered_ratings = filter_active_users_in_ratings(ratings)
    normalized_ratings = ratings_normalization(filtered_ratings)

    # 2. item-user matrica
    item_user_matrix, book_index, user_index = create_item_user_matrix_sparse(
        normalized_ratings
    )

    if user_id not in user_index:
        raise ValueError(
            f"User with ID {user_id} does not exist or does not have enough ratings."
        )

    user_pos = np.where(user_index == user_id)[0][0]

    # 3. KNN model (item-based)
    model_knn = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=k_neighbors + 1
    )
    model_knn.fit(item_user_matrix)

    # 4. knjige koje je korisnik ocenio
    rated_books = item_user_matrix[:, user_pos].nonzero()[0]

    if len(rated_books) == 0:
        return []

    # 5. akumulacija skorova
    predicted_scores = defaultdict(lambda: [0.0, 0.0])  # [numerator, denominator]

    # 6. IBCF: iteriramo samo kroz ocenjene knjige
    for rated_book in rated_books:

        user_rating = item_user_matrix[rated_book, user_pos]

        # KNN za ovu knjigu
        distances, indices = model_knn.kneighbors(
            item_user_matrix[rated_book]
        )

        similarities = 1 - distances.flatten()[1:]
        neighbors = indices.flatten()[1:]

        for sim, neighbor_book in zip(similarities, neighbors):

            # preskačemo knjige koje je korisnik već ocenio
            if neighbor_book in rated_books:
                continue

            if sim <= 0:
                continue

            predicted_scores[neighbor_book][0] += sim * user_rating
            predicted_scores[neighbor_book][1] += abs(sim)

    # 7. finalni skor
    final_predictions = {
        book_index[book_idx]: num / den
        for book_idx, (num, den) in predicted_scores.items()
        if den > 0
    }

    # 8. sortiranje i top-N
    return sorted(
        final_predictions.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]