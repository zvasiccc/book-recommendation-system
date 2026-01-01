from src.UsersProcessing import filter_ratings, ratings_normalization
from src.MatrixCreator import create_user_item_matrix_sparse, create_item_user_matrix_sparse
from sklearn.neighbors import NearestNeighbors 
from collections import defaultdict
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from collections import defaultdict
import numpy as np

#main function
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
            predicted_rating = score_pair[0] / score_pair[1]
            final_predictions[book_isbn] = predicted_rating

    #top n books
    top_books = sorted(
        final_predictions.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    return top_books

def ibcf_recommended_books_knn(user_id, ratings, top_n=10, k_neighbors=10):

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

    model_knn = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=k_neighbors + 1,
        n_jobs=-1
    )
    model_knn.fit(item_user_matrix)

    rated_books = item_user_matrix[:, user_pos].nonzero()[0]

    if len(rated_books) == 0:
        return []

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
    

def svd_recommended_books(user_id, ratings, top_n=10):

    # priprema podataka za Surprise
    reader = Reader(rating_scale=(ratings['Book-Rating'].min(), ratings['Book-Rating'].max()))
    data = Dataset.load_from_df(ratings[['User-ID', 'ISBN', 'Book-Rating']], reader)

    trainset = data.build_full_trainset()  # koristi sve podatke za treniranje

    # inicijalizacija i treniranje SVD modela
    algo = SVD()
    algo.fit(trainset)

    # skupljanje svih ISBN-ova koje korisnik nije ocenio
    all_books = set(ratings['ISBN'].unique())
    user_rated_books = set(ratings[ratings['User-ID'] == user_id]['ISBN'])
    books_to_predict = all_books - user_rated_books

    # predikcija ocena
    predictions = []
    for book in books_to_predict:
        pred = algo.predict(user_id, book)
        predictions.append((book, pred.est))

    # sortiranje i vraćanje top-N knjiga
    top_books = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]
    return top_books
