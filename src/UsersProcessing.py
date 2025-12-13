from src.GlobalVariables import USER_BASED_THRESHOLD_PERCENTILE, TOP_N_RECOMMENDATIONS, MIN_NUBMER_OF_RATINGS
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors 


def preprocess_users(users,ratings):
    users = users.dropna(subset=["User-ID"])
    users = users[users["User-ID"].apply(lambda x: str(x).isdigit())]

    users = users.drop_duplicates(subset=["User-ID"],keep="first")

    users_with_ratings = ratings["User-ID"].unique()
    users = users[users["User-ID"].isin(users_with_ratings)].reset_index(drop=True)

    return users



def ubcf_recommended_books_knn(ratings, top_n, k_neighbors=50):
    
    # 1. Filtriraj aktivne korisnike
    filtered_ratings = filter_active_users_in_ratings(ratings)
    
    # 2. Normalizacija ocena (mean-centered)
    normalized_ratings = ratings_normalization(filtered_ratings)
    
    # 3. Kreiranje Sparse matrice
    user_item_matrix_sparse, user_index, book_index = create_user_item_matrix_sparse(normalized_ratings)
    
    N_users = len(user_index)
    recommendations = {}

    model_knn = NearestNeighbors(metric='cosine', algorithm='ball_tree', n_neighbors=k_neighbors + 1, n_jobs=-1)
    print("Obučavam KNN model...")
    model_knn.fit(user_item_matrix_sparse)
    
    print("Pokrećem KNN predikciju (iterativno)...")
    
    for i, user_id in enumerate(user_index):
        
        distances, indices = model_knn.kneighbors(user_item_matrix_sparse[i], n_neighbors=k_neighbors + 1)
        
      
        similarities = 1 - distances.flatten()[1:]
        neighbor_indices = indices.flatten()[1:]
        
    
        neighbor_ratings = user_item_matrix_sparse[neighbor_indices]
        
       
        rated_books_indices = user_item_matrix_sparse[i].nonzero()[1]
        
        predicted_scores = {}
        
        
        for sim, neighbor_idx in zip(similarities, neighbor_indices):
          
            book_indices_rated_by_neighbor = user_item_matrix_sparse[neighbor_idx].nonzero()[1]
            
            unrated_for_this_user = np.setdiff1d(book_indices_rated_by_neighbor, rated_books_indices)

            for book_idx in unrated_for_this_user:

                rating_value = user_item_matrix_sparse[neighbor_idx, book_idx]
                
                if book_idx not in predicted_scores:
                    predicted_scores[book_idx] = [0.0, 0.0]
                    
                predicted_scores[book_idx][0] += sim * rating_value
                predicted_scores[book_idx][1] += abs(sim)

        

        final_predictions = {
            book_index[idx]: score[0] / score[1]
            for idx, score in predicted_scores.items() if score[1] > 0
        }
        

        top_books_scores = sorted(final_predictions.items(), key=lambda item: item[1], reverse=True)[:top_n]
        recommendations[user_id] = [isbn for isbn, score in top_books_scores]

        if (i + 1) % 100 == 0:
            print(f"  ... Obrađeno {i+1}/{N_users} korisnika.")
    
    return recommendations

def filter_active_users_in_ratings(ratings):
    user_ratings_number = ratings.groupby("User-ID").size()
    threshold = max(user_ratings_number.quantile(USER_BASED_THRESHOLD_PERCENTILE), MIN_NUBMER_OF_RATINGS)
    active_users = user_ratings_number[user_ratings_number >= threshold].index
    filtered_ratings = ratings[ratings["User-ID"].isin(active_users)].reset_index(drop=True)
    

    book_counts = filtered_ratings.groupby("ISBN").size()
    popular_books = book_counts[book_counts >= 5].index
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
    
    user_map = {user: i for i, user in enumerate(user_index)}
    book_map = {book: i for i, book in enumerate(book_index)}
    
    rows = ratings["User-ID"].map(user_map).to_numpy()
    cols = ratings["ISBN"].map(book_map).to_numpy()
    data = ratings["Book-Rating-Normalized"].to_numpy()
    
    user_item_matrix = csr_matrix((data, (rows, cols)), shape=(len(user_index), len(book_index)))
    
    return user_item_matrix, user_index, book_index


