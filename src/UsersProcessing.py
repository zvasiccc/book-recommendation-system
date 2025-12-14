from src.GlobalVariables import USER_BASED_THRESHOLD_PERCENTILE, TOP_N_RECOMMENDATIONS, MIN_NUBMER_OF_RATINGS
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors 
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



# --- Pomoćne Funkcije za Pripremu Podataka ---

def preprocess_users(users, ratings):
    """
    Čisti i filtrira podatke o korisnicima.
    """
    users = users.dropna(subset=["User-ID"])
    # Osigurava da su User-ID-ovi numerički
    users = users[users["User-ID"].apply(lambda x: str(x).isdigit())]

    users = users.drop_duplicates(subset=["User-ID"], keep="first")

    # Zadržava samo korisnike koji su zaista ocenjivali
    users_with_ratings = ratings["User-ID"].unique()
    users = users[users["User-ID"].isin(users_with_ratings)].reset_index(drop=True)

    return users

def filter_active_users_in_ratings(ratings):
    """
    Filtrira aktivne korisnike (koji su ocenili više od praga) i popularne knjige.
    """
    user_ratings_number = ratings.groupby("User-ID").size()
    # Računanje praga (veći od percentila i MIN_NUBMER_OF_RATINGS)
    threshold = max(user_ratings_number.quantile(USER_BASED_THRESHOLD_PERCENTILE), MIN_NUBMER_OF_RATINGS)
    active_users = user_ratings_number[user_ratings_number >= threshold].index
    filtered_ratings = ratings[ratings["User-ID"].isin(active_users)].reset_index(drop=True)
    
    # Filtriranje popularnih knjiga (Opciono: Smanjuje dimenzionalnost matrice)
    book_counts = filtered_ratings.groupby("ISBN").size()
    popular_books = book_counts[book_counts >= 20].index
    filtered_ratings = filtered_ratings[filtered_ratings["ISBN"].isin(popular_books)].reset_index(drop=True)
    
    return filtered_ratings


def ratings_normalization(ratings):
    """
    Mean-centered normalizacija ocena korisnika (R_norm = R - R_avg).
    """
    user_mean = ratings.groupby("User-ID")["Book-Rating"].mean()
    ratings["Book-Rating-Normalized"] = ratings.apply(
        lambda row: row["Book-Rating"] - user_mean[row["User-ID"]],
        axis=1
    )
    return ratings


def create_user_item_matrix_sparse(ratings):
    """
    Kreira Sparse (CSR) matricu korisnik × knjiga, kao i indekse mapiranja.
    """
    user_index = ratings["User-ID"].unique()
    book_index = ratings["ISBN"].unique()
    
    user_map = {user: i for i, user in enumerate(user_index)}
    book_map = {book: i for i, book in enumerate(book_index)}
    
    rows = ratings["User-ID"].map(user_map).to_numpy()
    cols = ratings["ISBN"].map(book_map).to_numpy()
    data = ratings["Book-Rating-Normalized"].to_numpy()
    
    # Kreiranje CSR matrice
    user_item_matrix = csr_matrix((data, (rows, cols)), shape=(len(user_index), len(book_index)))
    
    return user_item_matrix, user_index, book_index

# --- Glavni Algoritam UBCF-KNN (Ispravljen i Optimizovan) ---

def ubcf_recommended_books_knn(ratings, top_n, k_neighbors=10):
    
    # 1. & 2. Priprema podataka
    filtered_ratings = filter_active_users_in_ratings(ratings)
    normalized_ratings = ratings_normalization(filtered_ratings)
    
    # 3. Kreiranje Sparse matrice i mapiranje
    user_item_matrix_sparse, user_index, book_index = create_user_item_matrix_sparse(normalized_ratings)
    
    N_users = len(user_index)
    recommendations = {}


    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=k_neighbors + 1, n_jobs=1)
    print("Obučavam KNN model...")
    model_knn.fit(user_item_matrix_sparse)
    

    user_rated_books_sets = {i: set(user_item_matrix_sparse[i].indices) for i in range(N_users)}

    for i, user_id in enumerate(user_index):
        
        # A. Pronalaženje K najsličnijih suseda
        distances, indices = model_knn.kneighbors(user_item_matrix_sparse[i], n_neighbors=k_neighbors + 1)
        similarities = 1 - distances.flatten()[1:] # Konverzija udaljenosti u sličnost (1 - distanca)
        neighbor_indices = indices.flatten()[1:]
        
        # B. Inicijalizacija
        
        predicted_scores = defaultdict(lambda: [0.0, 0.0])
        rated_books = user_rated_books_sets[i]
        
        for sim, neighbor_idx in zip(similarities, neighbor_indices):
            neighbor_row = user_item_matrix_sparse[neighbor_idx]
            for j, book_idx in enumerate(neighbor_row.indices):
                if book_idx not in rated_books:
                    predicted_scores[book_idx][0] += sim * neighbor_row.data[j]
                    predicted_scores[book_idx][1] += abs(sim)

        # Compute final weighted prediction
        final_scores = {book_index[idx]: s[0]/s[1] for idx, s in predicted_scores.items() if s[1] > 0}

        # Select Top-N
        top_books = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        recommendations[user_id] = [isbn for isbn, score in top_books]

        if (i+1) % 100 == 0:
            print(f"Processed {i+1}/{N_users} users.")
    
    return recommendations