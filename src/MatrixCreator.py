from scipy.sparse import csr_matrix
from src.GlobalVariables import USER_PERCENTILE, BOOK_PERCENTILE
import numpy as np

def prepare_data_for_matrix_sparse(ratings):
    
    user_indices = ratings["User-ID"].unique()
    book_indices = ratings["ISBN"].unique()

    #create dictionaries for users and books
    user_map = {user: i for i, user in enumerate(user_indices)}
    book_map = {book: i for i, book in enumerate(book_indices)}
    
    mapped_users = ratings["User-ID"].map(user_map).to_numpy()
    mapped_books = ratings["ISBN"].map(book_map).to_numpy()
    
    user_ratings_number = ratings.groupby("User-ID")["Book-Rating-Normalized"].transform("count")
    book_ratings_number = ratings.groupby("ISBN")["Book-Rating-Normalized"].transform("count")

    ratings["user_rating_count"] = user_ratings_number
    ratings["book_rating_count"] = book_ratings_number

    users_threshold = np.quantile(user_ratings_number, USER_PERCENTILE)
    books_threshold = np.quantile(book_ratings_number, BOOK_PERCENTILE)

    ratings["user_weight"] = np.minimum(1, users_threshold / user_ratings_number)
    ratings["book_weight"] = np.minimum(1, books_threshold / book_ratings_number)
    
    data = ratings["Book-Rating-Normalized"].to_numpy()

    return  mapped_users, mapped_books, data, book_indices, user_indices

def create_user_item_matrix_sparse(ratings):

    mapped_users, mapped_books,data, book_indices, user_indices = prepare_data_for_matrix_sparse(ratings)
    
    user_item_matrix = csr_matrix((data, (mapped_users, mapped_books)), shape=(len(user_indices), len(book_indices)))
    
    return user_item_matrix, user_indices, book_indices

def create_item_user_matrix_sparse(ratings):

    mapped_users,mapped_books,data,book_indices, user_indices = prepare_data_for_matrix_sparse(ratings)

    item_user_matrix = csr_matrix((data, (mapped_books, mapped_users)), shape=(len(book_indices), len(user_indices)))
    return item_user_matrix, book_indices, user_indices
