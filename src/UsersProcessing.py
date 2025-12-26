from src.GlobalVariables import USER_BASED_THRESHOLD_PERCENTILE, TOP_N_RECOMMENDATIONS, MIN_NUBMER_OF_RATINGS, MIN_NUBMER_OF_BOOK_RATINGS
import pandas as pd


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