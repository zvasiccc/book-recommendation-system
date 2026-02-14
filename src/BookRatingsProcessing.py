from src.BooksProcessing import normalize_isbn
from src.GlobalVariables import  MIN_RATING, MAX_RATING, USER_BASED_THRESHOLD_PERCENTILE,  MIN_NUBMER_OF_RATINGS, MIN_NUBMER_OF_BOOK_RATINGS

def preprocess_ratings(ratings):
    ratings = ratings.dropna(subset=["User-ID", "ISBN"]) #User-ID and ISBN are mandatory
    ratings = ratings[ratings["User-ID"].apply(lambda x: str(x).isdigit())]
    ratings = ratings[ratings["ISBN"].apply(lambda x: isinstance(x, str))] 
    ratings["ISBN"] = ratings["ISBN"].apply(normalize_isbn) 

    ratings = ratings.groupby(["User-ID", "ISBN"], as_index=False)["Book-Rating"].mean() #aggregating duplicated in a single row with average rating

    ratings = ratings[(ratings["Book-Rating"] >= MIN_RATING) & (ratings["Book-Rating"] <= MAX_RATING)]
    
    return ratings


def filter_ratings(ratings):
    book_counts = ratings.groupby("ISBN").size()
    popular_books = book_counts[book_counts >= MIN_NUBMER_OF_BOOK_RATINGS].index
    ratings = ratings[ratings["ISBN"].isin(popular_books)]
    
    user_counts = ratings.groupby("User-ID").size()
    threshold = max(user_counts.quantile(USER_BASED_THRESHOLD_PERCENTILE), MIN_NUBMER_OF_RATINGS)
    active_users = user_counts[user_counts >= threshold].index
    
    ratings = ratings[ratings["User-ID"].isin(active_users)].reset_index(drop=True)
    
    return ratings


def ratings_normalization(ratings):
    user_means = ratings.groupby("User-ID")["Book-Rating"].mean()
    
    ratings["User-Mean"] = ratings["User-ID"].map(user_means)
    
    ratings["Book-Rating-Normalized"] = ratings["Book-Rating"] - ratings["User-Mean"]
    
    return ratings
