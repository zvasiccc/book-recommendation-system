from src.GlobalVariables import USER_PERCENTILE, BOOK_PERCENTILE, MIN_RATING, MAX_RATING, USER_BASED_THRESHOLD_PERCENTILE,  MIN_NUBMER_OF_RATINGS, MIN_NUBMER_OF_BOOK_RATINGS

def preprocess_ratings(ratings):
    ratings = ratings.dropna(subset=["User-ID", "ISBN"]) #User-ID and ISBN are mandatory
    ratings = ratings[ratings["User-ID"].apply(lambda x: str(x).isdigit())]
    ratings = ratings[ratings["ISBN"].apply(lambda x: isinstance(x, str))] 

    ratings = ratings.groupby(["User-ID", "ISBN"], as_index=False)["Book-Rating"].mean() #aggregating duplicated in a single row with average rating

    ratings = ratings[(ratings["Book-Rating"] >= MIN_RATING) & (ratings["Book-Rating"] <= MAX_RATING)]
    
    user_ratings_number = ratings.groupby("User-ID").size() #Pandas series: key is User-ID, value is count of ratings for that user
    book_ratings_number = ratings.groupby("ISBN").size()

    users_threshold = user_ratings_number.quantile(USER_PERCENTILE)
    books_threshold = book_ratings_number.quantile(BOOK_PERCENTILE)

    extreme_users = user_ratings_number[user_ratings_number > users_threshold].index #is Key in user_ratings_number Pandas Series bigger then threshold
    extreme_books = book_ratings_number[book_ratings_number > books_threshold].index

    #new columns with number of user ratings and book ratings
    ratings["user_rating_count"] = ratings["User-ID"].map(user_ratings_number)
    ratings["book_rating_count"] = ratings["ISBN"].map(book_ratings_number)

    #users scalling
    #ratings.loc[ratings["User-ID"].isin(extreme_users), "Book-Rating"] *= users_threshold / ratings.loc[ratings["User-ID"].isin(extreme_users), "user_rating_count"]

    #books scalling
    #ratings.loc[ratings["ISBN"].isin(extreme_books), "Book-Rating"] *= books_threshold / ratings.loc[ratings["ISBN"].isin(extreme_books), "book_rating_count"]

    return ratings



def filter_ratings(ratings):
    book_counts = ratings.groupby("ISBN").size()
    popular_books = book_counts[book_counts >= MIN_NUBMER_OF_BOOK_RATINGS].index
    ratings = ratings[ratings["ISBN"].isin(popular_books)]
    
    user_counts = ratings.groupby("User-ID").size()
    threshold = max(user_counts.quantile(USER_BASED_THRESHOLD_PERCENTILE), MIN_NUBMER_OF_RATINGS)
    active_users = user_counts[user_counts >= threshold].index
    
    return ratings[ratings["User-ID"].isin(active_users)].reset_index(drop=True)

# def ratings_normalization(ratings):

#     user_mean = ratings.groupby("User-ID")["Book-Rating"].mean()
#     ratings["Book-Rating-Normalized"] = ratings.apply(
#         lambda row: row["Book-Rating"] - user_mean[row["User-ID"]],
#         axis=1
#     )
#     return ratings

def ratings_normalization(ratings):
    # 1. Izračunaj prosek jednom za sve korisnike
    user_means = ratings.groupby("User-ID")["Book-Rating"].mean()
    
    # 2. Dodaj kolonu user_mean (biće ti korisna za denormalizaciju kasnije)
    # map() će svakom User-ID-u dodeliti njegov prosek munjevitom brzinom
    ratings["user_mean"] = ratings["User-ID"].map(user_means)
    
    print(f"user means",user_means)
    
    # 3. Izračunaj normalizovanu ocenu bez apply() petlje
    ratings["Book-Rating-Normalized"] = ratings["Book-Rating"] - ratings["user_mean"]
    
    return ratings
