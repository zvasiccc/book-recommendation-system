from src.GlobalVariables import USER_PERCENTILE, BOOK_PERCENTILE, MIN_RATING, MAX_RATING

def preprocess_ratings(ratings):
    ratings = ratings.dropna(subset=["User-ID", "ISBN"])
    ratings = ratings[ratings["User-ID"].apply(lambda x: str(x).isdigit())]
    ratings = ratings[ratings["ISBN"].apply(lambda x: isinstance(x, str))]

    # Spajanje duplikata
    ratings = ratings.groupby(["User-ID", "ISBN"], as_index=False)["Book-Rating"].mean()

    # Opseg ocena
    ratings = ratings[(ratings["Book-Rating"] >= MIN_RATING) & (ratings["Book-Rating"] <= MAX_RATING)]

    # Broj ocena
    user_ratings_number = ratings.groupby("User-ID").size()
    book_ratings_number = ratings.groupby("ISBN").size()

    users_threshold = user_ratings_number.quantile(USER_PERCENTILE)
    books_threshold = book_ratings_number.quantile(BOOK_PERCENTILE)

    extreme_users = user_ratings_number[user_ratings_number > users_threshold].index
    extreme_books = book_ratings_number[book_ratings_number > books_threshold].index

     # Dodavanje kolona sa brojem ocena po korisniku i knjizi
    ratings["user_rating_count"] = ratings["User-ID"].map(user_ratings_number)
    ratings["book_rating_count"] = ratings["ISBN"].map(book_ratings_number)

    # Normalizacija ekstremnih korisnika
    ratings.loc[ratings["User-ID"].isin(extreme_users), "Book-Rating"] *= \
        users_threshold / ratings.loc[ratings["User-ID"].isin(extreme_users), "user_rating_count"]

    # Normalizacija ekstremnih knjiga
    ratings.loc[ratings["ISBN"].isin(extreme_books), "Book-Rating"] *= \
        books_threshold / ratings.loc[ratings["ISBN"].isin(extreme_books), "book_rating_count"]



    return ratings
