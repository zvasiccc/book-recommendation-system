from src.LoadingData import load_data 
from src.BookRatingsPreprocessing import preprocess_ratings
from src.UsersPreprocessing import preprocess_users
from src.BooksPreprocessing import preprocess_books


users, books, ratings = load_data()

ratings = preprocess_ratings(ratings)
users = preprocess_users(users, ratings)
books = preprocess_books(books, ratings)

print("Users:", len(users))
print("Books:", len(books))
print("Ratings:", len(ratings))
