from src.LoadingData import load_data 
from src.BookRatingsProcessing import preprocess_ratings
from src.UsersProcessing import preprocess_users, ubcf_recommended_books_knn
from src.BooksProcessing import preprocess_books
from src.GlobalVariables import TOP_N_RECOMMENDATIONS


users, books, ratings = load_data()

ratings = preprocess_ratings(ratings)
users = preprocess_users(users, ratings)
books = preprocess_books(books, ratings)

recommendations = ubcf_recommended_books_knn(ratings,TOP_N_RECOMMENDATIONS)

# Prikaz preporuka za prvih 5 korisnika
print("Recomends",recommendations)
# for user_id, books_list in list(recommendations.items())[:5]:
#     print(f"Korisnik {user_id} preporučene knjige: {books_list}")

print("Users:", len(users))
print("Books:", len(books))
print("Ratings:", len(ratings))
