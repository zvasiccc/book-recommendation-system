from src.LoadingData import load_data 
from src.BookRatingsProcessing import preprocess_ratings
from src.UsersProcessing import preprocess_users
from src.BooksRecommender import ubcf_recommended_books_knn, ibcf_recommended_books_knn, svd_recommended_books
from src.BooksProcessing import preprocess_books
from src.GlobalVariables import TOP_N_RECOMMENDATIONS


users, books, ratings = load_data()

ratings = preprocess_ratings(ratings)
users = preprocess_users(users, ratings)
books = preprocess_books(books, ratings)

ratings = ratings[ratings["ISBN"].isin(books["ISBN"])].reset_index(drop=True)


#quick lookup 
books_lookup = books.set_index("ISBN")[["Book-Title", "Book-Author"]]

#user_id = 278851
#user_id = 278854
user_id= 276828

recommendations_ubcf = ubcf_recommended_books_knn(user_id, ratings,TOP_N_RECOMMENDATIONS)
recommendations_ibcf = ibcf_recommended_books_knn(user_id, ratings,TOP_N_RECOMMENDATIONS)
recommendations_svd =  svd_recommended_books(user_id, ratings,TOP_N_RECOMMENDATIONS)

print(f"UBCF:")
for isbn, score in recommendations_ubcf:
    book = books[books["ISBN"] == isbn].iloc[0]
    print(f"{book['Book-Title']} — {book['Book-Author']}  (score={score:.3f})")
    
print(f"IBCF:")   
for isbn, score in recommendations_ibcf:
    book = books[books["ISBN"] == isbn].iloc[0]
    print(f"{book['Book-Title']} — {book['Book-Author']}  (score={score:.3f})")
    
print(f"SVD:")   
for isbn, score in recommendations_svd:
    book = books[books["ISBN"] == isbn].iloc[0]
    print(f"{book['Book-Title']} — {book['Book-Author']}  (score={score:.3f})")
