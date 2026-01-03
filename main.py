from src.LoadingData import load_data 
from src.BookRatingsProcessing import preprocess_ratings
from src.UsersProcessing import preprocess_users
from src.BooksRecommender import svd_recommended_books, evaluate_svd_rmse, uporedna_evaluacija
from src.UBCF import ubcf_recommended_books_knn,  ubcf_evaluation_rmse
from src.IBCF import  ibcf_evaluation_rmse, ibcf_recommended_books_knn
from src.BooksProcessing import preprocess_books
from src.GlobalVariables import TOP_N_RECOMMENDATIONS


users, books, ratings = load_data()

ratings = preprocess_ratings(ratings)
users = preprocess_users(users, ratings)
books = preprocess_books(books, ratings)

ratings = ratings[ratings["ISBN"].isin(books["ISBN"])].reset_index(drop=True)

# ratings_per_user = ratings.groupby('User-ID').size().sort_values(ascending=False)
# print(ratings_per_user.head(10))

#quick lookup 
books_lookup = books.set_index("ISBN")[["Book-Title", "Book-Author"]]

#high number of grades
user_id= 198711

#low number of grades
#user_id = 276828

recommendations_ubcf = ubcf_recommended_books_knn(user_id, ratings,TOP_N_RECOMMENDATIONS)
recommendations_ibcf = ibcf_recommended_books_knn(user_id, ratings,TOP_N_RECOMMENDATIONS)
recommendations_svd =  svd_recommended_books(user_id, ratings,TOP_N_RECOMMENDATIONS)

true_ratings = ratings[ratings['User-ID']==user_id].set_index('ISBN')['Book-Rating'].to_dict()




ubcf_rmse = ubcf_evaluation_rmse(user_id, ratings)
print(f"UBCF RMSE za korisnika {user_id}: {ubcf_rmse:.4f}")

ibcf_rmse = ibcf_evaluation_rmse(user_id, ratings)

print(f"IBCF RMSE za korisnika {user_id}: {ibcf_rmse:.4f}")


rmse_svd = evaluate_svd_rmse(ratings)

uporedno = uporedna_evaluacija(ratings)


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


