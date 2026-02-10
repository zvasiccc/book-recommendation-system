from src.LoadingData import load_data 
from src.BookRatingsProcessing import preprocess_ratings, filter_ratings, ratings_normalization
from src.UsersProcessing import preprocess_users
from src.ModelBasedCF import svd_recommended_books,nmf_recommended_books, model_based_evaluation
from src.UBCF import ubcf_recommended_books_knn,  ubcf_evaluation_rmse
from src.IBCF import  ibcf_evaluation_rmse, ibcf_recommended_books_knn
from src.BooksProcessing import preprocess_books
from src.GlobalVariables import TOP_N_RECOMMENDATIONS

users, books, ratings = load_data()

ratings = preprocess_ratings(ratings)

users = preprocess_users(users, ratings)
books = preprocess_books(books, ratings)

ratings = ratings[ratings["ISBN"].isin(books["ISBN"])].reset_index(drop=True)
ratings = filter_ratings(ratings)
ratings = ratings_normalization(ratings)

books = books[books["ISBN"].isin(ratings["ISBN"].unique())].reset_index(drop=True)
users = users[users["User-ID"].isin(ratings["User-ID"].unique())].reset_index(drop=True)
    
# ratings_per_user = ratings.groupby('User-ID').size().sort_values(ascending=True)
# print(ratings_per_user.head(6000))


user_id=200245

recommendations_ubcf = ubcf_recommended_books_knn(user_id, ratings,TOP_N_RECOMMENDATIONS)
recommendations_ibcf = ibcf_recommended_books_knn(user_id, ratings,TOP_N_RECOMMENDATIONS)
recommendations_svd =  svd_recommended_books(user_id, ratings,TOP_N_RECOMMENDATIONS)
recommendations_nmf =  nmf_recommended_books(user_id, ratings,TOP_N_RECOMMENDATIONS)

true_ratings = ratings[ratings['User-ID']==user_id].set_index('ISBN')['Book-Rating'].to_dict()

ubcf_rmse = ubcf_evaluation_rmse(user_id, ratings)
print(f"UBCF RMSE: {ubcf_rmse:.4f}")

ibcf_rmse = ibcf_evaluation_rmse(user_id, ratings)

if ibcf_rmse is not None:
    print(f"IBCF RMSE: {ibcf_rmse:.4f}")
else:
    print(f"IBCF RMSE: Insufficient data")


model_based_evaluation(ratings)

#quick lookup 
books_lookup = books.set_index("ISBN")[["Book-Title", "Book-Author"]]

print(f"-------------------------------------------------------------------------------")
print(f"UBCF:")
for isbn, score in recommendations_ubcf:
    book = books_lookup.loc[isbn]
    print(f"{book['Book-Title']} — {book['Book-Author']}  (score={score:.3f})")
print(f"-------------------------------------------------------------------------------")

print(f"IBCF:")   
for isbn, score in recommendations_ibcf:
    book = books_lookup.loc[isbn]
    print(f"{book['Book-Title']} — {book['Book-Author']}  (score={score:.3f})")
print(f"-------------------------------------------------------------------------------")

print(f"SVD:")   
for isbn, score in recommendations_svd:
    book = books_lookup.loc[isbn]
    print(f"{book['Book-Title']} — {book['Book-Author']}  (score={score:.3f})")
print(f"-------------------------------------------------------------------------------")

print(f"NMF:")   
for isbn, score in recommendations_nmf:
    book = books_lookup.loc[isbn]
    print(f"{book['Book-Title']} — {book['Book-Author']}  (score={score:.3f})")

