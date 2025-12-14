from src.LoadingData import load_data 
from src.BookRatingsProcessing import preprocess_ratings
from src.UsersProcessing import preprocess_users, ubcf_recommended_books_knn
from src.BooksProcessing import preprocess_books
from src.GlobalVariables import TOP_N_RECOMMENDATIONS


users, books, ratings = load_data()

ratings = preprocess_ratings(ratings)
users = preprocess_users(users, ratings)
books = preprocess_books(books, ratings)


# Brzi lookup po ISBN-u
books_lookup = books.set_index("ISBN")[["Book-Title", "Book-Author"]]


recommendations = ubcf_recommended_books_knn(ratings,TOP_N_RECOMMENDATIONS)

for user_id, books_list in recommendations.items():
    print(f"\nKorisnik {user_id} – preporučene knjige:")

    if not books_list:
        print("  (nema preporuka)")
        continue

    for isbn in books_list:
        if isbn in books_lookup.index:
            title = books_lookup.loc[isbn, "Book-Title"]
            author = books_lookup.loc[isbn, "Book-Author"]
            print(f"  • {title} — {author}")
        else:
            print(f"  • Nepoznata knjiga (ISBN: {isbn})")


print("Users:", len(users))
print("Books:", len(books))
print("Ratings:", len(ratings))
