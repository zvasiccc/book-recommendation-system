import re
def normalize_isbn(isbn):
    if isinstance(isbn,str):
        isbn = re.sub(r'[^A-Za-z0-9]', '', isbn)
        return isbn.upper()
    return isbn

def preprocess_books(books,ratings):
    books = books.dropna(subset=["ISBN"])
    books = books[books["ISBN"].apply(lambda x: isinstance(x,str))]

    books["ISBN"] = books["ISBN"].apply(normalize_isbn)

     # Spajanje duplikata
    books = books.groupby("ISBN").agg({
        "Book-Title": "first",
        "Book-Author": "first",
        "Year-Of-Publication": "first",
        "Publisher": "first"
    }).reset_index()

      # Zadrzavaju se sa samo ocenjene knjige
    books_with_ratings = ratings["ISBN"].unique()
    books = books[books["ISBN"].isin(books_with_ratings)].reset_index(drop=True)

    return books
