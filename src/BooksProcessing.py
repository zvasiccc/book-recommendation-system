import re

def preprocess_books(books,ratings):
    books = books.dropna(subset=["ISBN"]) #ISBN is mandatory
    books = books[books["ISBN"].apply(lambda x: isinstance(x,str))] #ISBN has to be string

    books["ISBN"] = books["ISBN"].apply(normalize_isbn) #normalize every cell from ISBN column

    #aggregation of duplicates
    books = books.groupby("ISBN").agg({
        "Book-Title": "first",
        "Book-Author": "first",
        "Year-Of-Publication": "first",
        "Publisher": "first"
    }).reset_index()

    books_with_ratings = ratings["ISBN"].unique()
    books = books[books["ISBN"].isin(books_with_ratings)].reset_index(drop=True)

    return books

def normalize_isbn(isbn):
    if isinstance(isbn,str):
        isbn = re.sub(r'[^A-Za-z0-9]', '', isbn)
        return isbn.upper()
    return isbn