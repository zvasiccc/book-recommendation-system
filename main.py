import pandas as pd
import re 

#Ucitavanje inicijalnih fajlova
users = pd.read_csv("data/Users.csv",sep=";",  encoding="latin-1")
books = pd.read_csv("data/Books.csv",sep=";",  encoding="latin-1")
ratings = pd.read_csv("data/Book-Ratings.csv",sep=";",  encoding="latin-1")

#Izbacivanje redova sa praznim User ID ili ISBN
users = users.dropna(subset=["User-ID"])
books = books.dropna(subset=["ISBN"])
ratings = ratings.dropna(subset=["User-ID","ISBN"])

#Izbacivanje redova gde User ID nije broj 
users = users[users["User-ID"].apply(lambda x: str(x).isdigit())]
ratings = ratings[ratings["User-ID"].apply(lambda x: str(x).isdigit())]

#Izbacivanje redova gde ISBN nije string 
books = books[books["ISBN"].apply(lambda x: isinstance(x,str))]
ratings = ratings[ratings["ISBN"].apply(lambda x: isinstance(x,str))]


#Priprema podataka iz fajla Users.csv
users=users.drop_duplicates(subset=["User-ID"],keep="first")

users_with_ratings = ratings["User-ID"].unique()
users = users[users["User-ID"].isin(users_with_ratings)].reset_index(drop=True)

#Priprema podataka iz fajla Books.csv
def normalize_isbn(isbn):
    if isinstance(isbn, str):
        isbn = re.sub(r'[^A-Za-z0-9]', '', isbn)  # uklanja sve sto nije slovo ili broj
        isbn = isbn.upper()
        return isbn
    return isbn

books["ISBN"]=books["ISBN"].apply(normalize_isbn)

books = books.groupby("ISBN").agg({
    "Book-Title": "first",
    "Book-Author": "first",
    "Year-Of-Publication": "first",
    "Publisher": "first"
}).reset_index()

books_with_ratings = ratings["ISBN"].unique()
books = books[books["ISBN"].isin(books_with_ratings)].reset_index(drop=True)

#Priprema podataka iz fajla Book-Ratings.csv

ratings = ratings.groupby(["User-ID","ISBN"],as_index=False)["Book-Rating"].mean()

ratings = ratings[(ratings["Book-Rating"] >= 0) & (ratings["Book-Rating"] <10)]

users_ratings_number = ratings.groupby("User-ID").size()
books_ratings_number = ratings.groupby("ISBN").size()

users_threshold = users_ratings_number.quantile(0.99)
books_threshold = books_ratings_number.quantile(0.99)

print("99. percentil broja ocena po korisniku:", users_threshold)
print("99. percentil broja ocena po knjizi:", books_threshold)

extreme_users = users_ratings_number[users_ratings_number > users_threshold].index

print("Broj ekstremnih korisnika:", len(extreme_users))

extreme_books = books_ratings_number[books_ratings_number > books_threshold].index

print("Broj ekstremno ocenjenih knjiga:", len(extreme_books))

# Smanjenje uticaja ekstremnih korisnika
ratings.loc[ratings["User-ID"].isin(extreme_users), "Book-Rating"] *= (users_threshold / (users_ratings_number[ratings["User-ID"]].reindex(ratings.index, fill_value=0)))

# Smanjenje uticaja ekstremnih knjiga
ratings.loc[ratings["ISBN"].isin(extreme_books), "Book-Rating"] *= (books_threshold / (books_ratings_number[ratings["ISBN"]].reindex(ratings.index, fill_value=0)))


print("Users", len(users))
print("Books", len(books))
print("Book ratings", len(ratings))