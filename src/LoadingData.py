import pandas as pd 

def load_data():
    users = pd.read_csv("data/Users.csv",sep=";",  encoding="latin-1")
    books = pd.read_csv("data/Books.csv",sep=";",  encoding="latin-1")
    ratings = pd.read_csv("data/Book-Ratings.csv",sep=";",  encoding="latin-1")
    return users, books, ratings
