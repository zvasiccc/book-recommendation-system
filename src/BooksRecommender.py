from surprise import Dataset, Reader, SVD, accuracy,SVDpp, NMF
from surprise.model_selection import train_test_split


def svd_recommended_books(user_id, ratings, top_n=10):

    # priprema podataka za Surprise
    reader = Reader(rating_scale=(ratings['Book-Rating'].min(), ratings['Book-Rating'].max()))
    data = Dataset.load_from_df(ratings[['User-ID', 'ISBN', 'Book-Rating']], reader)

    trainset = data.build_full_trainset()  # koristi sve podatke za treniranje

    # inicijalizacija i treniranje SVD modela
    algo = SVD()
    algo.fit(trainset)

    # skupljanje svih ISBN-ova koje korisnik nije ocenio
    all_books = set(ratings['ISBN'].unique())
    user_rated_books = set(ratings[ratings['User-ID'] == user_id]['ISBN'])
    books_to_predict = all_books - user_rated_books

    # predikcija ocena
    predictions = []
    for book in books_to_predict:
        pred = algo.predict(user_id, book)
        predictions.append((book, pred.est))

    # sortiranje i vraćanje top-N knjiga
    top_books = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]
    return top_books


def evaluate_svd_rmse(ratings, test_size=0.2, random_state=42):

    reader = Reader(rating_scale=(ratings['Book-Rating'].min(), ratings['Book-Rating'].max()))
    data = Dataset.load_from_df(ratings[['User-ID', 'ISBN', 'Book-Rating']], reader)

    #podela na train i test skup
    trainset, testset = train_test_split(data, test_size=test_size, random_state=random_state)

    #treniranje SVD modela
    algo = SVD()
    algo.fit(trainset)

    #predikcija na test skupu
    predictions = algo.test(testset)

    rmse = accuracy.rmse(predictions)
    return rmse

def svdpp_recommended_books(user_id, ratings, top_n=10):
    reader = Reader(rating_scale=(ratings['Book-Rating'].min(), ratings['Book-Rating'].max()))
    data = Dataset.load_from_df(ratings[['User-ID', 'ISBN', 'Book-Rating']], reader)
    trainset = data.build_full_trainset()

    algo = SVDpp() # SVD++
    algo.fit(trainset)

    all_books = set(ratings['ISBN'].unique())
    user_rated_books = set(ratings[ratings['User-ID'] == user_id]['ISBN'])
    books_to_predict = all_books - user_rated_books

    predictions = [(book, algo.predict(user_id, book).est) for book in books_to_predict]
    return sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]

def evaluate_svdpp_rmse(ratings, test_size=0.2):
    reader = Reader(rating_scale=(ratings['Book-Rating'].min(), ratings['Book-Rating'].max()))
    data = Dataset.load_from_df(ratings[['User-ID', 'ISBN', 'Book-Rating']], reader)
    trainset, testset = train_test_split(data, test_size=test_size)
    
    algo = SVDpp()
    algo.fit(trainset)
    predictions = algo.test(testset)
    return accuracy.rmse(predictions)

def nmf_recommended_books(user_id, ratings, top_n=10):
    reader = Reader(rating_scale=(ratings['Book-Rating'].min(), ratings['Book-Rating'].max()))
    data = Dataset.load_from_df(ratings[['User-ID', 'ISBN', 'Book-Rating']], reader)
    trainset = data.build_full_trainset()

    algo = NMF()
    algo.fit(trainset)

    all_books = set(ratings['ISBN'].unique())
    user_rated_books = set(ratings[ratings['User-ID'] == user_id]['ISBN'])
    books_to_predict = all_books - user_rated_books

    predictions = [(book, algo.predict(user_id, book).est) for book in books_to_predict]
    return sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]

def evaluate_nmf_rmse(ratings, test_size=0.2):
    reader = Reader(rating_scale=(ratings['Book-Rating'].min(), ratings['Book-Rating'].max()))
    data = Dataset.load_from_df(ratings[['User-ID', 'ISBN', 'Book-Rating']], reader)
    trainset, testset = train_test_split(data, test_size=test_size)

    algo = NMF()
    algo.fit(trainset)
    predictions = algo.test(testset)
    return accuracy.rmse(predictions)

def uporedna_evaluacija(ratings):
    rezultati = {}
    
    # Lista modela koje želimo da testiramo
    modeli = {
        "SVD": SVD(),
        # "SVD++": SVDpp(),
        "NMF": NMF()
    }
    
    reader = Reader(rating_scale=(ratings['Book-Rating'].min(), ratings['Book-Rating'].max()))
    data = Dataset.load_from_df(ratings[['User-ID', 'ISBN', 'Book-Rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2)
    
    for ime, algo in modeli.items():
        print(f"Treniram model: {ime}...")
        algo.fit(trainset)
        predikcije = algo.test(testset)
        rezultati[ime] = accuracy.rmse(predikcije)
        
    return rezultati