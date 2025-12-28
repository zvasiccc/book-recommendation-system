from scipy.sparse import csr_matrix

def prepare_data_for_matrix_sparse(ratings):
    
    user_index = ratings["User-ID"].unique()
    book_index = ratings["ISBN"].unique()

    #create dictionaries for users and books
    user_map = {user: i for i, user in enumerate(user_index)}
    book_map = {book: i for i, book in enumerate(book_index)}
    
    mapped_users = ratings["User-ID"].map(user_map).to_numpy()
    mapped_books = ratings["ISBN"].map(book_map).to_numpy()
    data = ratings["Book-Rating-Normalized"].to_numpy()
    return  mapped_users, mapped_books, data, book_index, user_index

def create_user_item_matrix_sparse(ratings):

    mapped_users,mapped_books,data, book_index, user_index = prepare_data_for_matrix_sparse(ratings)
    
    user_item_matrix = csr_matrix((data, (mapped_users, mapped_books)), shape=(len(user_index), len(book_index)))
    
    return user_item_matrix, user_index, book_index

def create_item_user_matrix_sparse(ratings):

    mapped_users,mapped_books,data,book_index, user_index = prepare_data_for_matrix_sparse(ratings)

    item_user_matrix = csr_matrix((data, (mapped_books, mapped_users)), shape=(len(book_index), len(user_index)))
    return item_user_matrix, book_index, user_index
