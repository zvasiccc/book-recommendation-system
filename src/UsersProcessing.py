def preprocess_users(users, ratings):
    users = users[["User-ID"]] #only User-ID column is used in this project
    users = users.dropna(subset=["User-ID"]) #ID is mandatory
    users = users[users["User-ID"].apply(lambda x: str(x).isdigit())] #ID has to be number

    users = users.drop_duplicates(subset=["User-ID"], keep="first") 

    users_with_ratings = ratings["User-ID"].unique()
    users = users[users["User-ID"].isin(users_with_ratings)].reset_index(drop=True)

    return users


