import unittest
import pandas as pd


# Starting code
import pandas as pd


def check_similar_users(user_id_x, user_id_y, ratings):
    if user_id_x == user_id_y:
        return False

    joint_ratings = pd.merge(
        ratings[ratings["user id"] == user_id_x],
        ratings[ratings["user id"] == user_id_y],
        on="item id",
    )
    if joint_ratings.empty:
        return False

    joint_ratings["rating_diff"] = abs(
        joint_ratings["rating_x"] - joint_ratings["rating_y"]
    )
    if max(joint_ratings["rating_diff"]) <= 1:
        return True
    return False


def get_recommendations(users, movies, ratings, full_name, method, year):

    if method == 'by_popularity':
        merged = movies.merge(ratings, how='inner', left_on='movie id', right_on='item id')
        breakpoint()
        loc = merged.iloc[merged.iloc[merged['rating'].idxmax()]]

        return loc['full name']

    return ''


def test_first():
    users = pd.read_csv("data/users.csv")
    ratings = pd.read_csv("data/ratings.csv",on_bad_lines='skip')
    movies = pd.read_csv("data/movies.csv", on_bad_lines='skip')

    result = get_recommendations(users, movies, ratings, 'Ryan James', 'by_popularity', 1995)
    assert result == 'Toy Story (1995)'




