import pandas as pd
# coding=gbk
movies = pd.read_csv("ml-1m/movies.dat", sep="::", encoding='gbk', names=["movie_id", "title", "genres"])
print('get')