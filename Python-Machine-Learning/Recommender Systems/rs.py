"""
    Recomender System
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb



columns_names = ["user_id","item_id","rating","timestamp"]
df = pd.read_csv("u.data",sep="\t",names=columns_names)

movie_titles = pd.read_csv("Movie_Id_Titles")


#join yapıyoruz
df = pd.merge(df, movie_titles, on="item_id")

sb.set_style("white")

group_rating = df.groupby("title")["rating"].mean().sort_values(ascending=False)#verilen yıldıza göre sırala

group_count = df.groupby("title")["rating"].count().sort_values(ascending=False)


ratings = pd.DataFrame(group_rating)
ratings["number of ratings"] = pd.DataFrame(group_count)

#ratings["number of ratings"].hist(bins=70)
#plt.show()
#
#ratings["rating"].hist(bins=70)
#plt.show()
#
#
#
#sb.jointplot(x="rating", y= "number of ratings", data=ratings, alpha=.5)#çok izlenenlker genelde yüksek puanlı oluyor.
#plt.show()


#matrix haline getirelim
moviemat = df.pivot_table(index="user_id", columns="title",values="rating")

#en çok rating alan filimlerden seçelim
starwars_user_ratings = moviemat["Star Wars (1977)"]  




similar_to_starwars = moviemat.corrwith(starwars_user_ratings)


corr_starwars = pd.DataFrame(similar_to_starwars, columns=["Correlation"])
corr_starwars.dropna(inplace=True)#null olanlar silinsin.

corr_starwars = corr_starwars.sort_values("Correlation", ascending=False)



corr_starwars = corr_starwars.join(ratings["number of ratings"])
corr_starwars = corr_starwars[corr_starwars["number of ratings"]>100].sort_values("Correlation", ascending=False)


recomended_movies_for_starwars_fan = corr_starwars.head(5) 