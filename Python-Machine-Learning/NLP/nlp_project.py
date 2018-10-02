"""
    NLP Project
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb



df = pd.read_csv("yelp.csv")

#sb.countplot(x="stars", data=df,palette="rainbow")
#plt.show()
#
#
#stars = df.groupby("stars").mean()
#stars["text_length"]= df["text"].apply(len)
#sb.heatmap(stars.corr(),cmap="coolwarm", annot=True)

df_classified = df[(df["stars"]==5) | (df["stars"]==1)]

X = df_classified["text"]
y = df_classified["stars"]



from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

X=cv.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 0)


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train,y_train)
y_pred = nb.predict(X_test)


from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)


#from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.pipeline import Pipeline
#pipeline = Pipeline([
#        ("bow",CountVectorizer()),
#        ("tfidf", TfidfTransformer()),
#        ("model", MultinomialNB())
#        ])
#pipeline.fit(X_train,y_train)
#pred = pipeline.predict(X_test)
#cr_pipeline = classification_report(y_test)