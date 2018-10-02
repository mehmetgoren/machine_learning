"""
    Natural Language Processing
"""


import nltk
#nltk.download_shell()
nltk.download("stopwords")


messages = [line.rstrip() for line in open("smsspamcollection/SMSSpamCollection")]

import pandas as pd
messages = pd.read_csv("smsspamcollection/SMSSpamCollection", sep="\t", names=["label","message"])
describe=messages.groupby("label").describe()

#for mess_no, messages in enumerate(messages)
messages["length"] = messages["message"].apply(len)


#import matplotlib.pyplot as plt
#import seaborn as sb
#messages["length"].plot.hist(bins=150)
#plt.show()


#spam ve olmayan mesajların uzunlukı dağılımları.
#messages.hist(column="length", by="label",bins=60,figsize=(12,4))



from nltk.corpus import stopwords as sw
stopwords = sw.words("english")

import string
def text_process(str):
    nopunc = [ch for ch in str if ch not in string.punctuation]
    nopunc = "".join(nopunc)
    ret=  [word for word in nopunc.split() if word.lower() not in stopwords]
    #return ret;
    #print(" ".join(ret))
    return " ".join(ret)


messages["message"] = messages["message"].apply(text_process)



#creatinf bag of words model
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
cv = CountVectorizer()# CountVectorizer(max_features=1500)#☼kolon sayısı
X = cv.fit_transform(messages["message"]).toarray()#matrix. sparse matrix yaptık (o' ı çok olan matrix)
y = messages["label"] == "ham"
y = np.array(y, dtype=int)



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 0)


def f1_score(cm):
    tp = cm[0][0]
    tn = cm[1][1]
    fp = cm[1][0]
    fn = cm[0][1]
    
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = tp / (tp+fp)
    recall = tp/(tp+fp)
    score = 2*precision*recall/(precision+recall)
    return (accuracy, score)


from sklearn.metrics import confusion_matrix, classification_report
def test(classifier):
    classifier.fit(X_train, y_train)  
    y_pred = classifier.predict(X_test) 
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    return f1_score(cm), cr


from sklearn.naive_bayes import MultinomialNB
f1, cr = test(MultinomialNB())


#başka bir yöntem
#from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.pipeline import Pipeline
#pipeline = Pipeline([
#        ("bow",CountVectorizer(analyzer=text_process)),
#        ("tfidf", TfidfTransformer()),
#        ("classifier", MultinomialNB())
#        ])
#pipeline.fit(X_train,y_train)
#pred = pipeline.predict(X_test)
#cr_pipeline = classification_report(y_test)
