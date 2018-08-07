"""
    Natural LAnguage Processeing
"""

import pandas as pd
import re   #regular expression
import nltk
from nltk.stem.porter import PorterStemmer#bu lib (stem) higgly -< high, recomended -> recomend
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix#to evaluate clasiffication algorihm



nltk.download("stopwords")


ps = PorterStemmer()

dataset = pd.read_csv("Restaurant_Reviews.csv")

english_words = set(stopwords.words("english"))

#preprocessing

#lets remove the stop words (it' s like  word that: 'that', 'this', 'the' which has no includes emotions)
comments = []
for index, row in dataset.iterrows():
    comment = re.sub("[^a-zA-Z]"," ", row[0]).lower()#çıkar. Lower olması duygu anlamında bir değişikliğe neden olmaz. 
    words = comment.split()#it that gibi kelimeler
    words = [ps.stem(word) for word in words if not word in english_words]
    comment = " ".join(words)
    comments.append(comment)

#preprocessing

    
    

    
#Feauture Extraction (buradaki Bag of Words diye geçiyor.)
#machine learning is starting nowv b()
cv = CountVectorizer(max_features=100000)
x = cv.fit_transform(comments).toarray()#☺it' s a sparse matrix which contains commonly zero values(by the way x is the independent values)
# bu yorumlarların hepsi kelimeler olarak 0 ve 1  lere dönüştü
# tek kolon 1536 kolom oldu.

#Feauture Extraction




#Buradan sonrası Makine öğrenmesi, diğer algoritmaları  deneyebilirsin.

y = dataset.iloc[:,1].values#y is the dependent values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)

gnb = GaussianNB()#it' s a classification algorithm
gnb.fit(x_train, y_train)
prediction = gnb.predict(x_test)

cm = confusion_matrix(y_test, prediction)#% 70,5 accuracy

