"""
    Natural Language Processing
"""
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)#tab dilimiter. tsv === tab sv

import re
import nltk
nltk.download("stopwords")#to remove the worlds which is meaningles such as 'wow'
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.metrics import confusion_matrix



 #cleaning text
sw = set(stopwords.words("english"))
ps = PorterStemmer()
def clean_text(text):
    s = re.sub("[^a-zA-Z]", " ", text)
    s = s.lower()
    s = s.split()
    s = [ps.stem(word) for word in s if not word in sw]
    s = " ".join(s)
    return s



corpus = []
for review in dataset["Review"]:
    corpus.append(clean_text(review))



#creatinf bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()# CountVectorizer(max_features=1500)#☼kolon sayısı
X = cv.fit_transform(corpus).toarray()#matrix. sparse matrix yaptık (o' ı çok olan matrix)
y = dataset.iloc[:,-1].values#vector



# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
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


def test(classifier):
    classifier.fit(X_train, y_train)  
    y_pred = classifier.predict(X_test) 
    cm = confusion_matrix(y_test, y_pred)
    return f1_score(cm)


from sklearn.naive_bayes import GaussianNB
naive_bayes = test(GaussianNB())

from sklearn.linear_model import LogisticRegression
logistic_regression = test(LogisticRegression(random_state=0))

from sklearn.neighbors import KNeighborsClassifier
k_nearest_neighbors = test(KNeighborsClassifier(n_neighbors=5))

from sklearn.svm import SVC
svc_linear = test(SVC(kernel="linear", random_state=0))
svc_rbf = test(SVC(kernel="rbf", random_state=0))


from sklearn.tree import DecisionTreeClassifier
decision_tree = test(DecisionTreeClassifier(random_state=0))


from sklearn.ensemble import RandomForestClassifier
random_forest = test(RandomForestClassifier(n_estimators=100, random_state=0))
