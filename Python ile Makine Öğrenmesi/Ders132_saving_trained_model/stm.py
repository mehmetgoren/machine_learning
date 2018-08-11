"""
    Saving Trained Model
"""

import pandas as pd
from sklearn import model_selection

url = "http://www.bilkav.com/wp-content/uploads/2018/03/satislar.csv"

dataset = pd.read_csv(url)

x=dataset.iloc[:,0:1]
y = dataset.iloc[:,1]

split = .2
x_train, x_test, y_train,y_test = model_selection.train_test_split(x,y,test_size=split)


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)


#lets save the model!.
import pickle

file ="trained_model"
pickle.dump(lr, open(file, "wb"))#write binary
#now out model has been saved.


trained_model = pickle.load(open(file,"rb"))

saved_y_predict = trained_model.predict(x_test)# saved_y_predict === y_predict.
