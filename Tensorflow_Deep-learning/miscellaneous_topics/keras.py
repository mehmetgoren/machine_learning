"""
    an example of keras
"""

from sklearn.datasets import load_wine

wine_data = load_wine()

columns = wine_data["feature_names"]
X = wine_data["data"]
y = wine_data["target"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2, random_state=101)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


from tensorflow.contrib.keras import models, layers, losses,optimizers,metrics,activations
model = models.Sequential()
#input layer
model.add(layers.Dense(units=13, input_dim=13, activation="relu"))

#hidden layer
model.add(layers.Dense(units=39, activation="relu"))
model.add(layers.Dense(units=26, activation="relu"))
model.add(layers.Dense(units=13, activation="relu"))
model.add(layers.Dense(units=7, activation="relu"))

#output layer
model.add(layers.Dense(units=3, activation="softmax"))#3 class olduğu için units=3

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=50)

y_pred = model.predict_classes(X_test)

from sklearn.metrics import confusion_matrix, classification_report
cr = classification_report(y_test, y_pred)