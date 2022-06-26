import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('heart.csv')
X = data.drop(columns='target', axis=1)
Y = data["target"]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
classifier = RandomForestClassifier(n_estimators=10, criterion="entropy")
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
A = accuracy_score(y_test, y_pred)
#input = (37, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1)  # person with heart disease
input = (67, 1, 0, 160, 286, 0, 0, 108, 1, 1.5, 1, 3, 2)  # person does not have heart disease
input_data_array = np.asarray(input)
input_data_reshape = input_data_array.reshape(1, -1)
prediction = classifier.predict(input_data_reshape)
A = accuracy_score(y_test, y_pred)
print(A)
print(prediction)
if prediction == 0:
    print("The person does not have heart disease")
else:
    print("The person have heart disease")