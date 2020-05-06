# importing the libraries
import numpy as np
#import matplotlib.pylot as plt
import pandas as pd
# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Importing the dataset
startcolumn = 3
endcolumn=13
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, startcolumn:endcolumn].values
y = dataset.iloc[:, endcolumn].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X1 = LabelEncoder()
X[:, 0] = labelencoder_X1.fit_transform(X[:, 0])
labelencoder_X2 = LabelEncoder()
X[:, 1] = labelencoder_X2.fit_transform(X[:, 1])
ct = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Part 2- Now let's make a ANN !
# importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
# using dropout technique to improve the independent correllation of Neuron by disabling some random neurons,
# that means each time we re-train
# the Network, the neuron is refreshed.
# That helps to avoid overfitting when the network is trained too much from time to time
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer with dropout,
# the num_Neuron in hidden = (input+ouput)/2 is the experiment
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10))
classifier.add(Dropout(rate = 0.1))
# Adding the second hidden layer
# units = ouput_dim
# init = kernel_initializer _weight

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate = 0.1))
# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = 'sigmoid'))
# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics= ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size= 10 , epochs =100)

# Part 3
# predicting the test set results
y_pred =classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#Predicting a single new observation
""" Predict if the customer with the following informations will leave the bank:
Geography: France

Gender: Female
Age:42
Tenure:2
Number of Products: 1
Is Active Member:Yes
Contract Online: No
Internet Action: 0
Having Vollkasko: No
Kasco Change: 2
"""
new_prediction = classifier.predict(sc.transform(np.array([[0, 0,  42, 2, 1, 1, 0,0,0,2 ]])))
new_prediction = (new_prediction > 0.5)



# making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Part 4 -Evaluating, Improving and tuning the ANN
# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=10))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer="uniform", activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10 , epochs = 100)
# n_jobs is the number of CPU to run all training in parallel, n_jobs = -1 means all cpu
# cv is the number of folds in K-folds cross-validation
accuracies = cross_val_score(estimator= classifier, X = X_train, y = y_train, cv = 10,  n_jobs = -1)

# Overfitting happens when high variance and low accuracies
mean = accuracies.mean()
variance = accuracies.std()

#improving the ANN
#Drop out Regularization to reduce overfitting if needed

#Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=10))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer="uniform", activation='sigmoid'))
    classifier.compile(optimizer= optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


classifier = KerasClassifier(build_fn = build_classifier)
# creating a dictionary for parameters which we want to tune
parameters = {'batch_size':[25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator=classifier, param_grid= parameters, scoring='accuracy', cv = 10)
grid_search = grid_search.fit(X_train,y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
