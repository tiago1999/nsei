## machine learning libraries and functions
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# pandas
import pandas as pd

# saving model
import pickle

# load the dataset 
iris_bunch = load_iris() # a bunch object, works like a dict

iris_df = pd.DataFrame(iris_bunch['data'], columns = iris_bunch['feature_names'])

target = iris_bunch['target']

# splitting dataset into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(iris_df, target, random_state=1, test_size = 0.2)

# creating logistic regression model 
logistic = LogisticRegression(max_iter=100)

# training model
logistic.fit(X_train, y_train)

# prediction
print(logistic.predict(X_test))

# model evaluation 
print(logistic.score(X_test, y_test))

# saving the model 
pkl_file = 'logistic_model.p'

with open(pkl_file, 'wb') as file:
    pickle.dump(logistic, file)
