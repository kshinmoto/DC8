import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Plot / Graph stuffs
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing


# read in dataset
init_data = pd.read_csv('/Users/kaishinmoto/Downloads/winemag-data-130k-v2.csv')
print("Length of dataframe before duplicates are removed:", len(init_data))

# remove duplicates
parsed_data = init_data[init_data.duplicated('description', keep=False)]
print("Length of dataframe after duplicates are removed:", len(parsed_data))

#remove null values
parsed_data.dropna(subset=['description','points','price'])
print("Length of dataframe after NaNs are removed:", len(parsed_data))


dp = parsed_data[['description','points','price']]
dp = dp.assign(description_length = dp['description'].apply(len))


def transform_points_simplified(points):
    if points < 84:
        return 1
    elif points >= 84 and points < 88:
        return 2
    elif points >= 88 and points < 92:
        return 3
    elif points >= 92 and points < 96:
        return 4
    else:
        return 5

#Applying transform method and assigning result to new column "points_simplified"
dp = dp.assign(points_simplified = dp['points'].apply(transform_points_simplified))
dp = dp.dropna()
print(dp)


X = dp['description']
y = dp['points_simplified']

vectorizer = CountVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)
print('Shape of Sparse Matrix: ', X.shape)
print('Amount of Non-Zero occurrences: ', X.nnz)
# Percentage of non-zero values
density = (100.0 * X.nnz / (X.shape[0] * X.shape[1]))
print('Density: {}'.format((density)))



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state=42)


from sklearn.naive_bayes import MultinomialNB
#Create a Gaussian Classifier
mnb = MultinomialNB()

#Train the model using the training sets
mnb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = mnb.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



