import nltk
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB

# read in dataset, make sure to change path for file in your computer
init_data = pd.read_csv('/Users/kaishinmoto/Downloads/winemag-data-130k-v2.csv')
print("Length of dataframe before duplicates are removed:", len(init_data))

# remove duplicates
parsed_data = init_data[init_data.duplicated('description', keep=False)]
print("Length of dataframe after duplicates are removed:", len(parsed_data))

#remove null values
parsed_data.dropna(subset=['description','points','price'])
print("Length of dataframe after NaNs are removed:", len(parsed_data))


dp = parsed_data[['description','points']]
print(dp)

# remove the punctuations from description and create new column called new_description
dp = dp.assign(new_description = dp['description'].str.replace('[^\w\s]',''))

# this section would be used to lemmatize the description, but I found that it lowered the accuracy
    # you can uncomment and test it out for yourself if interested
    # I also tried to take out stopwords but that lowered the accuracy as well
'''w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w,'v') for w in w_tokenizer.tokenize(text)]

dp = dp.assign(lemma_description = dp['new_description'].apply(lemmatize_text))
print(dp['lemma_description'])

# Function to convert
def listToString(s):
    # initialize an empty string
    str1 = " "
    # return string
    return (str1.join(s))
# create final_description column of lemmatized words in a string form
dp = dp.assign(final_description = dp['lemma_description'].apply(listToString))'''


dp = dp.assign(description_length = dp['description'].apply(len))

# same function used in example given
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

# assign x and y, if testing lemmatized version change new_description to final_description
X = dp['new_description']
y = dp['points_simplified']

vectorizer = CountVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)

# print final version of dp being trained and tested, if doing lemmatized version change new_description to final_description
print("Final DP: \n", dp['new_description'])

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.1,random_state=10 )


#Create a Gaussian Classifier
mnb = MultinomialNB()

#Train the model using the training sets
mnb.fit(train_x, train_y)

#Predict the response for test dataset
y_pred = mnb.predict(test_x)


# Model Accuracy, how often is the classifier correct?
print("MN Naive Bayes Accuracy: ",metrics.accuracy_score(test_y, y_pred))

# Train multi-classification model with logistic regression, we raised max_iter because of convergence issues
    # if you have any convergence issues yourself, you may raise the max_iter here
lr = linear_model.LogisticRegression(max_iter= 10000)
lr.fit(train_x, train_y)

# Train multinomial logistic regression model
mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(train_x, train_y)

#print("Logistic regression Train Accuracy: ", metrics.accuracy_score(train_y, lr.predict(train_x)))
print("Logistic regression Test Accuracy: ", metrics.accuracy_score(test_y, lr.predict(test_x)))
#print("Multinomial Logistic regression Train Accuracy: ", metrics.accuracy_score(train_y, mul_lr.predict(train_x)))
print("Multinomial Logistic regression Test Accuracy: ", metrics.accuracy_score(test_y, mul_lr.predict(test_x)))

# compare results to RFC
rfc = RandomForestClassifier()
rfc.fit(train_x, train_y)

# Testing the model
predictions = rfc.predict(test_x)
print("Randon Forest Classifier: ", metrics.accuracy_score(test_y, predictions))


