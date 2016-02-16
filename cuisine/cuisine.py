"""
Steps to solve the problem

1. Read the train data from pandas
2. Use svm with train and test

"""

import pandas
from nltk.corpus import wordnet as wn

from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def singularize_words(dataset):
    """Return the dataset with all the words in singular from"""

    # Each row of the dataset is iterated by the ingredients of the recipe (items of the list),
    # many ingredients are composed of two or more words (rome tomatoes, extra-virgin olive oil)
    # so we have to iterate over each word in the ingredient, so the string is splitted,
    # and wn.morphy is applied to each word, if the word doen't exist in the nltk corpus
    # then the original word is return. After done with the words they are joined in a new string
    singular_dataset = dataset.map(
        lambda row: [" ".join(map(                 # map over each word in the ingredient, then join them
            lambda item: wn.morphy(item) or item,  # each word is transformed to its basic form or the same word used
            item.split()))                         # the ingredient is splitted
                     for item in row])             # for each ingredient in the recipe

    return singular_dataset


TRAIN_PATH = "/home/pedro/proyectos/data-science/cuisine/train.json"
TEST_PATH = "/home/pedro/proyectos/data-science/cuisine/test.json"

# Load the datasets
train = pandas.read_json(TRAIN_PATH)
test = pandas.read_json(TEST_PATH)

# Read the data from the datasets
train_ingredients = train["ingredients"]
test_ingredients = test["ingredients"]
classes = train["cuisine"]

# CountVector for vectorize the data
count_vect = CountVectorizer()

# Preprocessing the training data
train_ingredients = singularize_words(train_ingredients)    # Singularize each word
#train_ingredients = train_ingredients.map(lambda row: [item.replace(" ", "") for item in row])
train_ingredients = train_ingredients.map(lambda row: " ".join(row)) #join all the words
train_counts = count_vect.fit_transform(train_ingredients)   # vectorize the data


#models = [LogisticRegression(), MultinomialNB(fit_prior=False, alpha=0.5),
#  BernoulliNB(), SVC(), DecisionTreeClassifier() ]
#models = [LogisticRegression()]
#final_model = {}

# Iterate over the models to validate with each model
# The best model will be used to submit
"""
for model in models:
    print model.__class__
    validations = cross_validation.cross_val_score(model, X, y, scoring='accuracy')
    print validations
    avg_validations = sum(validations)/len(validations)

    if len(final_model) > 0:
        if final_model['avg_validations'] < avg_validations:
            final_model['model'] = model
            final_model['avg_validations'] = avg_validations
    else:
        final_model['model'] = model
        final_model['avg_validations'] = avg_validations

print final_model
"""

# SPLIT THE DATASET FOR SCORING IT
# THE REAL FINAL SCORE IS DONE IN THE KAGGLE WEB, HERE
# IS ONLY A TEST
# Part the train dataset for training and testing
# Get score with cross validation
X, y = train_counts, classes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

logistic = LogisticRegression()
logistic.fit(X_train, y_train)
y_pred = logistic.predict(X_test)
print classification_report(y_test, y_pred)

# END OF THE SCORING

# Preprocessing the test data
test_ingredients = singularize_words(test_ingredients)  # Singularize each word
#test_ingredients = test_ingredients.map(lambda row: [item.replace(" ", "") for item in row])
test_ingredients = test_ingredients.map(lambda row: " ".join(row)) # Join all the words
test_counts = count_vect.transform(test_ingredients) # vectorize

clf = LogisticRegression().fit(train_counts, classes)
predictions = clf.predict(test_counts)


predict_df = pandas.DataFrame(data={'cuisine':predictions, 'id':test.id})

predict_df.to_csv("submission.csv", index=False)






