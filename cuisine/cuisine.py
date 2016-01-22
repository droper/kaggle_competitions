"""
Steps to solve the problem

1. Read the train data from pandas
2. Use svm with train and test

"""

import pandas
from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

TRAIN_PATH = "/home/pedro/proyectos/data-science/cuisine/train.json"
TEST_PATH = "/home/pedro/proyectos/data-science/cuisine/test.json"

train = pandas.read_json(TRAIN_PATH)
test = pandas.read_json(TEST_PATH)

# Convert the column to list, to a set to eliminate duplicates and to list again
train_ingredients = train["ingredients"]
test_ingredients = test["ingredients"]
classes = train["cuisine"]

count_vect = CountVectorizer()

# List to strings
train_ingredients = train_ingredients.map(lambda row: [item.replace(" ", "") for item in row])
train_ingredients = train_ingredients.map(lambda row: " ".join(row))
train_counts = count_vect.fit_transform(train_ingredients)

# Get score with cross validation
X, y = train_counts, classes
#models = [LogisticRegression(), MultinomialNB(), BernoulliNB(), SVC(), DecisionTreeClassifier() ]
models = [MultinomialNB(fit_prior=False, alpha=0.5)]
final_model = {}

# Iterate over the models to validate with each model
# The best model will be used to submit
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


test_ingredients = test_ingredients.map(lambda row: [item.replace(" ", "") for item in row])
test_ingredients = test_ingredients.map(lambda row: " ".join(row))
test_counts = count_vect.transform(test_ingredients)

#clf = MultinomialNB().fit(train_counts, classes)
#clf = SVC().fit(train_counts, classes)
clf = LogisticRegression().fit(train_counts, classes)
predictions = clf.predict(test_counts)


predict_df = pandas.DataFrame(data={'cuisine':predictions, 'id':test.id})

predict_df.to_csv("submission.csv", index=False)




