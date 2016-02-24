"""
Steps to solve the problem

1. Read the train data from pandas
2. Use svm with train and test

"""

import pandas
import re
from nltk.corpus import wordnet as wn

from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def print_score_report(ingredients, classes):
    """ SPLIT THE DATASET FOR SCORING IT
     THE REAL FINAL SCORE IS DONE IN THE KAGGLE WEB, HERE
     IS ONLY A TEST
     Part the train dataset for training and testing
     Get score with cross validation
    """

    X, y = ingredients, classes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    predictor = LogisticRegression()
    predictor.fit(X_train, y_train)
    y_pred = predictor.predict(X_test)
    print classification_report(y_test, y_pred)



def singularize_words(dataset):
    """Return the dataset with all the words in singular from"""

    # Each row of the dataset is iterated by the ingredients of the recipe (items of the list),
    # many ingredients are composed of two or more words (rome tomatoes, extra-virgin olive oil)
    # so we have to iterate over each word in the ingredient, so the string is splitted,
    # and wn.morphy is applied to each word, if the word doen't exist in the nltk corpus
    # then the original word is return. After done with the words they are joined in a new string
    new_dataset = dataset.map(
        lambda row: [" ".join(map(                 # map over each word in the ingredient, then join them
            lambda item: wn.morphy(item) or item,  # each word is transformed to its basic form or the same word used
            item.split()))                         # the ingredient is splitted
                     for item in row])             # for each ingredient in the recipe

    return new_dataset


def remove_special_characters(dataset):
    """Return the dataset with all the ingredients without special characters"""

    special_characters = ['&', "'", '"', "''", "/", '%', "!", "(", ")", "\\", ',', '.']

    new_dataset = dataset

    for special_character in special_characters:
        new_dataset = new_dataset.map(lambda row: [item.replace(special_character, " ") for item in row])

    return new_dataset


def lowercase(dataset):
    """Return the dataset with all the letters lowercase"""

    new_dataset = dataset.map(lambda row: [item.lower() for item in row])

    return new_dataset


def remove_numbers(dataset):
    """Return the dataset without numbers in the ingredients"""

    new_dataset = dataset.map(lambda row: [re.sub("\d+", "", item) for item in row])

    return new_dataset



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
# Use one word ngram and two word ngrams in the vectorizer
count_vect = CountVectorizer(ngram_range=(1, 2), strip_accents='unicode')

# Preprocessing the training data
train_ingredients = lowercase(train_ingredients)
train_ingredients = remove_special_characters(train_ingredients)
train_ingredients = remove_numbers(train_ingredients)
train_ingredients = singularize_words(train_ingredients)    # Singularize each word
train_ingredients = train_ingredients.map(lambda row: [item.replace(" ", "-") for item in row])
train_ingredients = train_ingredients.map(lambda row: " ".join(row)) #join all the words
train_counts = count_vect.fit_transform(train_ingredients)   # vectorize the data

# TRAINING DATA REPORT
print_score_report(train_counts, classes)


# Preprocessing the test data
test_ingredients = lowercase(test_ingredients)
test_ingredients = remove_special_characters(test_ingredients)
test_ingredients = remove_numbers(test_ingredients)
test_ingredients = singularize_words(test_ingredients)  # Singularize each word
test_ingredients = test_ingredients.map(lambda row: [item.replace(" ", "-") for item in row])
test_ingredients = test_ingredients.map(lambda row: " ".join(row)) # Join all the words
test_counts = count_vect.transform(test_ingredients) # vectorize


# Do the prediction
clf = LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(train_counts, classes)
predictions = clf.predict(test_counts)


predict_df = pandas.DataFrame(data={'cuisine':predictions, 'id':test.id})

predict_df.to_csv("submission.csv", index=False)






