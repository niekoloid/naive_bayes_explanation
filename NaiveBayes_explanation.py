# -*- coding: utf-8 -*-
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing

# Load the dataset
train_data = pd.read_csv("sentences_sports.csv")

######################################
### Vectorizing Training Sentences ###
######################################

# About X
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(train_data['Text'].values).toarray()
X_head = vectorizer.get_feature_names()
df_X = pd.DataFrame(data=X, columns=X_head)

# About Y
label_encoder = preprocessing.LabelEncoder()
Y = label_encoder.fit_transform(train_data['Category'].values)
Y_head = label_encoder.classes_
df_Y = pd.DataFrame(data=Y, columns=["Category"])

# Presenting the vectorized training data
print(df_X.join(df_Y))

#%%
##################################
### Training Naive Bayes Model ###
##################################

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X, Y)

#%%
#################################
### Prediction by Naive Bayes ###
#################################

test_data = ["A very close game"]

vectorizer = CountVectorizer()
vec_test = vectorizer.fit_transform(test_data).toarray()
vec_test
vectorizer.get_feature_names()

df_tmp = df_X.drop(range(0,len(df_X)))
df_test = pd.DataFrame(data=vec_test, columns=vectorizer.get_feature_names(), index=["test"])
df_test = pd.concat([df_tmp, df_test]).fillna(0).astype('int')
df_test

# Prediction
clf.predict(df_test)
clf.predict_log_proba(df_test)
#clf.predict_proba(df_test)

#%%
clf.predict_log_proba(X)

#%%
clf.predict_proba(X)

#%%
clf.predict(X)

#%%
#clf.get_params()