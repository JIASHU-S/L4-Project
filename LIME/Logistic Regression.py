import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer


categories = ['alt.atheism', 'soc.religion.christian']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

vectorizer = TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(newsgroups_train.data)
test_vectors = vectorizer.transform(newsgroups_test.data)

clf = LogisticRegression()
clf.fit(train_vectors, newsgroups_train.target)

pipeline = make_pipeline(vectorizer, clf)

explainer = LimeTextExplainer(class_names=newsgroups_train.target_names)

idx = 42  # Index of the text instance in the test data
text_instance = newsgroups_test.data[idx]
exp = explainer.explain_instance(text_instance, pipeline.predict_proba, num_features=6)

print('Explanation for index {}:'.format(idx))
print('\n'.join(map(str, exp.as_list())))
