import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
#import nltk

def load_dataset(path, text_col_name='NOTE', val_percentage=0.1, random_seed=123):
    df = pd.read_csv(path, sep='|')

    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,10))
    X = vectorizer.fit_transform(df[text_col_name])
    y = df['LABEL']

    # Shuffle together
    np.random.seed(random_seed)
    p = np.random.permutation(X.shape[0])
 
    X = X[p]
    y = y[p]
  
    return X, y

if __name__ == "__main__":
    X, y = load_dataset('../labeled_radiology_reports.csv')
    NBclassifier = MultinomialNB()
    scores = cross_val_score(NBclassifier, X, y, cv=5)
    results = np.unique(sklearn.model_selection.cross_val_predict(NBclassifier, X, y, cv=5), return_counts=True)
    print(results[1][1]/np.sum(results[1]))

    print(np.mean(scores))
