import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
#import nltk

def load_dataset(path, text_col_name='text', val_percentage=0.1, random_seed=123):
    df = pd.read_csv(path, sep='|')

    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,2))
    X = vectorizer.fit_transform(df[text_col_name])
    y = df['label']

    # Shuffle together
    np.random.seed(random_seed)
    p = np.random.permutation(X.shape[0])
 
    X = X[p]
    y = y[p]
  
    return X, y

if __name__ == "__main__":
    X, y = load_dataset('../new_labeled_reports_full_preprocessed.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    NBclassifier = MultinomialNB()
    NBclassifier.fit(X_train, y_train)
    print(NBclassifier.score(X_test, y_test))
    print(confusion_matrix(y_test, NBclassifier.predict(X_test), labels=[0,1]))
    LRclassifier = LogisticRegression(solver='liblinear')
    LRclassifier.fit(X_train, y_train)
    print(LRclassifier.score(X_test, y_test))
    print(confusion_matrix(y_test, LRclassifier.predict(X_test), labels=[0,1]))
    '''
    scores = cross_val_score(NBclassifier, X, y, cv=5)
    results = np.unique(sklearn.model_selection.cross_val_predict(NBclassifier, X, y, cv=5), return_counts=True)
    print(results)

    print(np.mean(scores))
    '''
