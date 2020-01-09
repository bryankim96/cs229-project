import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB


def load_dataset(path, text_col_name='REPORT', random_seed=123):
    df = pd.read_csv(path, sep='|')

    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 1))
    X = vectorizer.fit_transform(df[text_col_name])
    y = df['LABEL']

    # Shuffle together
    np.random.seed(random_seed)
    p = np.random.permutation(X.shape[0])

    X = X[p]
    y = y[p]

    return X, y, vectorizer.get_feature_names()


if __name__ == "__main__":
    X, y, feature_names = load_dataset('../labeled_radiology_reports.csv', text_col_name='NOTE')
    NBclassifier = MultinomialNB(class_prior=[0.5, 0.5], fit_prior=False)
    scores = cross_val_score(NBclassifier, X, y, cv=5)
    print(scores)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=123)
    NBclassifier.fit(X_train, y_train)

    sorted_features = sorted(zip(feature_names, NBclassifier.coef_[0]), key=lambda x: x[1], reverse=True)

    print("Most predictive (of positive response):", sorted_features[:10])
    print("Least predictive (of positive response):", sorted_features[-10:])

    y_pred = NBclassifier.predict(X_val)
    y_true = y_val

    print("Precision: ", precision_score(y_true, y_pred))
    print("Recall: ", recall_score(y_true, y_pred))
    print("F1 score: ", f1_score(y_true, y_pred))

    print("Confusion matrix: ", confusion_matrix(y_true, y_pred, labels=[1, -1]))

    print(NBclassifier.score(X_val, y_val))

    print("Logistic Regression:\n")

    LRclassifier = LogisticRegression(solver='liblinear')
    LRclassifier.fit(X_train, y_train)
    print(LRclassifier.score(X_val, y_val))
    print(confusion_matrix(y_val, LRclassifier.predict(X_val), labels=[0, 1]))
