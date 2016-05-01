import six.moves.cPickle as pkl
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import LinearSVC

def load_data():
    f = open('./data/cleaned.pkl','rb')
    train = pkl.load(f)
    f.close()
    raw_train = train[0]
    x_train = train[1]
    y_train = train[2]

    f = open('./data/test.pkl','rb')
    test = pkl.load(f)
    f.close()
    raw_test = test[0]
    x_test = test[1]

    f = open('./data/dict.pkl','rb')
    dictionary = pkl.load(f)
    f.close()
    return raw_train, raw_test, x_train, y_train, x_test

def prepare():
    raw_train, raw_test, x_train, y_train, test = load_data()
    raw_train = [' '.join(i) for i in raw_train]
    raw_test = [' '.join(i) for i in raw_test]
    return raw_train, y_train, raw_test

def train(raw_train, y_train):
    tfidf = TfidfVectorizer()
    param_grid = {'vect__ngram_range': [(1,1),(1,2)],
                    'clf__penalty': ['l1', 'l2'],
                    'clf__C':[i*0.1 for i in range(1,100)]
                    }
    tfidf_clf = Pipeline([('vect', tfidf),
                         ('clf', LinearSVC())])

    gs_svm_tfidf = GridSearchCV(tfidf_clf,
                                param_grid,
                                scoring='accuracy',
                                cv=5,
                                verbose=1,
                                n_jobs=-1)

    gs_svm_tfidf.fit(raw_train, y_train)

    print('Best parameter set: %s ' % gs_svm_tfidf.best_params_)
    print('CV Accuracy: %.3f' % gs_svm_tfidf.best_score_)
    return gs_svm_tfidf

def predict(raw_test, gs_svm_tfidf):
    clf = gs_svm_tfidf.best_estimator_
    result = gs_svm_tfidf.predict(raw_test)
    return result

def write2csv(result):
    wt = pd.DataFrame()
    for i in range(len(result)):
        wt = wt.append([[i, result[i]]])
    wt.columns = ['id','labels']
    wt.to_csv('./data/result_SVM.csv',index=False)

if __name__== '__main__':
    raw_train, y_train, raw_test = prepare()
    gs_svm_tfidf = train(raw_train, y_train)
    result = predict(raw_test, gs_svm_tfidf)
    write2csv(result)
