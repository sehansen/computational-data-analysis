import h5py
import matplotlib.pyplot as plt
from numpy import array, concatenate, corrcoef, hstack, ones, vstack, zeros
import random
from sklearn import tree
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
# from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import StringIO

def sample_wr(population, k):
    "Chooses k random elements (with replacement) from a population"
    n = len(population)
    _random, _int = random.random, int  # speed hack
    return [_int(_random() * n) for i in xrange(k)]

dsetf = h5py.File('dset2.h5', 'r')
tmpf = h5py.File('tmp.h5', 'w')

def makeDifferencedData():
    f = h5py.File('dset.h5', 'r')

    d = {}
    for x in f['Months']:
        d[x[0]] = []

    for x in f['Months']:
        d[x[0]].append(x[1:])

    for x in f['General']:
        d[x[0]].append(x[18])

    l = []
    for x, y in d.iteritems():
        l.append([x, y[0], y[1], y[2], y[1] - y[0], y[2] - y[1], y[3]])
        # Should be sliced so that the exact month is ignored

    a = []
    for x in l:
        a.append(concatenate([array([x[0]])] + x[1:-1] + [array([x[-1]])]))

    b = vstack(a)

    tmpf['Full'] = b


def sample():
    activeSample = sample_wr(xrange(dsetf['ActiveIDs'].shape[0]),15000)
    activeSample.sort()
    datasample = array(dsetf['ActiveData'])[activeSample,1:]
    tmpf['X'] = vstack([datasample, dsetf['ChurnData'][:,1:], dsetf['ChurnData'][:,1:], dsetf['ChurnData'][:,1:]])
    tmpf['y'] = concatenate([ones((15000,1)), zeros((4762*3,1))])

def test():
    clf = RandomForestClassifier(n_estimators=140)
    # clf = DummyClassifier(strategy='most_frequent')
    # clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0)
    # clf = SVC()
    X_train, X_test, y_train, y_test = train_test_split(tmpf['X'],
        tmpf['y'], test_size=0.4, random_state=0)

    clf.fit(X_train, y_train)
    return (clf, clf.score(X_test, y_test))
    # return cross_val_score(clf, tmpf['X'], tmpf['y'], cv=5)
    # return clf.fit(tmpf['X'], tmpf['y'])

def gridsearch():
    X_train, X_test, y_train, y_test = train_test_split(tmpf['X'],
        tmpf['y'], test_size=0.4, random_state=0)


    tuned_parameters = [{'n_estimators': [1, 3, 10, 32, 100, 316, 1000]}]
    clf = GridSearchCV(RandomForestClassifier(), tuned_parameters,,parallel verbose=4)
    clf.fit(X_train, y_train, cv=5)

    print "Best estimator:", clf.best_estimator_

    print "Grid scores on development set:"
    print
    for params, mean_score, scores in clf.grid_scores_:
        print "%0.3f (+/-%0.03f) for %r" % (
            mean_score, scores.std() / 2, params)


    print "Detailed classification report:"
    print
    print "The model is trained on the full development set."
    print "The scores are computed on the full evaluation set."
    print
    y_true, y_pred = y_test, clf.predict(X_test)
    print classification_report(y_true, y_pred)


def corplot():
    tmpf['C'] = hstack([tmpf['X'], tmpf['y']])
    plt.figure()
    plt.imshow(corrcoef(tmpf['C'], rowvar=0), interpolation="nearest")

# clf = tree.DecisionTreeClassifier(min_samples_leaf=10)
# clf.fit(f['Months'][:81495], f['General'][:,18])

# with open("leaf10.dot", 'w') as f:
#     f = tree.export_graphviz(clf, out_file=f)
