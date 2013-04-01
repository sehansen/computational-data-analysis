import h5py
from numpy import array, concatenate, vstack
from sklearn import tree
import StringIO

def sample_wr(population, k):
    "Chooses k random elements (with replacement) from a population"
    n = len(population)
    _random, _int = random.random, int  # speed hack
    return [_int(_random() * n) for i in xrange(k)]


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


# clf = tree.DecisionTreeClassifier(min_samples_leaf=10)
# clf.fit(f['Months'][:81495], f['General'][:,18])

# with open("leaf10.dot", 'w') as f:
#     f = tree.export_graphviz(clf, out_file=f)
