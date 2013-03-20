import h5py
from sklearn import tree

f = h5py.File('dset.h5', 'r')

clf = tree.DecisionTreeClassifier()
clf.fit(f['Months'][:81495], f['General'][:,18])