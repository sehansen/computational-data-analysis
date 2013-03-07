from scipy.io               import loadmat
from sklearn.decomposition  import PCA
from sklearn.preprocessing  import scale
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
t = np.transpose
font = {'family' : 'normal',
        'size'   : 8}
plt.rc('font', **font)

churnFile = loadmat("churn_work.mat")

churnGeneral = np.nan_to_num(churnFile["churn2_work"].newbyteorder("="))
churnGNames = churnFile["churn2_names"][0]
# print churnGNames.shape

# print churnGeneral.shape
def month_transform(m):
    if 0 < m and m < 8: return m + 12
    else: return m
Xtmp = np.concatenate([churnGeneral[:,0:4], t(np.atleast_2d(map(month_transform,churnGeneral[:,4]))), churnGeneral[:,5:]],1)
# generalPCA = PCA()

# generalPCA.fit(scale(Xtmp))

# tran = generalPCA.transform(Xtmp)

# print generalPCA.components_[0]
# print generalPCA.explained_variance_ratio_

# filter out customers with status 0
churnFilter = np.nonzero(churnGeneral[:,-2])
churnFiltered = churnGeneral[churnFilter,:][0]

idx = np.arange(Xtmp.shape[0])
np.random.shuffle(idx)

Xsmall = Xtmp[idx[:999],:]

def convert_to_days_since(a,b,c):
	years = Xsmall[:,a]
	months = Xsmall[:,b]
	days = Xsmall[:,c]

	def date2(y,m,d): 
		if y > 0 and m > 0 and d > 0: 
			return dt.date(int(y),int(m),int(d)) 
		else: 
			# Convert back to 0 later
			return dt.date(1900,1,1) 

	datevect = np.vectorize(date2)

	dates = datevect(years,months,days)
	dayssince = dates - dates.min()
	return [i.days for i in dayssince]

first_renewal = convert_to_days_since(9,8,7)
last_renewal = convert_to_days_since(13,12,11)
deactivation_month = Xsmall[:,4]

X = np.vstack([Xsmall[:,0],Xsmall[:,1],Xsmall[:,2], deactivation_month, first_renewal, last_renewal, Xsmall[:,-6],Xsmall[:,-5],Xsmall[:,-4],Xsmall[:,-3],Xsmall[:,-2],Xsmall[:,-1]])
X = t(X)
names = ['ID','Area','Credit score','Deactivation month','First renewal', 'Last renewal', 'PAYMENT TYPE','PENALTIES FOR NON PAYMENT', 'RATE PLAN','RATE PLAN CHANGES', 'STATUS', 'VALUE SEGMENT']
for i in xrange(1,13):
            plt.subplot(4,3,i)
            plt.plot(X[X[:,10]==1,0],X[X[:,10]==1,i-1],'r.',X[X[:,10]==0,0],X[X[:,10]==0,i-1],'b.')
            # plt.hist(X[:,i], 50, normed=1, facecolor='green', alpha=0.75)
            plt.xticks([])
            plt.yticks([])
            plt.title(names[i-1])
plt.savefig('init2.png')

