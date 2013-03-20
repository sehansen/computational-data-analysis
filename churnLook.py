from numpy                  import nan_to_num, concatenate, atleast_2d, transpose,array,nonzero
from scipy.io               import loadmat
from sklearn.decomposition  import PCA
from sklearn.preprocessing  import scale
import argparse
import matplotlib.pyplot as plt
t = transpose
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 8}
plt.rc('font', **font)

churnFile = loadmat("churn_work.mat")

churnGeneral = nan_to_num(churnFile["churn2_work"].newbyteorder("="))
churnGNames = churnFile["churn2_names"][0]
# print churnGNames.shape

# print churnGeneral.shape
def month_transform(m):
    if 0 < m and m < 8: return m + 12
    else: return m
churnGeneralMod = concatenate([churnGeneral[:,0:4], t(atleast_2d(map(month_transform,churnGeneral[:,4]))), churnGeneral[:,5:]],1)
generalPCA = PCA()

generalPCA.fit(scale(churnGeneralMod))

tran = generalPCA.transform(churnGeneralMod)

# print generalPCA.components_[0]
# print generalPCA.explained_variance_ratio_

# filter out customers with status 0
churnFilter = nonzero(churnGeneral[:,-2])
churnFiltered = churnGeneral[churnFilter,:][0]
churnFiltered.view('float64,float64,float64,float64,float64,float64,float64,float64,float64,float64,float64,float64,float64,float64,float64,float64,float64,float64,float64,float64').sort(order=['f0'], axis=0)
# for i in xrange(1,20):
#             plt.subplot(4,5,i+1)
#             plt.plot(churnFiltered[:,0],churnFiltered[:,i],'.')
#             plt.xticks([])
#             plt.yticks([])
#             plt.title(churnGNames[i])
# plt.figure()
# churnFilter = nonzero(1 - churnGeneral[:,-2])
# churnFiltered = churnGeneral[churnFilter,:][0]
# for i in xrange(1,20):
#             plt.subplot(4,5,i+1)
#             plt.plot(churnFiltered[:,0],churnFiltered[:,i],'r.')
#             plt.xticks([])
#             plt.yticks([])
#             plt.title(churnGNames[i][0])

# # for i in xrange(7):
# #     for j in xrange(7):
# #         if i != j:
# #             plt.subplot(20,20,i*20+j+1)
# #             plt.plot(churnGeneral[:,i],churnGeneral[:,j],'.')
# #             plt.xticks([])
# #             plt.yticks([])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Churn dataset processing - nonsampled version')
    parser.add_argument('-s', '--show', action="store_true",
                       help='an integer for the accumulator')

    args = parser.parse_args()
    if args.show: plt.show()
    else: plt.savefig("initplot.png")
