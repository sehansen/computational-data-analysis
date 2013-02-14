from scipy.io        import loadmat
from numpy.linalg    import lstsq, inv, solve
from numpy           import logspace, eye, dot, mean, array, transpose, ones, log
t = transpose
import matplotlib.pyplot as plt

resSumSq = lambda r: sum(r*r)

def lda(X, y):
	n,p = X.shape
	X0 = array([X[i] for i in xrange(n) if y[i] == 0])
	X1 = array([X[i] for i in xrange(n) if y[i] == 1])

	plt.figure()
	plt.plot(X0[:,0],X0[:,1],"r.")
	plt.plot(X1[:,0],X1[:,1],"b.")

	n0,  n1  = len(X0), len(X1)
	pi0, pi1 = (1.0*n0)/n,    (1.0*n1)/n
	mu0, mu1 = mean(X0,0), mean(X1,0)
	Sigma1 = dot(t(X0 - ones((n0,1))*t(mu0)),X0 - ones((n0,1))*t(mu0))
	Sigma2 = dot(t(X1 - ones((n1,1))*t(mu1)),X1 - ones((n1,1))*t(mu1))
	Sigma = (Sigma1 +  Sigma2)/(n - 2)
	a = solve(Sigma,mu0 - mu1)
	b = log(pi0/pi1) - 0.5*dot(t(mu0 + mu1),solve(Sigma,mu0 - mu1))
	plt.plot(X[:,0], -a[0]/a[1]*X[:,0] - b/a[1],"k")
	return (a,b)


diabetesMat = loadmat("diabetes.mat")
diaX = diabetesMat["X"].newbyteorder("=")
diay = diabetesMat["y"].newbyteorder("=")

ldaMat = loadmat("data_m1_3.mat")
ldaX = ldaMat["X"].newbyteorder("=")
lday = ldaMat["y"].newbyteorder("=")

if __name__ == "__main__":
	pure =  lstsq(diaX, diay)
	print """Week 1:\n Ex 1.a:\n   Found model as\n%s\n""" % (pure[0])
	print """ Ex 1.b\n   Add a constant column to X\n"""
	RSS = resSumSq(dot(diaX,pure[0])-diay)
	TSS = sum((mean(diay) - diay)**2)
	print """ Ex 1.c\n   RSS: %.0f, TSS: %.0f, r^2: %.3f\n\n""" % (RSS,TSS,1-RSS/TSS)


	print """ Ex 2.a\n   Uden udledning:\n   \\beta=(X'.X+\lambda.I)^-1.X'.y\n"""

	betas  = []
	resses = []
	RSSs   = []
	for i in logspace(-4,3,100):
		beta = lstsq(diaX + i*eye(diaX.shape[0], diaX.shape[1]), diay)
		betas.append(beta)
		res = dot(diaX,beta[0])-diay
		resses.append(res)
		RSS = resSumSq(res)
		RSSs.append(RSS)
	print """ Ex 2.b\n   Se plot"""
	plt.semilogx(logspace(-4,3,100),array(map(lambda l: l[0],betas))[:,:,0])

	print """ Ex 3\n    %s""" % (lda(ldaX, lday),)

	print """ Ex 4.a\n   %s""" ("Suitable number of classes")

	plt.show()