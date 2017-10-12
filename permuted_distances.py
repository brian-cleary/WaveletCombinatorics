import sys
import numpy as np
from create_wavelet_clusters import wavelet_levels,wavelet_level_correlations

def permuted_distances(X):
	X_permute = np.copy(X).T
	day_sums = X_permute.sum(0)
	np.random.seed()
	for x in X_permute:
		np.random.shuffle(x)
	X_permute = X_permute/X_permute.sum(0)*day_sums
	X_permute = X_permute.T
	L,resolution = wavelet_levels(X_permute)
	D = wavelet_level_correlations(L)
	return D

def distance_empirical_pval_counts(D,Dshuffle,n=50.):
	counts = np.zeros(D.shape)
	counts_shuffle = np.zeros(D.shape)
	for _ in range(int(n)):
		D_permute = permuted_distances(X)
		counts += (D_permute <= D)
		counts_shuffle += (D_permute <= Dshuffle)
	return counts/n,counts_shuffle/n

Xpath,D1path,D2path,outpath = sys.argv[1:]
X = np.load(Xpath).T
D = np.load(D1path)
Dshuffle = np.load(D2path)
counts,counts_shuffle = distance_empirical_pval_counts(D,Dshuffle)
np.save('%s.original.npy' % outpath,counts)
np.save('%s.shuffle.npy' % outpath,counts_shuffle)




import numpy as np
import numexpr as ne
import glob,os
def get_distance_pvals(type):
	FP = glob.glob(os.path.join('/broad/hptmp/bcleary/wave_permute/','*.%s.npy' % type))
	p_values = np.load(FP[0])
	for fp in FP:
		pv = np.load(fp)
		p_values += pv
	return p_values/len(FP)

def fdr(PV,q):
	p_values = PV.flatten()
	accept = np.zeros(len(p_values))
	for p in sorted(set(p_values[(p_values < q)]),reverse=True):
		n = ne.evaluate("(p_values <= p) | (p_values >= 1-p)").sum()
		if 2*p*len(p_values)/n < q:
			break
	if 2*p*len(p_values)/n < q:
		accept[ne.evaluate("(p_values <= p) | (p_values >= 1-p)")] = 1.
	return accept.reshape(PV.shape)

Dshuffle = np.load('wavelet.distance.shuffle.npy')
PV = get_distance_pvals('shuffle')
np.save('wavelet.distance.pval.shuffle.npy',PV)
idx = ((Dshuffle > 1.2)+(Dshuffle < 0.6))
FDR10 = np.zeros(PV.shape)
fdr10 = fdr(PV[idx],0.1)
FDR10[idx] = fdr10
np.save('wavelet.distance.fdr10.shuffle.npy',FDR10)

D = np.load('wavelet.distance.npy')
PV = get_distance_pvals('original')
np.save('wavelet.distance.pval.npy',PV)
idx = ((D > 1.2)+(D < 0.6))
FDR10 = np.zeros(PV.shape)
fdr10 = fdr(PV[idx],0.1)
FDR10[idx] = fdr10
np.save('wavelet.distance.fdr10.npy',FDR10)

# plot p-value distributions
P1 = np.load('wavelet.distance.pval.npy')
P2 = np.load('wavelet.distance.pval.shuffle.npy')
# positive correlations on the right...
P1 = 1-P1
P2 = 1-P2
h1 = plt.hist(P1.flatten(),bins=5000,label='original data',linewidth=0)
h2 = plt.hist(P2.flatten(),bins=5000,label='permuted data',linewidth=0,alpha=0.5)
plt.legend()
plt.savefig('wavelet.distance.pval.hist.pdf')
plt.close()

# plot the distribution of p-values using the old thresholds
from scipy.stats import gaussian_kde
P = np.copy(PV)
P[(P > 0.5)] = 1 - P[(P > 0.5)]
pmin = P[(P > 0)].min()
P[(P <= 0)] = pmin
P = -np.log(P)
idx = ((D > 1.2)+(D < 0.6))
d1 = gaussian_kde(P[idx])
d2 = gaussian_kde(P[np.invert(idx)])
xs = np.linspace(0,10,200)
d1.covariance_factor = lambda : .25
d1._compute_covariance()
d2.covariance_factor = lambda : .25
d2._compute_covariance()
d1xs = d1(xs)
d2xs = d2(xs)
_=plt.plot(xs,d1xs,label='Previously included')
_=plt.plot(xs,d2xs,label='Not included')
_=plt.legend()
_=plt.xlabel('-log(p-value)')
plt.savefig('p-value.distributions.pdf')
plt.close()