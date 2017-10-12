import sys, getopt, os
import numpy as np
import pywt
from scipy.spatial import distance
from scipy import stats
from collections import defaultdict
import itertools


def wavelet_levels(Y):
	w = pywt.Wavelet('sym2')
	levels = pywt.dwt_max_level(Y.shape[0],w)
	w0 = pywt.wavedec(Y[:,0],w,level=levels)[1:]
	L = [np.empty((Y.shape[1],len(x))) for x in w0]
	for i in range(Y.shape[1]):
		wd = pywt.wavedec(Y[:,i],w)[1:]
		for j,x in enumerate(wd):
			L[j][i,:] = x
	return L,[Y.shape[0]/len(x) for x in w0]

def wavelet_level_correlations(L):
	D = np.array([distance.pdist(l,'cosine') for l in L])
	return D

def weighted_correlation(D,R,mu,sig):
	density = stats.norm(loc=mu,scale=sig)
	weights = [density.pdf(r) for r in R]
	weights = weights/np.linalg.norm(weights)
	Dw = np.empty(D[0].shape)
	for i,w in enumerate(weights):
		Dw += w*D[i]
	return Dw

def distance_combinatorics(Dorig,FDR,resolution,n,th,tl,as_str=True,mode=0,res_diff=1.):
	D = np.copy(Dorig)
	D[((D > tl) & (D < th))] = 1
	D[(FDR == 0)] = 1
	Dmerged = defaultdict(list)
	Dmax = np.zeros(D.shape[1])
	for low,high in itertools.combinations(range(D.shape[0]),2):
		if resolution[low]/resolution[high] < res_diff:
			Dcurrent = np.zeros(D.shape[1])
		elif mode == 1:
			# positive low, negative high
			# zeros where one or both signs incorrect
			Dcurrent = (1./(high - low))/(1./4 + 1./(high - low))*((D[high] - 1)*(1 - D[low]))**.5
			Dcurrent[np.invert((1 - D[high] < 0)*(1 - D[low] > 0))] = 0
		elif mode == 2:
			# positive low, positive high
			Dcurrent = (1./4 + 1./(high - low))/(1./(high - low))*((1 - D[high])*(1 - D[low]))**.5
			Dcurrent[np.invert((1 - D[high] > 0)*(1 - D[low] > 0))] = 0
		Dmax = np.array([Dmax,Dcurrent]).max(0)
		#Dmax[np.isnan(Dmax)] = 0
		del Dcurrent
	for i,idx in enumerate(itertools.combinations(xrange(n),2)):
		d = Dmax[i]
		if d > .0:
			if as_str:
				Dmerged[idx[0]].append('%d:%f' % (idx[1],d))
				Dmerged[idx[1]].append('%d:%f' % (idx[0],d))
			else:
				Dmerged[idx[0]].append((idx[1],d))
				Dmerged[idx[1]].append((idx[0],d))
	return Dmerged

mclheader = "(mclheader\nmcltype matrix\ndimensions NUMxNUM\n)\n(mclmatrix\nbegin\n"
def write_merged(Dm,n,x,outfile):
	f = open(outfile,'w')
	f.write(mclheader.replace('NUM',str(n)))
	for k,v in Dm.iteritems():
		f.write(' '.join(['%d:%d' % (k,x)] + v + ['$\n']))
	f.write(')\n')
	f.close()

help_message = 'usage example: python create_wavelet_clusters.py -i all_data.npy -d wavelet.distance.npy -q wavelet.distance.fdr.npy -o wavelets/'
if __name__ == "__main__":
	try:
		opts, args = getopt.getopt(sys.argv[1:],'hi:d:q:o:m:a',["outputdir=","mode="])
	except:
		print help_message
		sys.exit(2)
	distancepath = None
	fdrpath = None
	use_differences = False
	for opt, arg in opts:
		if opt in ('-h','--help'):
			print help_message
			sys.exit()
		elif opt in ('-i'):
			datapath = arg
		elif opt in ('-d'):
			distancepath = arg
		elif opt in ('-q'):
			fdrpath = arg
		elif opt in ('-o','--outputdir'):
			outputdir = arg
		elif opt in ('-m','--mode'):
			mode = int(arg)
		elif opt in ('-a'):
			use_differences = True
	X = np.load(datapath).T
	L,resolution = wavelet_levels(X)
	if use_differences:
		Xd = X[1:]-X[:-1]
		Ld,rd = wavelet_levels(Xd)
		L[-1] = Ld[-1]
	if distancepath == None:
		D = wavelet_level_correlations(L)
	else:
		D = np.load(distancepath)
	if fdrpath == None:
		FDR = np.ones(D.shape,dtype=np.bool)
	else:
		FDR = np.load(fdrpath)
	if mode == 1:
		Dmerged = distance_combinatorics(D,FDR,resolution,X.shape[1],1.3,0.5,mode=mode,res_diff=1.)
	elif mode == 2:
		Dmerged = distance_combinatorics(D,FDR,resolution,X.shape[1],1.2,0.6,mode=mode,res_diff=1.)
	del D
	write_merged(Dmerged,X.shape[1],resolution[0]/2 + resolution[-1]/2,outputdir+'merged.mci')
	for i in np.linspace(1.1,3,20):
		os.system('mcl %s/merged.mci -I %f -o %s/out.merged.mci.I%s' % (outputdir,i,outputdir,str(i).replace('.','')))


