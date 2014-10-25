#!/usr/bin/env python

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
	w0 = pywt.wavedec(Y[:,0],w,level=levels)
	L = [np.empty((Y.shape[1],len(x))) for x in w0]
	for i in range(Y.shape[1]):
		wd = pywt.wavedec(Y[:,i],w)
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

def distance_combinatorics(D,n,th,tl):
	D[((D > tl) & (D < th))] = 1
	Dmerged = defaultdict(list)
	Dmax = np.zeros(D.shape[1])
	for low,high in itertools.combinations(range(D.shape[0]),2):
		Dcurrent = (1./(high - low))/(1./4 + 1./(high - low))*((D[high] - 1)*(1 - D[low]))**.5
		Dmax = np.array([Dmax,Dcurrent]).max(0)
		del Dcurrent
	for i,idx in enumerate(itertools.combinations(xrange(n),2)):
		d = Dmax[i]
		if d > .0:
			Dmerged[idx[0]].append('%d:%f' % (idx[1],d))
			Dmerged[idx[1]].append('%d:%f' % (idx[0],d))
	return Dmerged

def merged_distance(Dh,Dl,n,th,tl):
	thresh_high = np.percentile(Dh,th)
	thresh_low = np.percentile(Dl,tl)
	Dmerged = defaultdict(list)
	for i,idx in enumerate(itertools.combinations(xrange(n),2)):
		if (Dhigh[i] > thresh_high) and (Dlow[i] < thresh_low):
			d = .5*((1-Dlow[i]) + (Dhigh[i]-1))
			Dmerged[idx[0]].append('%d:%f' % (idx[1],d))
			Dmerged[idx[1]].append('%d:%f' % (idx[0],d))
	return Dmerged

mclheader = "(mclheader\nmcltype matrix\ndimensions NUMxNUM\n)\n(mclmatrix\nbegin\n"
def write_merged(Dm,n,x,outfile):
	f = open(outfile,'w')
	f.write(mclheader.replace('NUM',str(n)))
	for k,v in Dm.iteritems():
		f.write(' '.join(['%d:%d' % (k,x)] + v + ['$\n']))
	f.write(')\n')
	f.close()

high_mu = 2
high_sig = 1
low_mu = 15
low_sig = 5
help_message = 'usage example: python create_wavelet_clusters.py -i all_data.npy -o wavelets/'
if __name__ == "__main__":
	try:
		opts, args = getopt.getopt(sys.argv[1:],'hr:i:o:',["inputdir=","outputdir="])
	except:
		print help_message
		sys.exit(2)
	for opt, arg in opts:
		if opt in ('-h','--help'):
			print help_message
			sys.exit()
		elif opt in ('-i','--inputdir'):
			inputdir = arg
		elif opt in ('-o','--outputdir'):
			outputdir = arg
	X = np.load(inputdir)
	L,resolution = wavelet_levels(X)
	for i,l in enumerate(L[1:]):
		np.save('%s/wavelet.resolution%d.details.npy' % (outputdir,resolution[i+1]),l)
	np.save('%s/wavelet.resolution%d.approximation.npy' % (outputdir,resolution[0]),L[0])
	D = wavelet_level_correlations(L)
	del L
	#for i,d in enumerate(D[1:]):
	#	np.save('%s/distance.resolution%d.details.npy' % (outputdir,resolution[i+1]),d)
	#np.save('%s/distance.resolution%d.approximation.npy' % (outputdir,resolution[0]),D[0])
	#Dhigh = weighted_correlation(D,resolution,high_mu,high_sig)
	#Dlow = weighted_correlation(D,resolution,low_mu,low_sig)
	#del D
	#Dmerged = merged_distance(Dhigh,Dlow,X.shape[1],90,10)
	Dmerged = distance_combinatorics(D,X.shape[1],1.5,.3)
	del D
	write_merged(Dmerged,X.shape[1],resolution[0]/2 + resolution[-1]/2,outputdir+'merged.mci')
	for i in [1.6,1.8,2.0,2.4,2.8]:
		os.system('mcl %s/merged.mci -I %d -o %s/out.merged.mci.I%s' % (outputdir,i,outputdir,str(i).replace('.','')))


