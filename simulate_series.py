import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.spatial import distance
import itertools
from collections import defaultdict
from create_wavelet_clusters import wavelet_levels,wavelet_level_correlations,distance_combinatorics

# nonlinear dynamics in coupled difference eqns
# L. Lloyd, J. Theor. Biol. 173 , 217 (1995) and Sugihara et al, Science (2012)
def nonlinear_ab(n,b1=0.02,b2=0.1,p=1,noise_scale=0.1):
	X = np.linspace(1,n,n)
	Y = np.zeros((len(X),2))
	Y[:p,0] = np.random.rand(1,p)
	Y[:p,1] = Y[0,0]
	for x in range(p,len(X)):
		Y[x,0] = Y[x-p,0]*(3.8 - 3.8*Y[x-p,0] - b1*Y[x-p,1])
		Y[x,1] = Y[x-p,1]*(3.5 - 3.5*Y[x-p,1] - b1*Y[x-p,0])
	e = np.average(abs(Y),axis=0)*noise_scale
	Y[:,0] += np.random.randn(n)*e[0]
	Y[:,1] += np.random.randn(n)*e[1]
	return X,Y

def two_freq(p,m,n,noise_scale=0.1,threshold=1.):
	offset = np.random.random()
	p = float(p)
	z = np.sin(np.arange(n) + offset)
	x = np.sin(np.arange(n) + offset)+np.sin(np.arange(n)/p + offset)*m
	y = np.sin(np.arange(n)+np.pi + offset)+np.sin(np.arange(n)/p + offset)*m
	e = np.average(abs(x))*noise_scale
	w = x + e*np.random.randn(len(x))
	x += e*np.random.randn(len(x))
	y += e*np.random.randn(len(x))
	z += e*np.random.randn(len(x))
	X = np.array([w,x,y,z])
	X[(X < threshold)] = 0
	X = np.exp(X)-1
	return X

def plot_random(Y,BloomSize,Period,labels=['w','x','y','z']):
	i0 = np.random.randint(Y.shape[1]/4)
	n = np.arange(Y.shape[0])
	for i in range(i0*4,i0*4+4):
		_=plt.plot(n,Y[:,i],label=labels[i%4])
	_=plt.title('Bloom Size: %.2f, Period %.2f' % (BloomSize[i],Period[i]))
	_=plt.legend()
	plt.savefig('simulations/random_timeseries.pdf')
	plt.close()

# generate many noisy, coupled series
series_length = 100
noise_scale = 1./2 #1/SNR
BloomSize = []
Period = []
Y = []
for bloomsize in np.linspace(2,100,10):
	m = np.log(bloomsize)
	for p in np.linspace(8,series_length/2,10):
		for _ in range(5):
			w,x,y,z = two_freq(p/np.pi,m,series_length,noise_scale=noise_scale)
			Y += [w,x,y,z]
			BloomSize += [bloomsize]*4
			Period += [p]*4

Y = np.array(Y).T
BloomSize = np.array(BloomSize)
Period = np.array(Period)
L,resolution = wavelet_levels(Y)
D = wavelet_level_correlations(L)
FDR = np.ones(D.shape,dtype=np.bool)
Dmerged = distance_combinatorics(D,FDR,resolution,Y.shape[1],1.,1.,mode=1,res_diff=1.,as_str=False)
Df1 = np.zeros((Y.shape[1],Y.shape[1]))
for k,v in Dmerged.items():
	for x,d in v:
		Df1[k,x] = d

Dmerged = distance_combinatorics(D,FDR,resolution,Y.shape[1],1.,1.,mode=2,res_diff=1.,as_str=False)
Df2 = np.zeros((Y.shape[1],Y.shape[1]))
for k,v in Dmerged.items():
	for x,d in v:
		Df2[k,x] = d

f = open('simulations/wavescore.pos_pos.snr2.txt','w')
f.write('\t'.join(['BloomSize','Period','+/+','+/-','+']) + '\n')
for i in range(0,Df2.shape[0],4):
	f.write('\t'.join([str(BloomSize[i]),str(Period[i])] + [str(x) for x in Df2[i,i+1:i+4]]) + '\n')

f.close()

f = open('simulations/wavescore.pos_neg.snr2.txt','w')
f.write('\t'.join(['BloomSize','Period','+/+','+/-','+']) + '\n')
for i in range(0,Df1.shape[0],4):
	f.write('\t'.join([str(BloomSize[i]),str(Period[i])] + [str(x) for x in Df1[i,i+1:i+4]]) + '\n')

f.close()

# only one bloom in each time series
BloomSize = []
Period = []
Y = []
for bloomsize in np.linspace(2,20,5):
	m = np.log(bloomsize)
	for p in np.linspace(10,series_length/2,5):
		for _ in range(25):
			bloom_loc = np.random.randint(0,series_length-int(p*1.2))
			w0,x0,y0,z0 = two_freq(p,0,bloom_loc,noise_scale=noise_scale)
			w1,x1,y1,z1 = two_freq(p,m,int(p*1.2),noise_scale=noise_scale)
			w2,x2,y2,z2 = two_freq(p,0,series_length-len(w0)-len(w1),noise_scale=noise_scale)
			Y += list(np.hstack([[w0,x0,y0,z0],[w1,x1,y1,z1],[w2,x2,y2,z2]]))
			BloomSize += [bloomsize]*4
			Period += [p]*4

Y = np.array(Y).T
BloomSize = np.array(BloomSize)
Period = np.array(Period)


# 1/SNR
ns = 1./1.
Dfinal = []
Dfinal2 = []
D2final = []
D2final2 = []
Dcorr = []
rd=1.
for _ in range(500):
	a,b = nonlinear_ab(100,noise_scale=ns)
	w1,x1,y1,z1 = two_freq(5,1,100,noise_scale=ns)
	w2,x2,y2,z2 = two_freq(15,1,100,noise_scale=ns)
	#Y = np.array([b[:,0],b[:,1],w1,x1,y1,z1,w2,x2,y2,z2]).T
	Y = np.array([w1,x1,y1,z1,w2,x2,y2,z2]).T
	L,resolution = wavelet_levels(Y)
	D = wavelet_level_correlations(L)
	FDR = np.ones(D.shape,dtype=np.bool)
	Dmerged = distance_combinatorics(D,FDR,resolution,Y.shape[1],1.,1.,mode=1,res_diff=rd,as_str=False)
	Dmerged2 = distance_combinatorics(D,FDR,resolution,Y.shape[1],1.3,0.5,mode=1,res_diff=rd,as_str=False)
	Df = np.zeros((Y.shape[1],Y.shape[1]))
	for k,v in Dmerged.items():
		for x,d in v:
			Df[k,x] = d
	Dfinal.append(Df)
	Df2 = np.zeros((Y.shape[1],Y.shape[1]))
	for k,v in Dmerged2.items():
		for x,d in v:
			Df2[k,x] = d
	Dfinal2.append(Df2)
	Dmerged = distance_combinatorics(D,FDR,resolution,Y.shape[1],1.,1.,mode=2,res_diff=rd,as_str=False)
	Dmerged2 = distance_combinatorics(D,FDR,resolution,Y.shape[1],1.2,0.6,mode=2,res_diff=rd,as_str=False)
	Df = np.zeros((Y.shape[1],Y.shape[1]))
	for k,v in Dmerged.items():
		for x,d in v:
			Df[k,x] = d
	D2final.append(Df)
	Df2 = np.zeros((Y.shape[1],Y.shape[1]))
	for k,v in Dmerged2.items():
		for x,d in v:
			Df2[k,x] = d
	D2final2.append(Df2)
	dc = 1-distance.squareform(distance.pdist(Y.T,'correlation'))
	dc[np.isnan(dc)] = 0
	Dcorr.append(dc)

plt.close()
#plt.subplot(3,1,1)
#plt.plot(a,b[:,0],label='coupled difference 1')
#plt.plot(a,b[:,1],label='coupled difference 2')
#plt.legend(loc='best',prop={'size': 7})
#plt.ylabel('signal')
plt.subplot(2,1,1)
plt.plot(a,w1,label='2-F A,1')
plt.plot(a,x1,label='2-F A,2')
plt.plot(a,y1,label='2-F A,3')
plt.plot(a,z1,label='2-F A,4')
#plt.legend(loc='best',prop={'size': 7})
#plt.ylabel('signal')
plt.subplot(2,1,2)
plt.plot(a,w2,label='2-F B,1')
plt.plot(a,x2,label='2-F B,2')
plt.plot(a,y2,label='2-F B,3')
plt.plot(a,z2,label='2-F B,4')
#plt.legend(loc='best',prop={'size': 7})
#plt.xlabel('time')
#plt.ylabel('signal')
#plt.suptitle('Simulated coupled dynamics with signal to noise ratio of %d' % (1/ns))
plt.savefig('time_series.SNR%d.png' % (1/ns),dpi=300)
plt.close()

Dfinal = np.array(Dfinal)
Dfinal2 = np.array(Dfinal2)
D2final = np.array(D2final)
D2final2 = np.array(D2final2)
Dcorr = np.array(Dcorr)
Dfinal = np.average(Dfinal,axis=0)
Dfinal2 = np.average(Dfinal2,axis=0)
D2final = np.average(D2final,axis=0) + np.eye(Y.shape[1])
D2final2 = np.average(D2final2,axis=0) + np.eye(Y.shape[1])
Dcorr = np.average(Dcorr,axis=0)

#labels = ['coupled difference 1','coupled difference 2','2-F A,1','2-F A,2','2-F A,3','2-F A,4','2-F B,1','2-F B,2','2-F B,3','2-F B,4']
labels = ['2-F A,1','2-F A,2','2-F A,3','2-F A,4','2-F B,1','2-F B,2','2-F B,3','2-F B,4']
plt.close()
fig,axx = plt.subplots()
ext = [0,Y.shape[1],0,Y.shape[1]]
im1 = axx.imshow(np.rot90(Dfinal),cmap=plt.cm.cubehelix,extent=ext,interpolation='none',vmin=0, vmax=1.0)
fig.colorbar(im1)
axx.set_yticks(np.arange(len(labels)) + 0.5)
axx.set_yticklabels(labels)
axx.set_xticks([])
axx.set_title('WaveClust similarity of simulated series (SNR %d)' % (1/ns))
plt.tight_layout()
plt.savefig('similarity_matrix.+-.SNR%d.png' % (1/ns),dpi=300)
plt.close()

plt.close()
fig,axx = plt.subplots()
ext = [0,Y.shape[1],0,Y.shape[1]]
im1 = axx.imshow(np.rot90(Dfinal2),cmap=plt.cm.cubehelix,extent=ext,interpolation='none',vmin=0, vmax=1.0)
fig.colorbar(im1)
axx.set_yticks(np.arange(len(labels)) + 0.5)
axx.set_yticklabels(labels)
axx.set_xticks([])
axx.set_title('WaveClust similarity of simulated series (SNR %d)' % (1/ns))
plt.tight_layout()
plt.savefig('similarity_matrix.+-.SNR%d.threshold.png' % (1/ns),dpi=300)
plt.close()

plt.close()
fig,axx = plt.subplots()
ext = [0,Y.shape[1],0,Y.shape[1]]
im1 = axx.imshow(np.rot90(D2final),cmap=plt.cm.cubehelix,extent=ext,interpolation='none',vmin=0, vmax=1.0)
fig.colorbar(im1)
axx.set_yticks(np.arange(len(labels)) + 0.5)
axx.set_yticklabels(labels)
axx.set_xticks([])
axx.set_title('WaveClust similarity of simulated series (SNR %d)' % (1/ns))
plt.tight_layout()
plt.savefig('similarity_matrix.++.SNR%d.png' % (1/ns),dpi=300)
plt.close()

plt.close()
fig,axx = plt.subplots()
ext = [0,Y.shape[1],0,Y.shape[1]]
im1 = axx.imshow(np.rot90(D2final2),cmap=plt.cm.cubehelix,extent=ext,interpolation='none',vmin=0, vmax=1.0)
fig.colorbar(im1)
axx.set_yticks(np.arange(len(labels)) + 0.5)
axx.set_yticklabels(labels)
axx.set_xticks([])
axx.set_title('WaveClust similarity of simulated series (SNR %d)' % (1/ns))
plt.tight_layout()
plt.savefig('similarity_matrix.++.SNR%d.threshold.png' % (1/ns),dpi=300)
plt.close()


plt.close()
fig,axx = plt.subplots()
ext = [0,Y.shape[1],0,Y.shape[1]]
im1 = axx.imshow(np.rot90(Dcorr),cmap=plt.cm.cubehelix,extent=ext,interpolation='none',vmin=0, vmax=1.0)
fig.colorbar(im1)
axx.set_yticks(np.arange(len(labels)) + 0.5)
axx.set_yticklabels(labels)
axx.set_xticks([])
axx.set_title('Correlation of simulated series (SNR %d)' % (1/ns))
plt.tight_layout()
plt.savefig('correlation_matrix.SNR%d.png' % (1/ns),dpi=300)
plt.close()
