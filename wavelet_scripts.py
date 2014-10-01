import pywt
from operator import itemgetter
from scipy.spatial import distance
from hcluster import pdist, linkage, dendrogram, fcluster
import pylab
import numpy as np

def partition(lst, n):
	division = len(lst) / float(n)
	return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in xrange(n) ]

def wavelet_avg(Y,X=None,reshape=False,plot_xy=True,zero_thresh=1*10**-7,Names=None,Title=None):
	if not X:
		X = range(Y.shape[0])
	w = pywt.Wavelet('sym2')
	L = pywt.dwt_max_level(Y.shape[0],w)
	#L = pywt.swt_max_level(Y.shape[0])
	Zavg = zeros((L,len(X)))
	Zavgabs = zeros((L,len(X)))
	if reshape:
		e = [0,L,0,L]
	else:
		e = [X[0],X[-1],0,L]
	sp = 1
	if not Names:
		Names = ["Series "+str(y+1) for y in range(Y.shape[1])]
	if Title:
		pylab.suptitle(Title)
	if plot_xy:
		tp = str(Y.shape[1]+3)
		pylab.subplot(int(''.join([tp,'1',str(sp)])))
		P = pylab.plot(X,Y)
		pylab.legend(P,Names,prop={'size':6})
		pylab.xlim([0,Y.shape[0]-1])
		pylab.title("Relative Abundance Time Series")
		sp += 1
	else:
		tp = str(Y.shape[1]+2)
	for y in range(Y.shape[1]):
		C = pywt.wavedec(Y[:,y],w,level=L)
		#C = pywt.swt(Y[:,y],w,level=L)
		#C = ['dummy'] + [c[1] for c in C]
		Z = wavelet_matrix(C,L,len(X))
		pylab.subplot(int(''.join([tp,'1',str(sp)])))
		pylab.imshow(abs(Z),extent=e)
		pylab.title(Names[y])
		sp += 1
		Zavg += Z
		Zavgabs += abs(Z)
	Zavg /= Y.shape[1]
	Zavg[Zavg<zero_thresh] = 0
	pylab.subplot(int(''.join([tp,'1',str(sp)])))
	pylab.imshow(abs(Zavg),extent=e)
	pylab.title("Decomposition Avg")
	sp += 1
	Zavgabs /= Y.shape[1]
	Zavgabs[Zavgabs<zero_thresh] = 0
	pylab.subplot(int(''.join([tp,'1',str(sp)])))
	pylab.imshow(abs(Zavgabs),extent=e)
	pylab.title("abs(Decomposition) Sum")
	sp += 1
	pylab.show()
	return Zavg

def merge_matrices(Mtuples):
	d = set(Mtuples[0][1])
	for mt in Mtuples[1:]:
		d = (d & set(mt[1]))
	Y = np.empty((len(d),sum([mt[0].shape[1] for mt in Mtuples])))
	j = 0
	N = []
	for mt in Mtuples:
		mx = [i for i,x in enumerate(mt[1]) if x in d]
		N += mt[2]
		Y[:,j:j+mt[0].shape[1]] = mt[0][mx]
		j += mt[0].shape[1]
	return Y,N

def wavelet_levels(Y):
	w = pywt.Wavelet('sym2')
	levels = pywt.dwt_max_level(Y.shape[0],w)
	w0 = pywt.wavedec(Y[:,0],w,level=levels)
	L = [np.empty((Y.shape[1],len(x))) for x in w0[1:]]
	for i in range(Y.shape[1]):
		wd = pywt.wavedec(Y[:,i],w)
		for j,l in enumerate(L):
			l[i,:] = wd[j+1]
	return L

def decomposition_example(p=20.,m=5):
	x = np.sin(np.arange(100))+np.sin(np.arange(100)/p)*m
	y = np.sin(np.arange(100)+np.pi)+np.sin(np.arange(100)/p)*m
	z = y*-1
	L = wavelet_levels(np.transpose([x,y,z]))
	pylab.subplot(611)
	pylab.plot(x)
	pylab.plot(y)
	pylab.plot(z)
	pylab.subplot(612)
	pylab.plot(L[0][0])
	pylab.plot(L[0][1])
	pylab.plot(L[0][2])
	pylab.subplot(613)
	pylab.plot(L[1][0])
	pylab.plot(L[1][1])
	pylab.plot(L[1][2])
	pylab.subplot(614)
	pylab.plot(L[2][0])
	pylab.plot(L[2][1])
	pylab.plot(L[2][2])
	pylab.subplot(615)
	pylab.plot(L[3][0])
	pylab.plot(L[3][1])
	pylab.plot(L[3][2])
	pylab.subplot(616)
	pylab.plot(L[4][0])
	pylab.plot(L[4][1])
	pylab.plot(L[4][2])
	pylab.show()

def wavelet_level_correlations(Y):
	L = wavelet_levels(Y)
	D = [distance.squareform(distance.pdist(l,'cosine')) for l in L]
	return D

def wavelet_level_clusters(Y,c=0.3):
	D = wavelet_level_correlations(Y)
	D = .5*(D[0] + 2 - D[-1])
	for i in range(D.shape[0]):
		D[i,i] = 0
	link = hier.linkage(distance.squareform(D),method='single',metric='cosine')
	clust = hier.fcluster(link,c,criterion='distance')
	return cluster_sets(clust)

def plot_cluster(c,Y,N,T=None):
	if T:
		pylab.subplot(2,1,1)
	c = random.sample(c,min(5,len(c)))
	for x in c:
		pylab.plot(Y[:,x])
	pylab.legend([N[i] for i in c],prop={'size':8})
	if T:
		pylab.subplot(2,1,2)
		pylab.plot(T)
	pylab.show()

def correlation_diff(d1,d2,c=0.2):
	diff = (abs(d1-d2) > 2-2*c)
	pairs = []
	for i in range(d1.shape[0]):
		for j in range(i+1,d1.shape[0]):
			if diff[i,j]:
				x1 = 1 - abs(1 - d1[i,j])
				x2 = 1 - abs(1 - d2[i,j])
				if (x1 < c) and (x2 < c):
					pairs.append((.5*(x1+x2),i,j))
	pairs.sort()
	return pairs

def name_hist(c,N):
	K = defaultdict(int)
	P = defaultdict(int)
	C = defaultdict(int)
	O = defaultdict(int)
	F = defaultdict(int)
	G = defaultdict(int)
	for x in c:
		for n in N[x].split('|')[2].split(';'):
			if 'k__' in n:
				K[n] += 1
			elif 'p__' in n:
				P[n] += 1
			elif 'c__' in n:
				C[n] += 1
			elif 'o__' in n:
				O[n] += 1
			elif 'f__' in n:
				F[n] += 1
			elif 'g__' in n:
				G[n] += 1
	print sorted(K.items(),key=itemgetter(1),reverse=True)[:5]
	print sorted(P.items(),key=itemgetter(1),reverse=True)[:5]
	print sorted(C.items(),key=itemgetter(1),reverse=True)[:5]
	print sorted(O.items(),key=itemgetter(1),reverse=True)[:5]
	print sorted(F.items(),key=itemgetter(1),reverse=True)[:5]
	print sorted(G.items(),key=itemgetter(1),reverse=True)[:5]

def wavelet_matrix(C,r,c):
	Z = zeros((r,c))
	C1 = C[1:]
	for i in range(r):
		yp = partition(range(c),len(C1[i]))
		for yj in range(len(yp)):
			for j in yp[yj]:
				Z[i,j] += C1[i][yj]
	return Z

def nonlinear_ab(n,b1=0.02,b2=0.1,p=1):
	X = np.linspace(1,n,n)
	Y = np.zeros((len(X),2))
	Y[:p,0] = np.random.rand(1,p)
	Y[:p,1] = Y[0,0]
	for x in range(p,len(X)):
		Y[x,0] = Y[x-p,0]*(3.8 - 3.8*Y[x-p,0] - b1*Y[x-p,1])
		Y[x,1] = Y[x-p,1]*(3.5 - 3.5*Y[x-p,1] - b1*Y[x-p,0])
	return X,Y

def doppler(n,m,bmin=0,bmax=1,reflections=1):
	x = linspace(0,1,n/(reflections+1))
	y = sqrt(x*(1-x))*sin((2.1*pi)/(x+.05))
	y += -y.min()
	y /= y.max()
	if reflections%2 == 1:
		y = array((list(y) + list(y)[::-1])*((reflections+1)/2))
	elif reflections%2 == 0:
		y = list(y) + (list(y)[::-1] + list(y))*(reflections/2)
		y += list(y[:n-len(y)])
		y = array(y)
	ym = zeros((n,m))
	ym[:,0] = y
	for j in range(1,m):
		b = random.rand()*(bmax - bmin) + bmin
		ym[:,j] = ym[:,0]*b + random.rand(1,n)*(1-b)
	return range(n),ym

def rand_abundances(y0,n):
	Y = zeros((len(y0),n+1))
	Y[:,0] = y0
	for i in range(Y.shape[0]):
		r = random.rand(1,n)
		Y[i,1:] = r*(1-Y[i,0])/r.sum()
	return Y

def normalize_to_abundances(Y,amin=0,amax=1):
	for i in range(Y.shape[0]):
		Y[i,:] = Y[i,:]/Y[i,:].sum()
	return Y

def c_dists(Y,use_swt=True,level_weights=False):
	w = pywt.Wavelet('sym2')
	if use_swt:
		L = pywt.swt_max_level(Y.shape[0])
		C = [pywt.swt(Y[:,i],w,level=L) for i in range(Y.shape[1])]
		C = [[list(reshape(l[0],-1)) + list(reshape(l[1],-1)) for l in c] for c in C]
	else:
		L = pywt.dwt_max_level(Y.shape[0],w)
		C = [pywt.wavedec(Y[:,i],w,level=L) for i in range(Y.shape[1])]
	if level_weights:
		if use_swt:
			raise NameError('No level weights with SWT')
		Wc = [1. for x in range(1,L+1)]
		D = zeros((len(C),len(C)))
		for i in range(len(C)):
			for j in range(i+1,len(C)):
				d = sum([distance.cosine(C[i][x],C[j][x])*Wc[x] for x in range(L)])/sum(Wc)
				D[i,j] = d
				D[j,i] = d
		return D
	else:
		Cn = []
		for c in C:
			cn = []
			for l in c:
				cn += list(l)
			Cn.append(cn)
		return abs(pdist(Cn,'cosine'))

def wavelet_clusters(Y,ct=0.5,weights=False,return_clusters=False,swt=False):
	if weights:
		D = abs(c_dists(Y,level_weights=True,use_swt=False))
		Dr = []
		for i in range(D.shape[0]-1):
			Dr += list(D[i,i+1:])
	else:
		Dr = c_dists(Y,use_swt=swt)
	if return_clusters:
		L = linkage(Dr,method='single',metric='cosine')
		C = fcluster(L,ct,criterion='distance')
		return cluster_sets(C)
	plot_clusters(Dr,ct)

def time_series_clusters(Y,ct=0.5,return_clusters=False):
	D = pdist(transpose(Y),'correlation')
	D = abs(D)
	if return_clusters:
		L = linkage(D,method='single',metric='cosine')
		C = fcluster(L,ct,criterion='distance')
		return cluster_sets(C)
	plot_clusters(D,ct)

def plot_clusters(Dr,ct):
	L = linkage(Dr,method='single',metric='cosine')
	dendrogram(L,color_threshold=ct)
	pylab.show()

def create_population(types,n,rand_periods=1):
	Y = []
	for t in types:
		if t[0] == 'nonlinear':
			x,y = nonlinear_ab(n,b1=random.rand()*.2,b2=random.rand()*.75,p=random.randint(1,1+rand_periods))
		elif t[0] == 'doppler':
			x,y = doppler(n,t[1],reflections=random.randint(0,rand_periods))
		elif t[0] == 'rand':
			y = random.rand(n,t[1])
		if len(Y) > 0:
			Y = concatenate((Y,y),axis=1)
		else:
			Y = y
	return normalize_to_abundances(Y*random.power(.5,(1,Y.shape[1]))[:,newaxis][0])

def check_clusters(result_sets,answer_sets):
	s = 0
	matched_sets = []
	for rs in result_sets:
		for ans in answer_sets:
			if len(rs & ans)/float(len(rs)) > .7:
				s += len(rs)
				matched_sets.append(rs)
				break
	return s,matched_sets

def cluster_sets(cluster_array):
	cluster_array = sorted(enumerate(cluster_array),key=itemgetter(1))
	rs = [[cluster_array[0][0]]]
	for i in range(1,len(cluster_array)):
		if cluster_array[i][1] != cluster_array[i-1][1]:
			rs.append([])
		rs[-1].append(cluster_array[i][0])
	return [set(l) for l in rs if len(l)>1]

def iter_rand_clusters(types,m,n=10,r=3,swt=True):
	A = {'nonlinear': [],'doppler': []}
	i = 0
	for t in types:
		if t[0] != 'rand':
			A[t[0]] += range(i,i+t[1])
			i += t[1]
	A0 = [set(A['nonlinear'] + A['doppler'])]
	A = [set(v) for v in A.values() if len(v)>1]
	ta = sum([len(s) for s in A])
	print ta
	tsum = 0
	wsum = 0
	if swt:
		wct = 0.1
	else:
		wct = 0.08
	for i in range(n):
		Y = create_population(types,m,rand_periods=r)
		t = time_series_clusters(Y,ct=0.5,return_clusters=True)
		t0 = check_clusters(t,A)
		t = wavelet_clusters(Y,ct=wct,return_clusters=True,swt=swt)
		t1 = check_clusters(t,A0)
		if (t0[0] < ta) and (t1[0] < ta):
			print t0[0],t1[0]
			tsum += t0[0]
			wsum += t1[0]
	print tsum,wsum

def cluster_diff(C1,C2):
	d = []
	for i in range(len(C1)):
		if max([len(C1[i] & c2) for c2 in C2]) < len(C1[i])*.85:
			d.append((i,C1[i]))
	return d

def plot_time_series(C,Y,N):
    P = pylab.plot(Y[:,list(C)[:5]])
    pylab.legend(P,[N[x] for x in list(C)[:5]],prop={'size':7})
    pylab.show()

def similarity_counts(D,Names,j0,t=0.1):
	S = defaultdict(int)
	X = (D < t)
	for i in range(j0):
		n1 = Names[i][1].split(';')
		for j in range(j0,D.shape[1]):
			if X[i,j]:
				n2 = Names[j][1].split(';')
				for n in n1:
					S[n] += 1
				for n in n2:
					S[n] += 1
	return sorted(S.items(),key=itemgetter(1),reverse=True)

# Analyzing combinatoric distances
def distance_analysis_sizeAbund(dist_file,data_file,name_file):
	X = np.load(data_file)
	X = np.average(X,0)
	Xpct = [np.percentile(X,i) for i in range(10,110,10)]
	N = np.load(name_file)
	N = [n.split('|')[1] for n in N]
	DistAbund = defaultdict(list)
	DistSize = defaultdict(list)
	f = open(dist_file)
	header = [f.readline() for _ in range(6)]
	for line in f:
		ls = line.strip().split()
		if len(ls) > 1:
			i = int(ls[0].split(':')[0])
			xi = np.where(X[i] <= Xpct)[0][0]
			for y in ls[1:-1]:
				j,dist = y.split(':')
				j = int(j)
				if i < j:
					break
				xj = np.where(X[j] <= Xpct)[0][0]
				DistAbund[abs(xi - xj)].append(float(dist))
				sizebin = tuple(sorted([N[i],N[j]]))
				DistSize[sizebin].append(float(dist))
	f.close()
	return DistAbund,DistSize

strints = [str(x) for x in range(10)]
def parse_mcl_clusters(fp):
	C = [[]]
	f = open(fp)
	for line in f:
		if line[0] in strints:
			C[-1] += [int(x) for x in line.replace('$','').strip().split()[1:]]
		elif line[0] == ' ':
			C[-1] += [int(x) for x in line.replace('$','').strip().split()]
		if '$' in line:
			C.append([])
	f.close()
	return C

def distance_analysis_clusters(dist_file,clusters,minsize=100,inflation=5):
	C = {}
	for i,c in enumerate(clusters):
		if len(c) > minsize:
			for x in c:
				C[x] = i
	DistCluster = defaultdict(int)
	f = open(dist_file)
	header = [f.readline() for _ in range(6)]
	for line in f:
		ls = line.strip().split()
		if len(ls) > 1:
			i = int(ls[0].split(':')[0])
			if i in C:
				ci = C[i]
				J = []
				X = []
				for y in ls[1:-1]:
					j,dist = y.split(':')
					J.append(int(j))
					X.append(float(dist))
				X = np.array(X)**inflation
				X /= X.sum()
				for j,x in zip(J,X):
					if j in C:
						DistCluster[(ci,C[j])] += x
	return DistCluster

C = parse_mcl_clusters('combinatorics/out.merged.mci.I28')
DC = distance_analysis_clusters('combinatorics/merged.mci',C)
np.save('combinatorics/distance.intercluster.npy',DC.items())

DA,DS = distance_analysis_sizeAbund('combinatorics/merged.mci','all_fractionated_data.npy','all_fractionated_names.npy')
C = []
B = []
h = np.histogram(DA[0],100)
b0 = np.array(list(h[1]) + [h[1][-1]+h[1][-1]-h[1][-2]])
for k,v in DA.items():
	c,b = np.histogram(v,bins=b0,density=True)
	C.append(c)
	B.append(b)

np.save('combinatorics/distance.abundance.counts.npy',C)
np.save('combinatorics/distance.abundance.bins.npy',B)
np.save('combinatorics/distance.abundance.keys.npy',DA.keys())

C = []
B = []
h = np.histogram(DS.values()[0],100)
b0 = np.array(list(h[1]) + [h[1][-1]+h[1][-1]-h[1][-2]])
for k,v in DS.items():
	c,b = np.histogram(v,bins=b0,density=True)
	C.append(c)
	B.append(b)

np.save('combinatorics/distance.size_fraction.counts.npy',C)
np.save('combinatorics/distance.size_fraction.bins.npy',B)
np.save('combinatorics/distance.size_fraction.keys.npy',DS.keys())

def best_pairs(dist_file):
	f = open(dist_file)
	header = [f.readline() for _ in range(6)]
	Similarity = []
	for line in f:
		ls = line.strip().split()
		if len(ls) > 1:
			i = int(ls[0].split(':')[0])
			for y in ls[1:-1]:
				j,dist = y.split(':')
				j = int(j)
				if i < j:
					break
				Similarity.append((float(dist),i,j))
	return sorted(Similarity,key=itemgetter(0),reverse=True)

S = best_pairs('combinatorics/merged.mci')
np.save('combinatorics/distance.best_pairs.npy',S)


				