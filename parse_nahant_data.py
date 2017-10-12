from collections import defaultdict
import numpy as np

#N1,D1,M1 = abundance_matrix('../data/myoutput.merged.unfrac.sorted.mat.lin.norm',min_appearances=10)
np.save('euks.names.npy',N1)
np.save('euks.dates.npy',D1)
np.save('euks.abundance.npy',M1)

N2,D2,M2 = abundance_matrix('../data/otu_table_unfrac_sorted_euk_chl_arch_cleaned.mat.lin.norm',min_appearances=10)
np.save('bact.names.npy',N2)
np.save('bact.dates.npy',D2)
np.save('bact.abundance.npy',M2)

N,D,M = merge_data(N1,N2,D1,D2,M1,M2)
np.save('all_unfractionated.names.npy',N)
np.save('all_unfractionated.dates.npy',D)
np.save('all_unfractionated.abundance.npy',M)


# data that has not gone through distribution-based clustering
N1,D1,M1 = abundance_matrix('clusters100-98/nahant_18S_seqs_trimmed_120.sorted.tax.unfrac.fracs.mean.count_table',min_appearances=10)
np.save('clusters100-98/euks.names.npy',N1)
np.save('clusters100-98/euks.dates.npy',D1)
np.save('clusters100-98/euks.abundance.npy',M1)

N2,D2,M2 = abundance_matrix('clusters100-98/nahant_16S_seqs_fw.sorted.tax.unfrac.euk.chloro.arch.clean.fracs.mean.count_table',min_appearances=10)
np.save('clusters100-98/bact.names.npy',N2)
np.save('clusters100-98/bact.dates.npy',D2)
np.save('clusters100-98/bact.abundance.npy',M2)

N,D,M = merge_data(N1,N2,D1,D2,M1,M2)
np.save('clusters100-98/all_unfractionated.names.npy',N)
np.save('clusters100-98/all_unfractionated.dates.npy',D)
np.save('clusters100-98/all_unfractionated.abundance.npy',M)


def abundance_matrix(fp,min_appearances=10,has_names=True):
	f = open(fp)
	if False:
		Cols = f.readline().strip().split('\t')[1:-1]
		# used by Antonio: 10N.204.37      10N.204.38      10N.204.39      10N.205.37...
		Dates = [Cols[0][:Cols[0].rfind('.')]]
		Di = [0]
		for c in Cols[1:]:
			if Dates[-1]+'.' in c:
				Di.append(Di[-1])
			else:
				Dates.append(c[:c.rfind('.')])
				Di.append(Di[-1]+1)
	else:
		Cols = f.readline().strip().split('\t')[1:-1]
		# other format: 204	205	206	207	208...
		# used for Diana's data (I think)
		Dates = Cols
		Di = range(len(Dates))
	Names = []
	M = [[] for _ in Dates]
	for j,line in enumerate(f):
		ls = line.strip().split('\t')
		if has_names:
			Names.append(ls[-1])
		else:
			Names.append(ls[0])
		for i in range(len(Di)):
			if len(M[Di[i]]) <= j:
				M[Di[i]].append(0)
			if ls[i+1] == '':
				M[Di[i]][j] += 0
			else:
				M[Di[i]][j] += float(ls[i+1])
	M = np.array(M).T
	M = np.transpose(M/[Di.count(i) for i in range(len(Dates))])
	mx = np.zeros(M.shape[0],dtype=np.bool)
	ms = M.sum(1)
	for i in range(M.shape[0]):
		if ms[i] > 0:
			mx[i] = True
	M = M[mx,:]
	print M.shape
	Dates = [Dates[i] for i in range(len(mx)) if mx[i]]
	appearances = np.zeros(M.shape[1],dtype=np.bool)
	for j in range(M.shape[1]):
		if np.count_nonzero(M[:,j]) > min_appearances:
			appearances[j] = True
	M = M[:,appearances]
	Names = [(i,Names[i]) for i in range(len(appearances)) if appearances[i]]
	return Names,Dates,M

def merge_data(N1,N2,D1,D2,M1,M2):
	d_intersect = sorted(set(D1) & set(D2))
	d1idx = [i for i,d in enumerate(D1) if d in d_intersect]
	d2idx = [i for i,d in enumerate(D2) if d in d_intersect]
	return N1+N2,d_intersect,np.vstack([M1[d1idx].T,M2[d2idx].T])

def parse_meta_old(fp):
	Lines = open(fp).readlines()
	Dates = [int(x) for x in Lines[0].split()[1:]]
	Labels = []
	X = np.empty((len(Lines)-1,len(Dates)))
	X[:] = np.nan
	for i in range(1,len(Lines)):
		ls = Lines[i].split()
		Labels.append(ls[0])
		for j in range(1,len(ls)):
			try:
				if ls[j].count('.') > 1:
					X[i-1,j-1] = float(ls[j][:ls[j].rfind('.')])
				else:
					X[i-1,j-1] = float(ls[j])
			except:
				pass
	return Dates,Labels,X

def parse_meta(meta_path):
	FP = glob.glob(os.path.join(meta_path,'*.txt'))
	Data = defaultdict(dict)
	Labels = []
	for fp in FP:
		f = open(fp)
		line = f.readline().strip().split('\t')
		label = line[1]
		Labels.append(label)
		for line in f:
			ls = line.strip().split('\t')
			try:
				Data[int(ls[0])][label] = float(ls[1])
			except:
				pass
		f.close()
	Dates = sorted(Data.keys())
	X = np.empty((len(Labels),len(Dates)))
	X[:] = np.nan
	for i,d in enumerate(Dates):
		for l,v in Data[d].items():
			X[Labels.index(l),i] = v
	return Dates,Labels,X
			
