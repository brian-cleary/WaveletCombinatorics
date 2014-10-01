from collections import defaultdict
import numpy as np

def abundance_time_series(fp,min_appearances=10):
    f = open(fp)
    L = f.readlines()
    Rows = L[1].strip().split('\t')[1:-1]
    Names = []
    M = np.zeros((len(Rows),len(L)-2))
    for j in range(2,len(L)):
        try:
            ls = L[j].strip().split('\t')
            Names.append(ls[-1])
            for i in range(len(Rows)):
               M[i,j-2] = float(ls[i+1])
        except:
            print i,j
    mx = np.zeros(M.shape[0],dtype=np.bool)
    ms = M.sum(1)
    for i in range(M.shape[0]):
        if ms[i] > 0:
            mx[i] = True
    M = M[mx,:]
    print M.shape
    Rows = [Rows[i] for i in range(len(mx)) if mx[i]]
    appearances = np.zeros(M.shape[1],dtype=np.bool)
    for j in range(M.shape[1]):
        if np.count_nonzero(M[:,j]) > min_appearances:
            appearances[j] = True
    M = M[:,appearances]
    Names = [(i,Names[i]) for i in range(len(appearances)) if appearances[i]]
    Replicates = {}
    Dates = defaultdict(list)
    for k in set([r[r.rfind('.')+1:] for r in Rows]):
        mx = np.zeros(M.shape[0],dtype=np.bool)
        for i in range(M.shape[0]):
            if Rows[i][-len(k):] == k:
                mx[i] = True
                Dates[k].append(int(Rows[i].split('.')[1]))
        Replicates[k] = M[mx,:]
    return Names,Dates,Replicates

def abundance_matrix(fp,min_appearances=20):
	f = open(fp)
	Cols = f.readline().strip().split('\t')[1:-1]
	Names = []
	Dates = [Cols[0][:Cols[0].rfind('.')]]
	Di = [0]
	for c in Cols[1:]:
		if Dates[-1]+'.' in c:
			Di.append(Di[-1])
		else:
			Dates.append(c[:c.rfind('.')])
			Di.append(Di[-1]+1)
	M = [[] for _ in Dates]
	for j,line in enumerate(f):
		ls = line.strip().split('\t')
		Names.append(ls[-1])
		for i in range(len(Di)):
			if len(M[Di[i]]) <= j:
				M[Di[i]].append(0)
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
	

def parse_meta(fp):
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

def smooth_nan(X):
	Y = np.zeros(X.shape)
	for x in X:
		if np.isnan(x[0]):
			
