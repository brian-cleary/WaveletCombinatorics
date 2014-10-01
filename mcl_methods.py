# mcl commands:
# create level 0 network from positive correlations
#	$ mcxarray -data wavelet.0.tab -co 0.6 -skipc 1 -o wavelet.0.60.mci -write-tab wavelet.0.dict
# find appropriate correlation level
#	$ mcx query -imx wavelet.0.60.mci --vary-correlation
# alter the network to remove correlations below 0.9
#	$ mcx alter -imx wavelet.0.60.mci -tf 'gq(0.9), add(-0.9)' -o wavelet.0.90.mci
#
# create level 3 network from anticorrelations
#	$ mcxarray -data wavelet.3.tab -co 0.2 -skipc 1 -tf 'mul(-1)' -o wavelet.3.20.mci
# find appropriate correlation level
#	$ mcx query -imx wavelet.3.20.mci --vary-correlation
# alter the network to remove correlations below 0.52
#	$ mcx alter -imx wavelet.3.20.mci -tf 'gq(0.52)' -o wavelet.3.52.mci
#
# find the interacting pairs from level 3
#	>>> X0 = get_interactions(path+'wavelet.3.52.mci')
# merge level 3 interactions with level 0 network
#	>>> merge_interactions(path+'wavelet.0.90.mci',path+'wavelet.merged.mci',X0)
#
# cluster the merged network
#	$ mcl wavelet.merged.mci -I 1.2
#	$ mcl wavelet.merged.mci -I 1.4
# write labels on cluster nodes
#	$ mcxdump -icl out.wavelet.merged.mci.I12 -o dump.wavelet.merged.mci.I12 -tabr wavelet.0.dict
#	$ mcxdump -icl out.wavelet.merged.mci.I14 -o dump.wavelet.merged.mci.I14 -tabr wavelet.0.dict

strints = [str(x) for x in range(10)]

def get_interactions(fp):
	f = open(fp)
	C = {}
	for line in f:
		ls = line.replace('$','').strip().split()
		if line[0] in strints:
			ci = ls[0].split(':')[0]
			C[ci] = [int(x.split(':')[0]) for x in ls[1:]]
		elif line[0] == ' ':
			C[ci] += [int(x.split(':')[0]) for x in ls]
	return C

def merge_interactions(infile,outfile,interactions):
	f = open(infile)
	g = open(outfile,'w')
	for _ in range(6):
		g.write(f.readline())
	interact_id = None
	interact_list = []
	for line in f:
		ls = line.replace('$','').strip().split()
		if line[0] in strints:
			if interact_list:
				g.write(interact_id)
				g.write(' '.join(interact_list)+' $\n')
				interact_list = []
			interact_id = line[:line.index(ls[1])]
			for x in ls[1:]:
				x_id = int(x.split(':')[0])
				if x_id in interactions.get(interact_id.split(':')[0],[]):
					interact_list.append(x)
		elif line[0] == ' ':
			for x in ls:
				x_id = int(x.split(':')[0])
				if x_id in interactions.get(interact_id.split(':')[0],[]):
					interact_list.append(x)
	if interact_list:
		g.write(interact_id)
		g.write(' '.join(interact_list)+' $\n')
	f.close()
	g.write(')\n')
	g.close()

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

def cluster_sects(C0,C1,s=0.6):
	C = []
	for c0 in C0:
		s0 = set(c0)
		for c1 in C1:
			s01 = set(s0 & set(c1))
			if len(s01) > s*min(len(c0),len(c1)):
				C.append(list(s01))
	return C