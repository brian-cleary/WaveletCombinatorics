from statsmodels.tsa import ar_model,arima_model,stattools
from statsmodels.regression import linear_model
import numpy as np
import pylab
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from time import time

# should include dates when it gets fixed in statsmodels
def garch(y,q0=1,p=1,q=1):
	model = ar_model.AR(y)
	results = model.fit(q0)
	et = results.resid**2
	model = arima_model.ARMA((et - sum(et)/len(et))/np.std(et),(p,q))
	return model.fit()

def garch_group(Y,q0=1,p=1,q=1,do_plots=False):
    residuals = np.zeros(Y.shape[0]-q0)
    for y in np.transpose(Y):
        model = ar_model.AR(y)
        results = model.fit(q0)
        et = results.resid**2
        residuals += (et - sum(et)/len(et))/np.std(et)
	residuals /= Y.shape[1]
    model = arima_model.ARMA(residuals,(p,q))
    r2 = model.fit()
    if do_plots:
        print r2.pvalues
        pylab.plot(r2.fittedvalues)
        pylab.show()
    else:
        return residuals,r2

def cluster_vs_meta_granger(c,X,M,Ml,lags=7,thresh=0.05):
	x1 = np.average(X[:,c],1)
	R = []
	for x2 in M.T:
		have_values = np.isfinite(x2)
		result = stattools.grangercausalitytests(np.transpose([x1[have_values],x2[have_values]]),lags)
		R.append([(result[i+1][0]['ssr_ftest'][1],i+1) for i in range(lags) if result[i+1][0]['ssr_ftest'][1] < thresh])
	RM = []
	for i,r in enumerate(R):
		if r:
			avgLag = np.average([x[1] for x in r])
			avgPvalue = np.average([x[0] for x in r])
			RM.append((avgPvalue,Ml[i],avgLag))
	return sorted(RM)

def cluster_vs_meta_granger_TM(c,X,M,Ml,lags=7,thresh=0.05):
	# use the Toda Yamamoto method (environmental data is stationary, but clusters are not)
	x1 = X[c].sum(0)
	adf = stattools.adfuller(x1,maxlag=lags)
	if (adf[0] > adf[4]['5%']):
		m1 = adf[2]
	else:
		m1 = 0
	R = []
	for j,x2 in enumerate(M):
		have_values = np.isfinite(x2)
		xi = x1[have_values]
		x2i = x2[have_values]
		adf = stattools.adfuller(x2i,maxlag=lags)
		if (adf[0] > adf[4]['5%']):
			m2 = adf[2]
		else:
			m2 = 0
		m = max(m1,m2)
		y = [xi[i+max(0,m2-m1):len(xi)+i-(m1+lags)] for i in range(m1+lags)] + [x2i[i+max(0,m1-m2):len(xi)+i-(m2+lags)] for i in range(m2+lags)]
		y = np.array(y).T
		lm = linear_model.OLS(xi[max(m1,m2)+lags:],y)
		result = lm.fit()
		Restr = np.eye(y.shape[1])[m+lags:]
		wald = result.wald_test(Restr)
		if wald.pvalue < thresh:
			R.append((wald.pvalue,Ml[j]))
	return m,sorted(R)
		

def cluster_vs_cluster_granger(C,X,lags=4,thresh=0.01):
	Xc = [np.average(X[:,c],1) for c in C]
	R = []
	for i in range(len(C)):
		x1 = Xc[i]
		for j in range(i+1,len(C)):
			x2 = Xc[j]
			result = stattools.grangercausalitytests(np.transpose([x1,x2]),lags)
			for l in range(lags):
				pv = result[l+1][0]['ssr_ftest'][1]
				if pv < thresh:
					R.append((pv,(i,j,l+1)))
			result = stattools.grangercausalitytests(np.transpose([x2,x1]),lags)
			for l in range(lags):
				pv = result[l+1][0]['ssr_ftest'][1]
				if pv < thresh:
					R.append((pv,(j,i,l+1)))
	return sorted(R)

def cluster_vs_meta_correlation(c,X,M,Ml,thresh=0.1):
	x = np.average(X[:,c],1)
	dist = []
	for i,m in enumerate(M.T):
		have_values = np.isfinite(m)
		d = distance.correlation(x[have_values],m[have_values])
		if d < thresh:
			dist.append((1-d,Ml[i]))
	return sorted(dist,reverse=True)

def plot_cluster_meta(c,X,m):
	x = np.average(X[:,c],1)
	pylab.subplot(211)
	pylab.plot(x)
	pylab.subplot(212)
	pylab.plot(m)
	pylab.show()

def variance_map(Y):
	garch_cols = []
	garch_models = []
	for i in range(Y.shape[1]):
		try:
			r = garch(Y[:,i])
			if (r.pvalues[1] < 0.005) or (r.pvalues[2] < 0.005):
				garch_cols.append(i)
				garch_models.append(r)
		except:
			pass
	# note dumb: should explicitly set lag here
	Ygarch = Y[1:,garch_cols]
	filter_means = [0.000001]
	filter_stds = [0.0000005]
	while filter_means[-1] < 0.25:
		filter_means.append(filter_stds[-1]*5)
		filter_stds.append(filter_means[-1]/2)
	Weights = np.array([np.exp(-(Ygarch-filter_means[i])**2/filter_stds[i]**2) for i in range(len(filter_means))])
	Map = Weights*np.transpose([r.fittedvalues for r in garch_models])
	return filter_means,Map.sum(2)/Weights.sum(2)

def plot_map(X,Y,M,Title,threed=True):
	Mx,My = np.meshgrid(X,np.log(Y))
	fig = pylab.figure()
	if threed:
		ax = fig.gca(projection='3d')
		surf = ax.plot_surface(Mx,My,M,rstride=1,cstride=1,cmap=cm.jet,linewidth=0,antialiased=False)
		ax.set_zlabel('GARCH Variance (normalized)')
	else:
		ax = fig.add_subplot(111)
		ax.pcolor(Mx,My,M,cmap=cm.jet)
	ax.set_yticks([np.log(y) for y in Y[::3]])
	ax.set_yticklabels([str(y*100)[:6]+'%' for y in Y[::3]])
	ax.set_xticks([x for x in X[::15]])
	ax.set_xticklabels([int(x) for x in X[::15]])
	ax.set_xlabel('Time')
	ax.set_ylabel('Abundance')
	pylab.suptitle(Title)
	pylab.title('Average GARCH Models')
	pylab.show()

def combine(Y,M,Dy,Dm,defaultzero=False):
	Z = np.empty((Y.shape[0]+M.shape[0],M.shape[1]))
	if defaultzero:
		Z[:] = 0
	else:
		Z[:] = np.nan
	for i in range(Y.shape[0]):
		for j in enumerate(Y[i]):
			try:
				Z[i,Dm.index(Dy[j[0]])] = j[1]
			except:
				pass
	Z[i+1:,:] = M
	return Z

def garch_meta_cov(G,M,Dg,Dm):
	Z = np.empty((G.shape[0]+M.shape[0],M.shape[1]))
	Z[:] = np.nan
	for i in range(G.shape[0]):
		for j in enumerate(G[i]):
			Z[i,Dm.index(Dg[j[0]])] = j[1]
	Z[i+1:,:] = M
	maskedarr = np.ma.array(Z,mask=np.isnan(Z))
	Corr = np.ma.corrcoef(maskedarr.T,rowvar=False,allow_masked=True)
	return Corr.data[:G.shape[0],G.shape[0]:]

def js_div(A,B):
    half=(A+B)/2
    return 0.5*kl_div(A,half)+0.5*kl_div(B,half)

def kl_div(A,B):
    return sum(np.multiply(A,np.log(A/B)))
	