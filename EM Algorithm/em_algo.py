import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
import pickle

def plot_gauss(w, m, s, cols, ax):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]
    
    ax.scatter(m[0], m[1], s=50, c=cols)
    
    nstd = 2
    vals, vecs = eigsorted(s)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    width, height = 2 * nstd * np.sqrt(vals)
    e = Ellipse(xy=m, width=width, height=height, angle=theta, 
                edgecolor=cols, facecolor='none', alpha=np.sqrt(w), linewidth=8*np.sqrt(w))
    ax.add_artist(e)

def comp_em(w, mu, sigma, X):
    def expectation(w, mu, sigma, X):
        gamma_nk = np.zeros(shape=(N, K))
        
        for k in range(K):
            dist = multivariate_normal(mu[k], sigma[k])
            gamma_nk[:, k] = dist.pdf(X)
        
        global fit
        fit = np.sum(gamma_nk)
        
        gamma_nk = (w*gamma_nk) / ((w*gamma_nk).sum(axis=1)[..., None])
        return gamma_nk
    
    def maximization(X, gamma_nk):
        w = 1/N * gamma_nk.sum(axis=0)
        mu = (gamma_nk[..., None]*X[:, None, :]).sum(axis=0)/(gamma_nk).sum(axis=0)[:, None]
        out = X[:, None, :] - mu
        out = (out[:, :, :, None] * out[:, :, None, :])
        sigma = (gamma_nk[...,None, None]*out).sum(axis=0)/(gamma_nk).sum(axis=0)[:, None, None]
        
        return w, mu, sigma
    
    gamma_nk = expectation(w, mu, sigma, X)
    w_new, mu_new, sigma_new = maximization(X, gamma_nk)
            
    pal = np.array([0.85,0.85,0.85])
    colors = gamma_nk*pal[None, ...]
        
    return w_new, mu_new, sigma_new, colors

iris = sns.load_dataset("iris")
X = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y = iris['species'].values

K=3
N=X.shape[0]
D=X.shape[1]

fi = 0.3
cols = np.array([(1,fi,fi), (fi,1,fi), (fi,fi,1)])[::-1]/1.5
tol = 0.001
tries = 5

best_fit = 0
for i in range(tries):
    prev = 1
    cur = 100

    w = np.array([1/K for k in range(K)])
    mu = iris.groupby('species').mean().values + np.random.uniform(-1.5, 1.5, size=(K, D))
    sigma = np.array([np.eye(D) for k in range(K)])
    
    try:
        _,_,_, colors = comp_em(w, mu, sigma, X)
        seq = [[w, mu, sigma, colors]]
        while abs(prev-cur) > tol:
            w, mu, sigma, colors = comp_em(w, mu, sigma, X)
            seq.append([w, mu, sigma, colors])
            prev = cur
            cur = fit
        
        if cur > best_fit:
            print("try {}, fit {}".format(i, cur))
            best_fit = cur
            best_seq = seq
            
    except Exception as e:
        print(e)
        continue

with open("fun.pkl", 'wb') as f:
    pickle.dump(best_seq, f)
    
sns.set(font_scale=1)

for fr in range(len(best_seq)):
    best_w, best_mu, best_sigma, best_colors = best_seq[fr]

    grid = sns.pairplot(iris, hue='species')
    grid.fig.set_size_inches(10,8)
    
    for k in range(K):
        for pi in range(0, D):
            for pj in range(pi+1, D):
                cur_mu = np.array([best_mu[k][pi], best_mu[k][pj]])
                cur_sigma = np.array([[best_sigma[k][pi, pi], best_sigma[k][pi, pj]], 
                                      [best_sigma[k][pj, pi], best_sigma[k][pj, pj]]])
                plot_gauss(best_w[k], cur_mu, cur_sigma, cols[k], grid.axes[pj, pi])
                plot_gauss(best_w[k], cur_mu[::-1], cur_sigma[::-1, ::-1], cols[k], grid.axes[pi, pj])
        
    plt.savefig("./save/output.png".format(fr), dpi=600)
    print(fr)
