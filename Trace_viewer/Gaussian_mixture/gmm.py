import numpy as np
from math import sqrt
import matplotlib
from Gaussian_mixture.GMM_custom import GMM
import matplotlib.pyplot as plt

def gaussian(x, A, mu1, sigma1):
    """
    Standard Gaussian function.
    """
    return A * (1/(sigma1 * sqrt(2 * np.pi))) * np.exp(- (x - mu1)**2 / (2 * sigma1**2))

def fit_gmm(FRET_list, select_list, init_means, covariance_type, n_comps):
    """
    Fit a Gaussian Mixture Model to the provided FRET data.
    """
    gmm = GMM(data=FRET_list, selected=select_list)
    means, covs, weights, X, n_components = gmm.fit(smooth=10, init=init_means,
                                                     covariance_type=covariance_type, n_components=n_comps)
    return means, covs, weights, X, gmm

def draw_gmm(fig, m, c, w, X):
    """
    Draw the GMM curves on the given figure.
    """
    yconvs = np.zeros((100000))
    xspace = np.linspace(0, 1, 100000)
    fig.update_traces(x=X.reshape(-1), selector=dict(name='gmm_hist'))
    fig.data = [list(fig.data)[0]]
    fig.layout.annotations = []
    if len(c) != len(m):
        c = np.ones(len(m)) * c
    for i in range(m.shape[0]):
        yconvs += gaussian(xspace, w[i], m[i], c[i])
        fig.add_scatter(x=xspace, y=gaussian(xspace, w[i], m[i], c[i]), name=f'ysep{i}',
                        marker=dict(color='orange'), line=dict(dash='dash'))
        fig.add_annotation(x=m[i], y=int(np.max(gaussian(xspace, w[i], m[i], c[i]))/2),
                           text=f'{np.round(w[i], 2)*100:.0f}%', showarrow=False, yshift=10)
    fig.add_scatter(x=xspace, y=yconvs, marker=dict(color='orange'))
    return fig

def save_gmm(gmm, path):
    """
    Save the GMM plot using matplotlib.
    """
    matplotlib.use('agg')
    gmm.plot_and_save(save_path=path)
