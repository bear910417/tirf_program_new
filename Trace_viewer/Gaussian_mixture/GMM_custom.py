import numpy as np
from  sklearn.mixture import GaussianMixture as sGMM
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d as uf
from math import sqrt
import matplotlib as mpl

font = {'family': 'Arial',
        'size': 5,
        }

mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['xtick.major.width'] = 0.5
mpl.rcParams['ytick.major.width'] = 0.5
mpl.rc('font', **font)
mpl.rc('xtick', labelsize=5) 
mpl.rc('ytick', labelsize=5) 
px = 1/72
mpl.rcParams["figure.figsize"] = (180*px,120*px)


class GMM():
    def __init__(self, path = None, data = None, selected = None):
        self.path = path
        self.data = data
        self.selected = selected
        self.means = None
        self.cov = None
        self.weights = None
        self.X = None

    def gaussian(self, x, A, mu1, sigma1):
        y=A * (1/(sigma1*sqrt(2*np.pi)))*np.exp(-1.0 * (x - mu1)**2 / (2 * sigma1**2))
        return y
    
    def load_data(self, channel = 'fret_g'):
        self.data = np.load(self.path+r'\data.npz')[channel]
        try:
            self.selected = np.load(self.path+r'\selected_g.npy')
        except:
            self.selected = np.ones(self.data.shape[0])
    
    def fit(self, smooth = 10, init = [], covariance_type = 'diag', n_components = 1, seed = 10, max_iter = 300, n_init = 3, init_params = 'k-means++', **kwargs):
        if not np.any(self.data):
            self.load_data()
        
        if np.any(init):
            n_components = len(init)
        
        X0 = []
        for i, trace in enumerate(self.data):
            if self.selected[i]!=-1:
                X0.append(uf(trace, smooth))


        X = np.array(X0).reshape(-1)
        X = X[X<=1.2]
        X = X[X>=-0.2]
        X = X.reshape(-1, 1)


        np.random.seed(seed)

        if np.any(init):
            gmm = sGMM(n_components = n_components, covariance_type = covariance_type, max_iter =  max_iter, n_init = n_init, means_init = np.array(init).reshape(-1,1), init_params = init_params, **kwargs)
        else:
            gmm = sGMM(n_components = n_components, covariance_type = covariance_type, max_iter =  max_iter, n_init = n_init, init_params = init_params, **kwargs)
        
        gmm.fit(X)




        means = gmm.means_
        cov = gmm.covariances_
        weights = gmm.weights_

        print(means)
        print(cov)
        print(covariance_type)

        
        self.means = means.flatten()
        self.cov = np.sqrt(cov.flatten())
        self.weights = weights
        self.X = X
        self.n_components = n_components

        return self.means, self.cov, self.weights, self.X, self.n_components

    def plot_and_save(self, text = False, ignore = [], highlight = [], ylim = 10, custom_name = None, save_path = None):

        yconv = np.zeros((100000))
        xspace = np.linspace(0, 1, 100000)
        tot = 0

        m = self.means 
        c = self.cov 
        w = self.weights

        if len(c) != len(m):
            c = np.ones(len(m)) * c 

        plt.hist(self.X, bins=np.arange(0,1,0.01), density=True, color='wheat')

        for i in range(0, self.n_components):
            if np.round(m[i], 2) not in np.round(ignore, 2):
                tot = tot + np.round(w[i], 2)

        for i in range(0, self.n_components):
            yconv = yconv + self.gaussian(xspace, w[i],m[i],c[i])
            if np.round(m[i],2) not in ignore:
                if np.round(m[i],2) in highlight:
                    plt.plot(xspace, self.gaussian(xspace,w[i],m[i],c[i]), color='orangered', linewidth=1.0, label='g'+str(i+1))    
                else:
                    plt.plot(xspace, self.gaussian(xspace,w[i],m[i],c[i]), color='y', linewidth=0.5, label='g'+str(i+1),linestyle='dashed')    
                if text ==True:
                    plt.text(m[i]-0.02,5,str(np.round(m[i],2)),multialignment ='center', fontdict=font)
                    plt.text(m[i]-0.02,3,str(np.round(w[i]/tot,2)),multialignment ='center',color='orange', fontdict=font)
            else:
                print(f'Ignored {np.round(m[i],2)} state.')

        plt.plot(xspace, yconv, color='orange', linewidth = 1.5, label=r'Fitted function')
        plt.xticks(np.arange(0.0, 1.1,0.1)) 
        plt.yticks(np.arange(0,10.1,1)) 
        plt.xlim(0.0, 1)
        plt.ylim(0,ylim)
        plt.xlabel('FRET Efficency', fontdict=font)
        plt.ylabel('Probability Density', fontdict=font)
        plt.tight_layout()

        print(f'{self.n_components} states:')
        print(f'means: {np.around(m, 2)}')
        print(f'weight: {np.around(w, 2)}')
        print(f'std: {np.around(c, 2)}')

        if save_path == None:
            save_path = self.path

        if custom_name == None:
            plt.savefig(save_path + r'\\custom_'+ str(self.n_components)+'.eps', format = 'eps', dpi=1200)
            plt.savefig(save_path + r'\\custom_'+ str(self.n_components)+'.tif', format = 'tif', dpi=1200)
        else:
            plt.savefig(save_path + f'\\{custom_name}.eps', format = 'eps', dpi=1200)
            plt.savefig(save_path + f'\\{custom_name}.tif', format = 'tif', dpi=1200)


        #plt.show()


