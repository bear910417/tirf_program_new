
import numpy as np
from sklearn import mixture 
import itertools
import matplotlib.pyplot as plt
import matplotlib
from math import sqrt
import os

matplotlib.use('agg')


class GMM:
    
    def __init__(self, path, data,select, channel = 'g'):
        
        self.path = path
        self.data = data
        self.select = select
        self.channel = channel
        
        if self.select ==1:
            try:
                self.selected = np.load(self.path+f'\\selected_{channel}.npy')
                print(f"Total Traces: {(self.selected == 1).sum():.0f} / {self.selected.shape[0]}")
            except:
                self.selected = np.ones(data.shape[0])
            
            try:
                self.bkps = np.load(self.path+r'\\breakpoints_g.npy',allow_pickle=True)
            except:
                self.bkps = [[-1] for _ in range(data.shape[0])]
                    
        else:
            self.selected = np.ones(data.shape[0])
            self.bkps = [[-1] for _ in range(data.shape[0])]
        
    def gaussian(self, x,A,mu1,sigma1):
        y=A * (1/(sigma1*sqrt(2*np.pi)))*np.exp(-1.0 * (x - mu1)**2 / (2 * sigma1**2))
        return y
    
    def fit(self, text, fsc):

            path = self.path
            text = text

                
            np.random.seed(10)
            X0=[]
            for i,trace in enumerate(self.data):
                #print(trace)
                if self.selected[i]!=-1:
                    if len(self.bkps[i])>1:
                        end = self.bkps[i][1]
                    else:
                        end = trace.shape[0]
                    for t in range(0, end):
                        if (-0.2 <= trace[t] <= 1.2):
                            X0.append(trace[t])
                  
            X = np.array(X0).reshape(-1,1)


            
            lowest_bic = np.infty
            bic = []
            n_components_range = range(1, 9)
            cv_types = ["full"]
            means=[]
            cov=[]
            weights=[]
            for cv_type in cv_types:
                for n_components in n_components_range:
                    try:
                        fsc.set("fret_progress", str( n_components / n_components_range))
                    except:
                        pass
                    # Fit a Gaussian mixture with EM
                    gmm = mixture.GaussianMixture(
                        n_components=n_components, covariance_type=cv_type, max_iter=300, n_init=1
                    )
                    gmm.fit(X)
                    bic.append(gmm.bic(X))
                    if bic[-1] < lowest_bic:
                        lowest_bic = bic[-1]
                        best_gmm = gmm
                    means.append(gmm.means_)
                    cov.append(gmm.covariances_)
                    weights.append(gmm.weights_)

                    
                        
            #print(means)
            
            bic = np.array(bic)
            color_iter = itertools.cycle(["navy", "turquoise", "cornflowerblue", "darkorange"])
            clf = best_gmm
            bars = []
            
            # Plot the BIC scores
            px = 1/72
            plt.rcParams["figure.figsize"] = (480*px,270*px)
            spl = plt.subplot(2, 1, 1)
            for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
                xpos = np.array(n_components_range) + 0.2 * (i - 2)
                bars.append(
                    plt.bar(
                        xpos,
                        bic[i * len(n_components_range) : (i + 1) * len(n_components_range)],
                        width=0.6,
                        color=color,
                    )
                )
            plt.xticks(n_components_range)
            plt.ylim([bic.min() * 1.01 - 0.01 * bic.max(), bic.max()])
            plt.title("BIC score per model")
            xpos = (
                np.mod(bic.argmin(), len(n_components_range))
                + 0.65
                + 0.2 * np.floor(bic.argmin() / len(n_components_range))
            )
            plt.text(xpos, bic.min() * 0.97 + 0.03 * bic.max(), "*", fontsize=14)
            spl.set_xlabel("Number of components")
            spl.legend([b[0] for b in bars], cv_types)
            os.makedirs(path+f'\\{self.channel}', exist_ok = True)
            plt.savefig(f'{path}\\{self.channel}\\BIC.tif', dpi=100)
            #plt.show()
            
            font = {'family': 'Arial',
                    'size': 12,
                    }
            
            
            plt.rc('font', **font)
            plt.rc('xtick', labelsize=12) 
            plt.rc('ytick', labelsize=12) 

            ignore=[]
            highlight=[]
            #ignore=[0.38]
            for n_state in n_components_range:
                tot=0
                m=means[n_state-1].flatten()
                c=np.sqrt(cov[n_state-1].flatten())
                w=weights[n_state-1]
                plt.clf()
                
                yconv=np.zeros((100000))
                data_entries, bins = np.histogram(X, bins=np.arange(-0.2,1,0.01), density=True)
                binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)]) 
                plt.bar(binscenters, data_entries, width=bins[1] - bins[0], color='wheat', label=r'Histogram entries')
                xspace = np.linspace(-0.2, 1, 100000)
                for i in range(0,n_state):
                    if np.round(m[i],2) not in ignore:
                      tot = tot+np.round(w[i],2)
                
                for i in range(0,n_state):
                    yconv=yconv+self.gaussian(xspace,w[i],m[i],c[i])
                    if np.round(m[i],2) not in ignore:
                        if np.round(m[i],2)  in highlight:
                            plt.plot(xspace, self.gaussian(xspace,w[i],m[i],c[i]), color='orangered', linewidth=3.0, label='g'+str(i+1))    
                        else:
                            plt.plot(xspace, self.gaussian(xspace,w[i],m[i],c[i]), color='y', linewidth=1.0, label='g'+str(i+1),linestyle='dashed')    
                        if text ==True:
                            plt.text(m[i]-0.02,5,str(np.round(m[i],2)),multialignment ='center', fontdict=font)
                            plt.text(m[i]-0.02,3,str(np.round(w[i]/tot,2)),multialignment ='center',color='orange', fontdict=font)
                plt.plot(xspace, yconv, color='orange', linewidth=2.5, label=r'Fitted function')
                plt.xticks(np.arange(0,1,0.1)) 
                plt.yticks(np.arange(0,10.1,1)) 
                plt.xlim(0, 1)
                plt.ylim(0,10)
                plt.xlabel('FRET Efficency', fontdict=font)
                plt.ylabel('Probability Density', fontdict=font)
                #plt.legend(loc='best')
                plt.tight_layout()
                
                print(f'{n_state} states:')
                print(f'means: {np.around(m, 2)}')
                print(f'weight: {np.around(w, 2)}')
                print(f'std: {np.around(c, 2)}')


                plt.savefig(path+f'\\{self.channel}\\{n_state}.tif', dpi=300)
                plt.savefig(path+f'\\{self.channel}\\{n_state}.jpg', dpi=300)
                #plt.show()
            return means, cov, weights
