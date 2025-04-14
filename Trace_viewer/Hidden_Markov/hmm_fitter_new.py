from hmmlearn.hmm import GaussianHMM
import numpy as np
from sklearn import preprocessing
import os
import matplotlib.pyplot as plt 
from tqdm import tqdm
import time
import pickle
import matplotlib as mpl
from scipy.ndimage import uniform_filter1d as uf

class HMM_fitter:
    
    def __init__(self, path):
        self.path=path
        self.N_traces=0
        self.hd_states = 0
        self.selected = []   
        self.filt_Q = []
        self.means=[]

    def load_traces(self):
        Q = np.load(self.path+r'\\data.npz')['fret_g']
        self.N_traces = Q.shape[0]
        self.selected = np.load(self.path+r'\\selected_g.npy').astype(int)
        self.bkps = np.load(self.path+r'\\breakpoints.npz', allow_pickle = True)['fret_g']
        self.time_g = np.load(self.path+r'\\data.npz')['time_g']
        self.Q = Q
        return Q
    
    
    def process_Q(self, w = 10):
       
        Q = uf(self.Q, w, mode = 'reflect', axis = 1)
        Q = np.clip(Q, 0.01, 0.99)
        #Q = np.log(Q)
        N_traces = Q.shape[0]
        pro_Q = np.zeros(0)
        length = []

        for i in range (0, N_traces):
            bkp = self.bkps[i]
            if self.selected[i] == 1:
                if len(bkp) > 0:
                    frag = Q[i][:bkp[0][0]]
                    pro_Q = np.concatenate([pro_Q, frag])
                    length.append(bkp[0][0])

                else:
                    pro_Q = np.concatenate([pro_Q, Q[i]])
                    length.append(Q.shape[1])
        pro_Q = pro_Q.reshape(-1, 1)
        self.pro_Q = pro_Q
        self.length = length

        return N_traces, length
            
          
    def fitHMM(self, r, w = 10, means = None, fix_means = False, epoch = 10, covariance_type = 'spherical', n_iter = 20):
        # fit Gaussian HMM to Q
        N_traces, length = self.process_Q(w)
        if not np.any(means):
            means = [0.5]

        self.means = means
        means = np.array(means)

        k = means.shape[0]
        #means = np.log(means)
        if fix_means:
            params = 'stc'
        else:
            params = 'stmc'
        
        if r:
            print('fitting')
            tic = time.perf_counter()
            models = []
            conv = []

            for e in tqdm(range(epoch)):
                print(f'\n Epoch {e}')
                model = GaussianHMM(n_components = k, n_iter = n_iter, verbose = True, min_covar = 0, startprob_prior = np.ones(k) / k, means_prior = means, covars_prior = 0.00001, covariance_type = covariance_type, transmat_prior = 0.01, init_params='stc', params = params, implementation = 'scaling')  
                model.means_ = means
                model.fit(self.pro_Q, length)
                models.append(model)
                conv.append(model.monitor_.history[-1])
            best = np.argmax(conv)
            print(f'Best likelihood {conv[best]}')
            model = models[best]
            toc = time.perf_counter()
            print(f"Finished in {toc - tic:0.4f} seconds")
            with open(self.path+r"\model.pkl", "wb") as file: pickle.dump(model, file)

        else:
            model = GaussianHMM(n_components = k, n_iter = 10, verbose = True, min_covar= 100, covariance_type = covariance_type, covars_prior = 0.001, transmat_prior = 0, init_params='stc', params = params)  
            with open(self.path+r"\model.pkl", "rb") as file: model=pickle.load(file)
            #with open(r'H:\TIRF\20221229\lane3\dmc1\320s\FRET\0'+r"\model.pkl", "rb") as file: model=pickle.load(file)
            #with open(r'H:\TIRF\20221229\lane2\dmc1\530s\FRET\0'+r"\model.pkl", "rb") as file: model=pickle.load(file)
        mus = np.array(model.means_)
        self.mus = mus

        #sigmas = np.array(np.sqrt(np.array([np.diag(model.covars_[0]),np.diag(model.covars_[1]),np.diag(model.covars_[2]),np.diag(model.covars_[3])])))
        sigmas=np.array(model.covars_)
        P = np.array(model.transmat_)
        print(sigmas)
        print(mus)
        print(P)
         
        # classify each observation as state 0 or 1
        print('predicting')
        tic = time.perf_counter()
        hidden_states = model.predict(self.pro_Q, length)
        likelihood = model.predict(self.pro_Q, length)
        aic = model.aic(self.pro_Q, length)
        bic = model.bic(self.pro_Q, length)
        
        toc = time.perf_counter()
        print(f"Finished in {toc - tic:0.4f} seconds")
        print(f"Log Likelihood =  {np.sum(likelihood):.4f}")
        print(f"averaged Log Likelihood =  {np.average(likelihood):.4f}")
        print(f"AIC =  {aic:.4f}")
        print(f"aBIC =  {bic:.4f}")
        
        
        self.hidden_states = hidden_states
        return model, hidden_states
    
    def cal_states(self, plot, p_length = None, text = False, mode = 'tif'):
        plt.switch_backend('agg')
        font = {'family': 'Arial',
                'size': 5,
                }
        plt.rc('font', **font)
        plt.rc('xtick', labelsize=5) 
        plt.rc('ytick', labelsize=5) 
        plt.rcParams["figure.figsize"] = (180/72, 120/72)
        mpl.rcParams['axes.linewidth'] = 0.5
        mpl.rcParams['xtick.major.width'] = 0.5
        mpl.rcParams['ytick.major.width'] = 0.5

        N_traces = self.Q.shape[0]
        length = self.length
        start = 0
        time =  uf(self.time_g, 10, mode = 'reflect', axis = 0)
        hd_states = []
        time_arr = []
        print('plotting')
        path = os.path.join(self.path,'HMM_traces')
        os.makedirs(path, exist_ok = True)

        j = 0
        

        for i in tqdm(np.arange(0, N_traces)):
            if self.selected[i] == 1:
                end = start + length[j]
                mus_frag = self.hidden_states[start : end]
                time_frag = time[:length[j]]
                time_arr.append(time_frag)
                trace_frag = self.pro_Q[start : end]
                hd_states_frag = self.mus[mus_frag]
                hd_states.append(hd_states_frag)

                if plot: 
                    plt.plot(time_frag, trace_frag)
                    plt.plot(time_frag, hd_states_frag, linewidth = 0.8)
                    if p_length == None:
                        p_length = np.max(time_frag)
                    for state, mean in enumerate(np.flip(np.sort(self.mus, axis = 0))):
                        plt.hlines(mean, 0, np.max(time_frag), colors='skyblue', linestyles='dashed')
                        if text:
                            #plt.text(p_length + 2, mean, f'{np.round(mean[0], 2):.2f}', color = 'red')
                            plt.text(p_length + 2, mean, f'{state}', color = 'red')
                    plt.ylim(0,1)
                    plt.xlim(0, p_length)

                   
                    plt.xlabel('time (s)')
                    plt.ylabel('FRET')     
                    plt.tight_layout()
                    plt.savefig(path+f'\\{i}.{mode}', dpi=300, format = mode)
                    plt.close()
                start = end
                j = j + 1
            else:
                hd_states.append([])
        print(len(hd_states))
        print(len(self.length))


        np.savez(path + r'\\hmm.npz', hd_states = np.array(hd_states, dtype = object), time = np.array(time_arr, dtype = object), means = self.mus)
            