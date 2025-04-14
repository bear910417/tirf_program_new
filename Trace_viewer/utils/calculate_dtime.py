import numpy as np
from sklearn import mixture 
from math import sqrt

def calculate_dtime(tot_bkps,dead_time,exposure_time):
    dtime=[]
    for traces in tot_bkps:
        if len(traces)>1:
            bkp=traces[1]  
            dtime.append((bkp)*exposure_time+dead_time-0.01)
    return dtime
        
        
def calculate_dtime2(tot_bkps,dead_time,exposure_time):
    dtime=[]
    for traces in tot_bkps:
        if len(traces)>2:
            dtime.append((traces[2]-traces[1])*exposure_time)
    return dtime
         
def calculate_conv(FRET_list):
      
    np.random.seed(10)

    X = np.array(FRET_list).reshape(-1,1)
    n_components_range = range(1, 9)
    
    means=[]
    cov=[]
    weights=[]
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(
            n_components=n_components, covariance_type='full', max_iter=300, n_init=1
        )
        gmm.fit(X)
        means.append(gmm.means_)
        cov.append(gmm.covariances_)
        weights.append(gmm.weights_)
        
   
    yseps=[]
    yconvs=[]
    for n_state in n_components_range: 
        ysep = []
        m=means[n_state-1].flatten()
        c=np.sqrt(cov[n_state-1].flatten())
        w=weights[n_state-1]
        xspace = np.linspace(0, 1, 1000)
        yconv = np.zeros(1000) 
        for i in range (n_state):
            yconv += gaussian(xspace,w[i],m[i],c[i])   
            ysep.append(gaussian(xspace,w[i],m[i],c[i]))
        yseps.append(ysep)
        yconvs.append(yconv)
        
    return means, cov, weights, yconvs, yseps


def calculate_FRET(FRET, selected_list, tot_bkps):
    FRET_list=[]
    for i in range(len(selected_list)):
        if selected_list[i] !=-1:
            if len(tot_bkps[i])>1:
                end = tot_bkps[i][1]
            else:
                end = FRET[i].shape[0]
            
            for t in range (0,end):
                if (0<=FRET[i][t]<=1):
                    FRET_list.append(FRET[i][t])  
                
    return FRET_list
    
def gaussian(x,A,mu1,sigma1):
    y=A * (1/(sigma1*sqrt(2*np.pi)))*np.exp(-1.0 * (x - mu1)**2 / (2 * sigma1**2))
    return y

        
     
   
