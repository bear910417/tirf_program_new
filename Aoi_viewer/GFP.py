import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib as mpl
import os
from scipy.ndimage import uniform_filter1d as uf


class GFP:
    def __init__(self, path):
        self.path = path
        try:
            self.selected = np.load(self.path+r'\\selected_g.npy')
        except:
            self.selected = np.ones(500)


    def plot(self, lag, snap_time_g):
        
        file_path = self.path + r'\\data.npz'
        
        y_max = 10000
        plt.close()
        avg_b = uf(np.load(file_path)['bb'], size = lag, mode = 'nearest')[:, snap_time_g[0]:snap_time_g[1]]
        fret_g = uf(np.load(file_path)['fret_g'] , size = lag, mode = 'nearest')[:, snap_time_g[0]:snap_time_g[1]]


        
        os.makedirs(self.path +r'\\gfp_scatter', exist_ok = True)
        
                
        font = {'family': 'Arial',
                'size': 5,
                }

        mpl.rcParams['axes.linewidth'] = 0.5
        mpl.rcParams['xtick.major.width'] = 0.5
        mpl.rcParams['ytick.major.width'] = 0.5
        plt.rc('font', **font)
        plt.rc('xtick', labelsize=5) 
        plt.rc('ytick', labelsize=5) 
        px = 1/72
        plt.rcParams["figure.figsize"] = (100*px,105*px)
        fret_selected = []
        avg_b_selected=[]
        for i in range (fret_g.shape[0]):
            if self.selected[i] ==1:
                fret_selected.append(fret_g[i])
                avg_b_selected.append(avg_b[i])
        plt.scatter(fret_selected, avg_b_selected, marker = '.', s = 0.5, edgecolors='None')
        plt.hlines(400,0,1,colors='skyblue', linestyles='dashed')
        plt.xlim(0,1)
        plt.ylim(-1000,20000)
        plt.xlabel('FRET')
        plt.ylabel('GFP Intensity')
        plt.tight_layout()
        plt.savefig(self.path +r'\\gfp_scatter\\all.eps', dpi = 1200, format = 'eps')
        #plt.show()
        plt.close()
        
        heatmap, xedges, yedges = np.histogram2d(np.array(fret_selected).reshape(-1),  np.array(avg_b_selected).reshape(-1), bins = 50, range =  [[-0.2, 1], [-1000, y_max]], density = True)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        fig, ax = plt.subplots(figsize=(120/72,100/72), ncols=1)
        plt.xticks(np.arange(-0.20, 1.1, 0.2))
        plt.yticks(np.arange(-1000, y_max+1, 1000))
        plt.xlabel('FRET')
        plt.ylabel('GFP Intensity')
        pos = plt.imshow(heatmap.T,  extent = extent, aspect = 'auto', origin='lower', cmap='Greys')
        pos.set_clim(0, 0.005)
        cbar = fig.colorbar(pos, ax=ax, format = "%5.3f", ticks = [0, 0.001, 0.002, 0.003, 0.004, 0.005])
        plt.hlines(2000,0,1,colors='skyblue', linestyles='dashed')
        plt.tight_layout()
        plt.savefig(self.path +r'\\gfp_scatter\\all_heat.tif', dpi = 1200, format = 'tif')
        plt.savefig(self.path +r'\\gfp_scatter\\all_heat.eps', dpi = 1200, format = 'eps')
        #plt.show()