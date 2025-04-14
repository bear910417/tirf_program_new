from scipy.ndimage import uniform_filter1d as uf
import numpy as np
import os
import matplotlib.pyplot as plt


class Processor:
    def __init__(self, n, proc_config):
        
        self.n = n
        self.leakage_g = proc_config['leakage_g']
        self.leakage_b = proc_config['leakage_b']
        self.gamma_g = 1
        self.gamma_b = 1
        self.direct_bg = 0.13
        self.path = proc_config['path']
        self.lag = 1
        self.ti = proc_config['ti']
        # self.snap_time_g = proc_config['snap_time_g']
        # self.snap_time_b = proc_config['snap_time_b']
        # self.snap_time_r = proc_config['snap_time_r']
        self.red = proc_config['red']
        self.red_intensity = proc_config['red_intensity']
        self.red_time = proc_config['red_time']
        self.green = proc_config['green']
        self.green_intensity = proc_config['green_intensity']
        self.green_time = proc_config['green_time']
        self.ps = proc_config['preserve_selected']


    def load_data_npz(self):

        path=self.path
        lag=self.lag
        
        path = path+r'\\raw\\hel0.npz'
        self.N_traces = np.load(path)['cnt']
        

        #load time stamp, load traces
        try:
            raw_gg = np.load(path)['trace_gg']
            raw_gr = np.load(path)['trace_gr']
            time_g = np.load(path)['time_g']

        except:
            raw_gg = np.zeros(self.N_traces)
            raw_gr = np.zeros(self.N_traces)
            time_g = np.zeros(self.N_traces)
        

        try:
            raw_rr = np.load(path)['trace_rr']
            time_r = np.load(path)['time_r']
        
        except:
            raw_rr = np.zeros(self.N_traces)
            time_r = np.zeros(self.N_traces)
       
        try:
            raw_bb = np.load(path)['trace_bb']
            raw_bg = np.load(path)['trace_bg']
            raw_br = np.load(path)['trace_br']
            time_b = np.load(path)['time_b']
        
        except:
            raw_bb = np.zeros(self.N_traces)
            raw_bg = np.zeros(self.N_traces)
            raw_br = np.zeros(self.N_traces)
            time_b = np.zeros(self.N_traces)

        if np.any(raw_gg):
            avg_gg = np.zeros_like(raw_gg)
            avg_gr = np.zeros_like(raw_gr)
        else:
            avg_gg = np.zeros((self.N_traces, 10))
            avg_gr = np.zeros((self.N_traces, 10))
   

        if np.any(raw_rr):
            avg_rr = np.zeros_like(raw_rr)
        else:
            avg_rr = np.zeros((self.N_traces, 10))

        if np.any(raw_bb):
            avg_bb = np.zeros_like(raw_bb)
            avg_bg = np.zeros_like(raw_bg)
            avg_br = np.zeros_like(raw_br)
        else:
            avg_bb = np.zeros((self.N_traces, 10))
            avg_bg = np.zeros((self.N_traces, 10))
            avg_br = np.zeros((self.N_traces, 10))

        avg_time_b = uf(time_b, size = lag, mode = 'nearest')
        avg_time_g = uf(time_g, size = lag, mode = 'nearest')
        avg_time_r = uf(time_r, size = lag, mode = 'nearest')



        for trace in range(int(self.N_traces)):
            if np.any(raw_gg):
                avg_gg[trace]= raw_gg[trace]
                avg_gr[trace]= raw_gr[trace]

            if np.any(raw_rr):
                avg_rr[trace]= raw_rr[trace]
               
            if np.any(raw_bb):
                avg_bb[trace]= raw_bb[trace]
                avg_bg[trace]= raw_bg[trace]
                avg_br[trace]= raw_br[trace]
           


        self.avg_gg = avg_gg
        self.avg_gr = avg_gr
        self.avg_rr = avg_rr
        self.avg_bb = avg_bb
        self.avg_bg = avg_bg
        self.avg_br = avg_br
        self.avg_time_g = avg_time_g
        self.avg_time_r = avg_time_r
        self.avg_time_b = avg_time_b


            
        return None
        
        

    def plot_intensity(self, data, title):
        plt.hist(data.reshape(-1), bins = np.arange(-2000, 40000, 2000), density = True, color= 'purple')
        plt.ylabel('Probability Density')
        plt.xlabel('Intensity')
        plt.title(title)
        plt.tight_layout()
        path = os.path.join(self.path,'total_intensity')
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path+f'\\{title}.tif')
        plt.close()

    def process_data(self):

        self.load_data_npz()  

        avg_time_g = self.avg_time_g
        avg_time_b = self.avg_time_b
        avg_time_r = self.avg_time_r

        avg_gg = self.avg_gg
        avg_gr = self.avg_gr - self.leakage_g * self.avg_gg
        avg_bb = self.avg_bb
        if self.avg_gg.shape == self.avg_bg.shape:
            print(f'using direct bg_leakage = {self.direct_bg}')
            avg_bg = self.avg_bg - self.leakage_b * self.avg_bb - self.direct_bg * self.avg_gg 
        else:
            avg_bg = self.avg_bg - self.leakage_b * self.avg_bb
        
        if self.avg_gg.shape == self.avg_bg.shape:
            avg_br = self.avg_br - self.leakage_g * self.avg_bg - self.direct_bg * self.avg_gr 
        else:
            avg_br = self.avg_br - self.leakage_g * self.avg_bg
        avg_rr = self.avg_rr

        
        try:
            if self.ps == 1:
                self.selected_g = np.load(self.path+f'\\FRET\\{self.n}\\selected_g.npy')
            else:
                self.selected_g = np.ones(self.N_traces)
        except:
            self.selected_g = np.ones(self.N_traces)

        try:
            if self.ps == 1:
                self.selected_b = np.load(self.path+f'\\FRET\\{self.n}\\selected_b.npy')
            else:
                self.selected_b = np.ones(self.N_traces)
        except:
            self.selected_b = np.ones(self.N_traces)
        
        if np.any(avg_gg):
            fret_g = (avg_gr / self.gamma_g) / (avg_gr / self.gamma_g + avg_gg)
        else:
            fret_g = np.zeros((self.N_traces, 10))

        if np.any(avg_bb):
            fret_b = (avg_br / self.gamma_g + avg_bg) / (avg_br / self.gamma_g + avg_bg + self.gamma_b * avg_bb)
        else:
            fret_b = np.zeros((self.N_traces, 10))

        for t_num in range(self.N_traces):
            try:
                if (np.average(avg_rr[t_num][self.red_time[0]:self.red_time[1]]) <= self.red_intensity) and self.red == 1:
                    self.selected_g[t_num] = -1
            except:
                if t_num == 0 :
                    print('No red channel')

        for t_num in range(self.N_traces):
            try:
                if (np.average(avg_bb[t_num] + avg_bg[t_num] + avg_br[t_num]) <= self.green_intensity) and self.green == 1:
                    self.selected_b[t_num] = -1
            except:
                if t_num == 0 :
                    print('No green channel')

        print(f"Total Green Traces: {(self.selected_g == 1).sum():.0f}")
        print(f"Total Blue Traces: {(self.selected_b == 1).sum():.0f}")
        path = os.path.join(self.path,'FRET' ,str(self.n))
        if not os.path.exists(path):
            os.makedirs(path)

        if self.ps == 0:
            np.save(self.path+f'\\FRET\\{self.n}\\selected_g.npy', self.selected_g)
            np.save(self.path+f'\\FRET\\{self.n}\\selected_b.npy', self.selected_b)

        self.avg_gg = avg_gg
        self.avg_gr = avg_gr
        self.avg_rr = avg_rr
        self.avg_bb = avg_bb
        self.avg_bg = avg_bg
        self.avg_br = avg_br
        self.fret_g = fret_g
        self.fret_b = fret_b

        print('plotting total intensities')
        self.plot_intensity(avg_gg, 'avg_gg')
        self.plot_intensity(avg_gr, 'avg_gr')
        self.plot_intensity(avg_gg + avg_gr, 'total_g')

        self.plot_intensity(avg_bb, 'avg_bb')
        self.plot_intensity(avg_bg, 'avg_bg')
        self.plot_intensity(avg_br, 'avg_br')
        self.plot_intensity(avg_bb + avg_bg + avg_br, 'total_b')


        path = self.path + f'//FRET//{self.n}'

        if not os.path.exists(path):
            os.makedirs(path)

        np.savez(path+r"/data.npz", 
                gg = avg_gg, 
                gr = avg_gr, 
                bb = avg_bb, 
                bg = avg_bg, 
                br = avg_br, 
                rr = avg_rr, 
                fret_g  = fret_g,
                fret_b = fret_b,
                time_g = avg_time_g,
                time_b = avg_time_b,
                time_r = avg_time_r
                ) 
        
        return fret_g, fret_b