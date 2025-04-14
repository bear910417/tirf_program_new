import numpy as np
import os
import time as rtime
from scipy.ndimage import uniform_filter1d
from sklearn import mixture 
from dash.exceptions import PreventUpdate
from Trace_viewer.Gaussian_mixture.GMM_custom import GMM
from math import sqrt
import matplotlib
import pickle
import shutil
import matplotlib.pyplot as plt  # Ensure matplotlib is imported


def uf(t, lag, axis = -1):
    return uniform_filter1d(t, size = lag, mode = 'nearest', axis = axis)
def sa(t, lag):

    t = t[:t.shape[0] - (t.shape[0] % lag)]
    t = t.reshape(-1, lag)
    return np.average(t, axis = 1)


def update_trace(fig, relayout, i, scatter, fret_g, fret_b, rr, gg, gr, bb, bg, br, time, hmm_fret_g, bkps, lag, smooth_mode, show):
    uf_time_b = np.zeros(10)
    uf_time_g = np.zeros(10)
    uf_time_r = np.zeros(10)

    mode_dict = {
        0 : 'lines',
        1 : 'markers'
    }


    try:
        hist_range = (relayout['xaxis.range[0]'], relayout['xaxis.range[1]'])
    except:
        hist_range = (0, np.inf)

    if np.any(fret_b):
        if smooth_mode == 'moving':
            uf_time_b = uf(time['b'], lag)
            fig.update_traces(x = uf_time_b, y = uf(fret_b[i], lag), mode = mode_dict[scatter], visible = ('FRET BG' not in show), selector = dict(name='fret_b'))
            fig.update_traces(x = uf_time_b, y = uf(bb[i], lag), mode = mode_dict[scatter], visible = ('BB' not in show), selector = dict(name='bb'))
            fig.update_traces(x = uf_time_b, y = uf(bg[i], lag), mode = mode_dict[scatter], visible = ('BG' not in show), selector = dict(name='bg'))
            fig.update_traces(x = uf_time_b, y = uf(br[i], lag), mode = mode_dict[scatter], visible = ('BR' not in show), selector = dict(name='br'))
            
            fig.update_traces(x = uf_time_b, y = uf(bb[i]+ bg[i]+ br[i], lag), mode = mode_dict[scatter], line = dict(dash = "longdash", width = 2), visible = ('Tot B' not in show), selector = dict(name='tot_b'))

            fig.update_traces(x = uf_time_b[[x[0] for x in bkps['b'][i]]], y = uf(bb[i], lag)[[y[0] for y in bkps['b'][i]]], mode = 'markers', selector = dict(name='b_bkps'))
            fig.update_traces(x = uf_time_b[[x[0] for x in bkps['fret_b'][i]]], y = uf(fret_b[i], lag)[[y[0] for y in bkps['fret_b'][i]]], mode = 'markers', selector = dict(name='fret_b_bkps'))

            hfilt_b = (uf_time_b > hist_range[0]) * (uf_time_b < hist_range[1])   
            fig.update_traces(y = uf(fret_b[i], lag)[hfilt_b], selector = dict(name='Histogram_b'))

            if ('Tot B'  in show):
                fig.update_layout(yaxis4 = dict(range = (0, np.max(np.concatenate((uf(bb[i], lag), uf(bg[i], lag), uf(br[i], lag))))))),
            else:
                fig.update_layout(yaxis4 = dict(range = (0, np.max((uf(bb[i], lag) + uf(bg[i], lag) + uf(br[i], lag)))))),


        elif smooth_mode == 'strided':
            sa_time_b = sa(time['b'], lag)
            fig.update_traces(x = sa_time_b, y = sa(fret_b[i], lag), mode = mode_dict[scatter], visible = ('FRET BG' not in show), selector = dict(name='fret_b'))
            fig.update_traces(x = sa_time_b, y = sa(bb[i], lag), mode = mode_dict[scatter], visible = ('BB' not in show), selector = dict(name='bb'))
            fig.update_traces(x = sa_time_b, y = sa(bg[i], lag), mode = mode_dict[scatter], visible = ('BG' not in show), selector = dict(name='bg'))
            fig.update_traces(x = sa_time_b, y = sa(br[i], lag), mode = mode_dict[scatter], visible = ('BR' not in show), selector = dict(name='br'))
            
            fig.update_traces(x = sa_time_b, y = sa(bb[i]+ bg[i]+ br[i], lag), mode = mode_dict[scatter], line = dict(dash = "longdash", width = 2), visible = ('Tot B' not in show), selector = dict(name='tot_b'))

            fig.update_traces(x = sa_time_b[[(x[0] // lag) for x in bkps['b'][i]]], y = sa(bb[i], lag)[[(y[0] // lag) for y in bkps['b'][i]]], mode = 'markers', selector = dict(name='b_bkps'))
            fig.update_traces(x = sa_time_b[[(x[0] // lag) for x in bkps['fret_b'][i]]], y = sa(fret_b[i], lag)[[(y[0] // lag) for y in bkps['fret_b'][i]]], mode = 'markers', selector = dict(name='fret_b_bkps'))

            hfilt_b = (sa_time_b > hist_range[0]) * (sa_time_b < hist_range[1])   
           
            fig.update_traces(y = sa(fret_b[i], lag)[hfilt_b], selector = dict(name='Histogram_b'))
        
            if ('Tot B'  in show):
                fig.update_layout(yaxis4 = dict(range = (0, np.max(np.concatenate((sa(bb[i], lag), sa(bg[i], lag), sa(br[i], lag))))))),
            else:
                fig.update_layout(yaxis4 = dict(range = (0, np.max((sa(bb[i], lag) + sa(bg[i], lag) + sa(br[i], lag)))))),

        

      
    else:
        selectors = ['fret_b', 'bb', 'bg', 'br', 'tot_b', 'b_bkps', 'fret_b_bkps']
        clear_trace(fig, selectors)
       
        
   
    if np.any(fret_g):
        if smooth_mode == 'moving':
            uf_time_g = uf(time['g'], lag)
            fig.update_traces(x = uf_time_g, y = uf(fret_g[i], lag), mode = mode_dict[scatter], visible = ('FRET GR' not in show), selector = dict(name='fret_g'))
            fig.update_traces(x = uf_time_g, y = uf(gg[i], lag), mode = mode_dict[scatter], visible = ('GG' not in show), selector = dict(name='gg'))
            fig.update_traces(x = uf_time_g, y = uf(gr[i], lag), mode = mode_dict[scatter], visible = ('GR' not in show), selector = dict(name='gr'))
            fig.update_traces(x = uf_time_g, y = uf(gg[i]+ gr[i], lag), mode = mode_dict[scatter], line = dict(dash = "longdash", width = 2), visible = ('Tot G' not in show), selector = dict(name='tot_g'))

            fig.update_traces(x = uf_time_g[[x[0] for x in bkps['g'][i]]], y = uf(gg[i], lag)[[y[0] for y in bkps['g'][i]]], mode = 'markers', selector = dict(name='g_bkps'))
            fig.update_traces(x = uf_time_g[[x[0] for x in bkps['fret_g'][i]]], y = uf(fret_g[i], lag)[[y[0] for y in bkps['fret_g'][i]]], mode = 'markers', selector = dict(name='fret_g_bkps'))

            hfilt_g = (uf_time_g > hist_range[0]) * (uf_time_g < hist_range[1])       
            fig.update_traces(y = uf(fret_g[i], lag)[hfilt_g], selector = dict(name='Histogram_g'))

        elif smooth_mode == 'strided':
            sa_time_g = sa(time['g'], lag)
            fig.update_traces(x = sa_time_g, y = sa(fret_g[i], lag), mode = mode_dict[scatter], visible = ('FRET GR' not in show), selector = dict(name='fret_g'))
            fig.update_traces(x = sa_time_g, y = sa(gg[i], lag), mode = mode_dict[scatter], visible = ('GG' not in show), selector = dict(name='gg'))
            fig.update_traces(x = sa_time_g, y = sa(gr[i], lag), mode = mode_dict[scatter], visible = ('GR' not in show), selector = dict(name='gr'))
            fig.update_traces(x = sa_time_g, y = sa(gg[i]+ gr[i], lag), mode = mode_dict[scatter], line = dict(dash = "longdash", width = 2), visible = ('Tot G' not in show), selector = dict(name='tot_g'))

            fig.update_traces(x = sa_time_g[[(x[0] // lag) for x in bkps['g'][i]]], y = sa(gg[i], lag)[[(y[0] // lag) for y in bkps['g'][i]]], mode = 'markers', selector = dict(name='g_bkps'))
            fig.update_traces(x = sa_time_g[[(x[0] // lag) for x in bkps['fret_g'][i]]], y = sa(fret_g[i], lag)[[(y[0] // lag) for y in bkps['fret_g'][i]]], mode = 'markers', selector = dict(name='fret_g_bkps'))

            hfilt_g = (sa_time_g > hist_range[0]) * (sa_time_g < hist_range[1])      
            fig.update_traces(y = sa(fret_g[i], lag)[hfilt_g], selector = dict(name='Histogram_g'))

        if ('Tot G' in show):
            if ('RR' in show):
                fig.update_layout(yaxis3 = dict(range = (0, np.max(np.concatenate((gg[i], gr[i]))))))
            else:
                fig.update_layout(yaxis3 = dict(range = (0, np.max(np.concatenate((gg[i], gr[i], rr[i]))))))
        else:
            if ('RR' in show):
                fig.update_layout(yaxis3 = dict(range = (0, np.max(gg[i] + gr[i]))))
            else:
                fig.update_layout(yaxis3 = dict(range = (0, np.max(np.concatenate((gg[i] + gr[i], rr[i]))))))

    else:
        selectors = ['fret_g', 'gg', 'gr', 'tot_g', 'g_bkps', 'fret_g_bkps']
        clear_trace(fig, selectors)


    if np.any(rr):
        if smooth_mode == 'moving':
            uf_time_r = uf(time['r'], lag)
            fig.update_traces(x = uf_time_r, y = uf(rr[i], lag), mode = mode_dict[scatter], visible = ('RR' not in show), selector = dict(name='rr'))
            fig.update_traces(x = uf_time_r[[x[0] for x in bkps['r'][i]]], y = uf(rr[i], lag)[[y[0] for y in bkps['r'][i]]], mode = 'markers', selector = dict(name='r_bkps'))
        elif smooth_mode == 'strided':
            sa_time_r = sa(time['r'], lag)
            fig.update_traces(x = sa_time_r, y = sa(rr[i], lag), mode = mode_dict[scatter], visible = ('RR' not in show), selector = dict(name='rr'))
            fig.update_traces(x = sa_time_r[[(x[0] // lag) for x in bkps['r'][i]]], y = sa(rr[i], lag)[[(y[0] // lag) for y in bkps['r'][i]]], mode = 'markers', selector = dict(name='r_bkps'))

        if not np.any(fret_g):
            fig.update_layout(yaxis3 = dict(range = (0, np.max(rr[i])))),
    else:
        selectors = ['rr', 'r_bkps']
        clear_trace(fig, selectors)

    if np.any(hmm_fret_g[i]):
        fig.update_traces(x = uf(time['g'], lag)[:hmm_fret_g[i].shape[0]], y = hmm_fret_g[i].reshape(-1), visible = ('HMM' not in show), selector = dict(name='HMM_fret_g'))
    else:
        fig.update_traces(x = [], y = [], selector = dict(name='HMM_fret_g'))

    
    if np.any(fret_b) or np.any(fret_g) or np.any(rr):
        fig.update_layout(xaxis1 = dict(range=(min(time['g'][0], time['b'][0], time['r'][0]), max(time['g'][-1], time['b'][-1], time['r'][-1]))))


    return fig

def clear_trace(fig, selectors):
        for s in selectors:
            fig.update_traces(x = [], y = [], selector = dict(name = s))


def change_trace(changed_id, event, i, N_traces, fig):
    if ('next' in changed_id) or (('n_events' in changed_id) and event['key'] == 'w'):
        if i < N_traces-1:
            i = i+1
            fig.layout.shapes=[]
            #fig = draw(fig, tot_bkps, i, time_g, dead_time, color, total_frame)
           
    elif ('previous' in changed_id)  or (('n_events' in changed_id) and event['key'] == 'q'):
        if i>0:
            i = i-1
            fig.layout.shapes=[]
            #fig = draw(fig,tot_bkps,i,time_gr,dead_time,color,total_frame)
              
    elif 'tr_go' in changed_id:
       
        try:
            if int(i) < N_traces:
                i = int(i)
            else:
                i = 0
        except:
            i = 0

        fig.layout.shapes=[]
    
    return i, fig

def select_good_bad(changed_id, event, i, select_list_g):
    
    
    if ('set_good' in changed_id) or (('n_events' in changed_id) and event['key'] == 'z'):  
        if select_list_g[i] == 1:
            select_list_g[i] = 0
        else:
            select_list_g[i] = 1
        
    if ('set_bad' in changed_id ) or (('n_events' in changed_id) and event['key'] == 'x'):  
        if select_list_g[i] == -1:
            select_list_g[i] = 0
        else:
            select_list_g[i] = -1
         
    return select_list_g

def select_colocalized(changed_id, event, i, colocalized_list):
    
    
    if ('set_colocalized' in changed_id) or (('n_events' in changed_id) and event['key'] == 'c'):  
        if colocalized_list[i] == 1:
            colocalized_list[i] = 0
        else:
            colocalized_list[i] = 1
         
    return colocalized_list

def render_colocalized(i, colocalized_list):
    white_button_style = {'background-color': '#f0f0f0', 'color': 'black'}
    blue_button_style = {'background-color': 'blue', 'color': 'white'}   
    if colocalized_list.shape[0] == 0:
        style = white_button_style
        return style
    
    if colocalized_list[i] == 1:
        style = blue_button_style
    elif colocalized_list[i] == 0:
        style = white_button_style

    return style


def render_good_bad(i, select_list_g):
    white_button_style = {'background-color': '#f0f0f0', 'color': 'black'}
    red_button_style = {'background-color': 'red', 'color': 'white'}                        
    green_button_style = {'background-color': 'green', 'color': 'white'}

    if select_list_g.shape[0] == 0:
        good_style = white_button_style
        bad_style = white_button_style
        return good_style, bad_style

    if select_list_g[i] == 1:
        good_style = green_button_style
        bad_style = white_button_style
    elif select_list_g[i] == 0:
        good_style = white_button_style
        bad_style = white_button_style
    else:
        good_style = white_button_style
        bad_style = red_button_style
    
    return good_style, bad_style


def breakpoints_utils(changed_id, clickData, mode, channel, i, time, bkps, smooth, smooth_mode):
    trans = {
        0 : 'fret_g',
        1 : 'fret_b',
        2 : 'b',
        3 : 'b',
        4 : 'b',
        5 : 'b',
        6 : 'g',
        7 : 'g',
        8 : 'g',
        9 : 'r', 
        10 : 'fret_g',
        11 : 'fret_b',
        12: 'b',
        13: 'g',
        14: 'r',
    }

    confirm_reset_show = False
    if 'dtime' in changed_id and channel != None:
        if mode == 'Add':

            bkps[channel][i].append((0, time[channel][0]))
            bkps[channel][i] = sorted(bkps[channel][i])                   

        elif mode == 'Remove':
     
            try:
                bkps[channel][i].pop(0)
                bkps[channel][i] = sorted(bkps[channel][i]) 
            except:
                pass
    
    if 'etime' in changed_id and channel != None:
        
        if mode == 'Add':

            bkps[channel][i].append((time[channel].shape[0]-1, time[channel][-1]))
            bkps[channel][i] = sorted(bkps[channel][i])                   

        elif mode == 'Remove':
     
            bkps[channel][i].pop(-1)
            bkps[channel][i] = sorted(bkps[channel][i]) 


    if 'graph.clickData' in changed_id:
        if isinstance(clickData, dict):
            c_num = clickData["points"][0]["curveNumber"]
            channel = trans[c_num]

            if mode == 'Add':
                if c_num <10:
                    if  smooth_mode == 'moving':
                        idx = clickData["points"][0]["pointNumber"]
                        idx_t = uf(time[channel], smooth)[idx]
                    elif smooth_mode == 'strided':
                        idx = clickData["points"][0]["pointNumber"]
                        idx_t = sa(time[channel], smooth)[idx]
                        idx = int(smooth * (idx + 0.5))  

                    bkps[channel][i].append((idx, idx_t))
                    bkps[channel][i] = sorted(bkps[channel][i])  
                    #fig.layout.shapes=[]
                    #fig = draw(fig,tot_bkps,i,time_gr,dead_time,color,total_frame)  
                            
            elif mode == 'Remove':     
                if 10 <= c_num <=14:        
                    rem_idx = clickData["points"][0]["pointNumber"]
                    bkps[channel][i].pop(rem_idx)            
                    # fig.layout.shapes=[]
                    # fig = draw(fig,tot_bkps,i,time_gr,dead_time,color,total_frame) 
                    
            elif mode == 'Except':
                if 10 <= c_num <=14:
                    exp_idx = clickData["points"][0]["pointNumber"]
                    bkps[channel][i] = [bkps[channel][i][exp_idx]]      

                    # fig.layout.shapes=[]
                    # fig = draw(fig,tot_bkps,i,time_gr,dead_time,color,total_frame)
    
                    
    if mode == 'Clear':
        mode = "Add"
        if channel != None:
            bkps[channel][i] = []
        #fig.layout.shapes=[]

    if mode == 'Clear All':
        for channel in bkps:
            bkps[channel][i] = []
        #fig.layout.shapes=[]
        mode = "Add"
        
    if mode == 'Set All':
        mode = "Add"
        if channel != None:
            for keys in bkps:
                bkps[keys][i] = bkps[channel][i]
    
    if mode =='Reset':
        mode = "Add"
        confirm_reset_show = True
    
    if 'confirm-reset' in changed_id:
        for channel in bkps:
            for i in range(len(bkps[channel])):
                bkps[channel][i] = []

    return bkps, mode, confirm_reset_show

def sl_bkps(changed_id, path, bkps, mode):
    
    if ('save_bkps' in changed_id) or (mode == 'Clear All'):
        
        # if not os.path.exists(path+r"/images"):
        #     os.mkdir(path+"/images")
        # fig.write_image(path+f"/images/trace{i}.png", engine="kaleido",width=1600,height=800,scale=10)
        mode = 'Add'
        try:
            seconds = rtime.time()
            t = rtime.localtime(seconds)
            shutil.copy(path+r'/breakpoints.npz', path+f'/breakpoints_backup_{t.tm_hour}_{t.tm_min}_{t.tm_sec}.npz')    
        except:
            print('No existing save file found.')
        
        for key in bkps:
            bkps[key] = np.array(bkps[key], dtype = object)

        np.savez(path+r'/breakpoints.npz', **bkps)
        print('file_saved')
        
    if 'load_bkps' in changed_id:
            try:
                bkps = dict(np.load(path+r'/breakpoints.npz', allow_pickle=True))
            except:
                print('File not found')
    return bkps

def find_chp(changed_id, fret_g, fret_b, rr, gg, gr, bb, bg, br, i, time, select_list_g, 
             chp_mode_0, chp_comp_0, chp_thres_0, chp_channel_0, chp_target_0, chp_mode_1, chp_comp_1, chp_thres_1, chp_channel_1, chp_target_1,
             bkps, smooth, smooth_mode):
    
    select_list_g = np.array(select_list_g)
    
    if 'chp_find_0' in changed_id:
        channel = chp_channel_0
        mode = chp_mode_0
        comp = chp_comp_0
        thres = chp_thres_0
        target_mode = chp_target_0
    elif 'chp_find_1' in changed_id:
        channel = chp_channel_1
        mode = chp_mode_1
        comp = chp_comp_1
        thres = chp_thres_1
        target_mode = chp_target_1
    
    else:
        return bkps
    
    if target_mode == 'current trace':
        i_list = [i]
    elif target_mode == 'all traces':
        i_list = np.arange(0, locals()[channel].shape[0])
    elif target_mode == 'all good':
        i_list = np.arange(0, locals()[channel].shape[0])[select_list_g == 1]


    for i in i_list:
        if  smooth_mode == 'moving':
            trace = uf(locals()[channel][i], smooth)
        elif smooth_mode == 'strided':
            trace = sa(locals()[channel][i], smooth)

        
        if comp == 'bigger':
            target = np.where(trace > thres)[0]
        elif comp == 'smaller':
            target = np.where(trace < thres)[0]
        
        if target.shape[0] < 1:
            print(f'No valid points found for {i}.')
            if  smooth_mode == 'moving':
                target_t = uf(time[channel], smooth)[-1]
                target = time[channel].shape[0] -1
            elif smooth_mode == 'strided':
                target_t = sa(time[channel], smooth)[-1]
                target = time[channel].shape[0] -1
                print(target)
            bkps[channel][i].append((target, target_t))
            continue

        if mode == 'first':
            target = target[0]
        elif mode == 'second':
            target = target[1]
        elif mode == 'previous':
            target = target[0] - 1
            if target < 0:
                print(f'The first point meets the threshold for {i}.')
                target_t = time[channel][0]
                target = 0
                bkps[channel][i].append((target, target_t))
                continue


        if  smooth_mode == 'moving':
            target_t = uf(time[channel], smooth)[target]
            bkps[channel][i].append((target, target_t))
        elif smooth_mode == 'strided':
            target_t = sa(time[channel], smooth)[target]
            target = int(smooth * (target + 0.5))  
            bkps[channel][i].append((target, target_t))
            
    return bkps



def show_blob(blobs, fig_blob, smooth, i, hoverData, time, aoi_max):
    if blobs == None or hoverData == None:
        return fig_blob
    
    t = hoverData["points"][0]["x"]
  
    b = blobs['b'] 
    g = blobs['g'] 
    r = blobs['r'] 


    minf = min(int(blobs['minf']), int(aoi_max[0]))
    maxf = max(int(blobs['minf']), int(aoi_max[1]))


    z_list = []
    if np.any(b):
        uf_time_b = uf(time['b'], smooth, 0)
        x = np.searchsorted(uf_time_b, t)
        z_list = z_list + [np.average(b[i, 0, x:x+smooth], axis = 0), np.average(b[i, 1, x:x+smooth], axis = 0), np.average(b[i, 2, x:x+smooth], axis = 0)]
    else:
        z_list = z_list + [np.zeros((9, 9)), np.zeros((9, 9)), np.zeros((9, 9))]

    if np.any(g):  
        uf_time_g = uf(time['g'], smooth, 0) 
        x = np.searchsorted(uf_time_g, t)
        z_list = z_list + [np.average(g[i, 0, x:x+smooth], axis = 0), np.average(g[i, 1, x:x+smooth], axis = 0)]
    else:
        z_list = z_list + [np.zeros((9, 9)), np.zeros((9, 9))]

    if np.any(r):
        uf_time_r = uf(time['r'], smooth, 0)
        x = np.searchsorted(uf_time_r, t)
        z_list = z_list + [np.average(r[i, 0, x:x+smooth], axis = 0)]
    else:
        z_list = z_list + [np.zeros((9, 9))]

    z = np.concatenate(z_list, axis = 1)

    fig_blob['layout']['coloraxis']['colorscale'] = 'gray'
    fig_blob.update_traces(zmax = maxf, zmin = minf, selector = dict(type = 'heatmap')) 
    fig_blob['data'][0]['z'] = z
    fig_blob['layout']['coloraxis']['cmax'] = maxf
    fig_blob['layout']['coloraxis']['cmin'] = minf

    return fig_blob


def gaussian(x, A, mu1, sigma1):
    y=A * (1/(sigma1*sqrt(2*np.pi)))*np.exp(-1.0 * (x - mu1)**2 / (2 * sigma1**2))
    return y

def fit_gmm(FRET_list, select_list, init_means, covariance_type, n_comps):

    gmm = GMM(data = FRET_list, selected = select_list)
    means, covs, weights, X, n_components = gmm.fit(smooth = 10, init = init_means, covariance_type = covariance_type, n_components = n_comps)

    return means, covs, weights, X, gmm



def draw_gmm(fig, m, c, w, X):
    yconvs = np.zeros((100000))
    xspace = np.linspace(0, 1, 100000)
    fig.update_traces(x = X.reshape(-1), selector = dict(name='gmm_hist'))
    fig.data = [list(fig.data)[0]]
    fig.layout.annotations = []

    if len(c) != len(m):
        c = np.ones(len(m)) * c

    for i in range(0, m.shape[0]):
        yconvs = yconvs + gaussian(xspace, w[i],m[i],c[i])
        fig.add_scatter(x = xspace, y = gaussian(xspace,w[i],m[i],c[i]), name = f'ysep{i}', marker=dict(color='orange'), line = dict(dash ='dash')  )  
        fig.add_annotation(x = m[i], y=int(np.max(gaussian(xspace,w[i],m[i],c[i]))/2), text = f'{np.round(w[i], 2)*100:.0f}%', showarrow=False, yshift=10)    

    fig.add_scatter(x = xspace, y = yconvs, marker=dict(color='orange'))

    return fig

def save_gmm(gmm, path):
    matplotlib.use('agg')
    gmm.plot_and_save(save_path = path)



def plot_fret_trace(time_array, fret_array, trace_index, base_path="."):
    """
    Plots a clean FRET vs. time figure with a green line and minimal styling,
    and saves the figure in a subfolder called "traces" under the base_path.

    Parameters:
        time_array : array-like
            The time points for the x-axis.
        fret_array : array-like
            The FRET values for the y-axis.
        trace_index : int
            The current trace index (used for the file name).
        base_path : str, optional
            The base directory in which to create a "traces" subfolder.
    
    Returns:
        file_path : str
            The full path to the saved figure.
    """
    # Create the matplotlib figure and axis.
    fig, ax = plt.subplots(figsize=(6, 2.5), dpi=100)
    ax.plot(time_array, fret_array, color='green', linewidth=1.5)
    ax.set_xlabel('time (s)', fontsize=12)
    ax.set_ylabel('FRET', fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, None)
    
    # Remove top and right spines for a minimalist look.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', length=4, width=1)
    plt.tight_layout()

    # Define the subfolder "traces" and create it if it doesn't exist.
    traces_folder = os.path.join(base_path, "traces")
    if not os.path.exists(traces_folder):
        os.makedirs(traces_folder)
    
    # Construct the file path for saving the figure.
    file_path = os.path.join(traces_folder, f"fret_g_trace_{trace_index}.png")
    
    # Save the figure and close it.
    plt.savefig(file_path, bbox_inches='tight', dpi = 300)
    plt.close(fig)
    
    return file_path





        
    
                 