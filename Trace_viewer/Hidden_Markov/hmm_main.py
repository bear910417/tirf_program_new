from hmm_fitter_new import HMM_fitter 


path=r'H:\TIRF\20231102\lane1\RM\1\FRET\0'


r = False
n_iter = 20
plot = True    
p_length = 60
text = True 
mode = 'eps'


hfit = HMM_fitter(path)
hfit.load_traces()
hfit.fitHMM(r, n_iter = n_iter)
hfit.cal_states(plot, p_length = p_length, text = text, mode = mode)

