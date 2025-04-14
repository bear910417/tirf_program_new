from GMM_custom import GMM

path = r'H:\TIRF\20231102\lane1\R_snap\2\FRET\0'

init = [0.18, 0.57, 0.75, 0.87]
ignore = [0.15]
text = True
covariance_type = 'diag'
custom_name = 'dT9_R'

gmm = GMM(path = path)
gmm.load_data(channel = 'fret_g')
gmm.fit(smooth = 10, init = init, covariance_type = covariance_type)
gmm.plot_and_save(text = text, ignore = ignore, custom_name = custom_name)


