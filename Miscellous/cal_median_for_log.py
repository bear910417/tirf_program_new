import numpy as np

path = r'H:\TIRF\20240416_mapping\1\FRET\0'

threshold = 800
start = 10

b = np.load(path + r'\data.npz')['bb'][:, start:start+10]
g = np.load(path + r'\data.npz')['gg'][:, start:start+10]
r = np.load(path + r'\data.npz')['rr'][:, start:start+10]

filter = (np.average(b, axis = 1) > threshold) * (np.average(g, axis = 1) > threshold) * (np.average(r, axis = 1) > threshold)
b = b[filter]
g = g[filter]
r = r[filter]

print('Blue:')
print(f'Average Value = {np.average(b):.2f}')
print(f'Standard Deviation = {np.std(b):.2f}')
print(f'Median Value = {np.median(b):.2f}')
print('\n')
print('Green:')
print(f'Average Value = {np.average(g):.2f}')
print(f'Standard Deviation = {np.std(g):.2f}')
print(f'Median Value = {np.median(g):.2f}')
print('\n')
print('Red:')
print(f' Average Value = {np.average(r):.2f}')
print(f' Standard Deviation = {np.std(r):.2f}')
print(f' Median Value = {np.median(r):.2f}')