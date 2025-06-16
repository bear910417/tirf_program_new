import numpy as np

map_path = r'D:\TIRF_Program\Bkp_picker\mapping\20240229'
date = "20240229"

map = np.load(map_path+r'\map_g_r.npy')
print(map)
new_map = np.zeros(32)
new_map[0] = map[0][2]
new_map[1] = map[0][1]
new_map[4] = map[0][0]

new_map[16] = map[1][2]
new_map[17] = map[1][1]
new_map[20] = map[1][0]

print(new_map)
np.savetxt(map_path+ f'\\map_idl_{date}.map', new_map)