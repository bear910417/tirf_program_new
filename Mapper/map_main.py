from Mapper.mapping import Glimpse_mapping
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import lmfit
from scipy.optimize import curve_fit
from tqdm import tqdm
# function to display the coordinates of
# of the points clicked on the image


def click_event(event, x, y, flags, params):
    global img, window
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
 
        # displaying the coordinates
        # on the Shell
        img = params[1]
        window = params[0]
        print(x, ' ', y)
        prior.append([x,y])
        cv2.circle(img, (x,y), 2, color = (255,0,0))
        cv2.imshow(window, img)

def affine(x,y,M):
    x1 = M[0][0] * x + M[0][1] * y + M[0][2]
    y1 = M[1][0] * x + M[1][1] * y + M[1][2]
    
    return [y1, x1]

def x_affine(xy,a1,b1,c1):
    x,y = xy
    x1 = a1 * x + b1 * y + c1
    
    return x1

def y_affine(xy,a2,b2,c2):
    x,y = xy
    y1 = a2 * x + b2 * y + c2
    
    return y1
path = r'J:\YR\20250227_mapping\1'
modes = ['g', 'b']
segs = 4
n = 12
mapper = Glimpse_mapping(path)
f_y = []
f_x = []
f_yo = []
f_xo = []

for seg in range (0, n):
    
    print(f'processing {seg+1} / {n}')
    coord = []
    image = []
    for mode in modes:
        coord.append(mapper.map(mode, seg%segs))
        image.append(mapper.get_image())
 
    print('mapping')  
    # driver function
    
    params = lmfit.Parameters()
    params.add('centery',value = 0)
    params.add('centerx',value = 0)
    params.add('amplitude',value = 5000)
    params.add('sigmay',value = 3)
    params.add('sigmax',value = 3)
    
    if seg  ==0:
        prior = []  
        # reading the image
        img_g = cv2.imread(path +f'\\{modes[0]}\\circled\\circled_{modes[0]}.tif', 1)
        img_r = cv2.imread(path +f'\\{modes[1]}\\circled\\circled_{modes[1]}.tif', 1)
        # displaying the image
        cv2.namedWindow('image_g', cv2.WINDOW_NORMAL)
        cv2.imshow('image_g', img_g)
        
        cv2.namedWindow('image_r', cv2.WINDOW_NORMAL)
        cv2.imshow('image_r', img_r)
        
        # setting mouse handler for the image
        # and calling the click_event() function
        cv2.setMouseCallback('image_g', click_event, ['image_g',img_g])
        cv2.setMouseCallback('image_r', click_event, ['image_r',img_r])
        
        # wait for a key to be pressed to exit
        cv2.waitKey(0)
     
        # close the window
        cv2.destroyAllWindows()
    
        #prior = [[67, 133], [58, 136], [164, 330], [162, 338], [100, 423], [93, 433]]
        #print (prior)
        pts1 = np.float32([prior[0], prior[2], prior[4]])
        pts2 = np.float32([prior[1], prior[3], prior[5]])
        

        M = cv2.getAffineTransform(pts1, pts2)
        print(M)

    
      
    for blob in tqdm(coord[0]):
        quality = 0
        y, x, r = blob
        r = 2
        y=round(y)
        x=round(x)
        trans = affine(blob[1], blob[0], M)
        ty, tx = trans
        ty = round(ty)
        tx = round(tx)
        aoi = image[0][y-r:y+r+1,x-r:x+r+1]
        t_aoi = image[1][ty-r:ty+r+1,tx-r:tx+r+1]

        
        if np.sum(t_aoi) > 700:
            z=image[1][ty-4:ty+4+1,tx-4:tx+4+1].flatten()
            yr=np.zeros(81)
            xr=np.zeros(81)
            for j in range(0,81):
                yr[j]=j//9
                xr[j]=j%9
            
            try:
                model = lmfit.models.Gaussian2dModel()
                result2 = model.fit(z, y=yr, x=xr,params=params)
                redchi2 = result2.redchi
                if  0<result2.best_values['centery'] <9 and 0 < result2.best_values['centerx']<9 :
                    ty = round (ty + result2.best_values['centery'] -4)
                    tx = round (tx + result2.best_values['centerx'] -4)
                    quality = 1
        
                else:
                    quality = 0
            except:
                quality = 0
                #print('fail')
                
        if quality == 1:     
            f_yo.append(y)
            f_xo.append(x)
            f_y.append(ty)
            f_x.append(tx)

    print(len(f_yo))
    poptx, pcov  = curve_fit(x_affine, (f_xo, f_yo), f_x)
    popty, pcov  = curve_fit(y_affine, (f_xo, f_yo), f_y)



    M = [poptx, popty]
    print(M)
    
if not os.path.exists(path+f'\\circled_{modes[0]}_{modes[1]}'):
    os.makedirs(path+f'\\circled_{modes[0]}_{modes[1]}')
    
i=0     
for blob in tqdm(coord[0]):
    quality = 0
    y, x, r = blob
    r = 3
    y=round(y)
    x=round(x)
    trans = affine(blob[1], blob[0], M)
    ty, tx = trans
    ty = round(ty)
    tx = round(tx)
    aoi = image[0][y-r:y+r+1,x-r:x+r+1]
    t_aoi = image[1][ty-r:ty+r+1,tx-r:tx+r+1]
    if np.sum(t_aoi) > 500:
        fig,axes=plt.subplots(1,2)
        plt.axis('off')
        i=i+1
        axes[0].set_xticks([])
        axes[1].set_xticks([])
        axes[0].set_yticks([])
        axes[1].set_yticks([])
        axes[0].imshow(aoi,cmap='Greys_r',vmin=0,vmax=128)        
        axes[1].imshow(t_aoi,cmap='Greys_r',vmin=0,vmax=128)
        axes[0].set_title(f'{np.sum(aoi):.2f}')
        axes[1].set_title(f'{np.sum(t_aoi):.2f}')
        plt.savefig(path + f'\\circled_{modes[0]}_{modes[1]}\\{i}.tif')
        #plt.show()
        plt.close()      

np.save(path + f'\\map_{modes[0]}_{modes[1]}.npy', M)



    




