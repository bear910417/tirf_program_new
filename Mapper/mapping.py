import h5py
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform
import bm3d
from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt
from skimage.exposure import rescale_intensity
import os
import lmfit
from tqdm import tqdm
import scipy.ndimage
import time
from PIL import Image

class Glimpse_mapping:
    
    def __init__(self, path):
        
        self.path = path
        self.path_g = path + r'\\g'
        self.path_r = path + r'\\r'
        self.path_b = path + r'\\b'
    
    def map(self, mode, seg):
        
        self.seg = seg
        path_g = self.path_g
        path_r = self.path_r
        path_b = self.path_b
        
        if mode == 'g':
            path = path_g
            sw = 1
        elif mode == 'r':
            path = path_r
            sw = 0
        else:
            path = path_b
            sw = 2
            
        plot = False

        tic = time.perf_counter()
        gaussian_peaks2 = np.zeros((3,3,7,7),dtype=np.float32)
        for k in range (0,3):
            for l in range (0,3):     
              offy = -0.5*float(k)
              offx = -0.5*float(1)
              
              for i in range (0, 7): 
                for j in range (0,7):
                  dist = 0.3 * ((float(i)-3.0+offy)**2 + (float(j)-3.0+offx)**2)
                  gaussian_peaks2[k][l][i][j]= 2.0*np.exp(-dist)
                  
        circle = np.zeros((11, 11), dtype=np.int16)
        circle[0] = [ 0,0,0,0,0,0,0,0,0,0,0]
        circle[1] = [ 0,0,0,0,1,1,1,0,0,0,0]
        circle[2] = [ 0,0,0,1,0,0,0,1,0,0,0]
        circle[3] = [ 0,0,1,0,0,0,0,0,1,0,0]
        circle[4] = [ 0,1,0,0,0,0,0,0,0,1,0]
        circle[5] = [ 0,1,0,0,0,0,0,0,0,1,0]
        circle[6] = [ 0,1,0,0,0,0,0,0,0,1,0]
        circle[7] = [ 0,0,1,0,0,0,0,0,1,0,0]
        circle[8] = [ 0,0,0,1,0,0,0,1,0,0,0]
        circle[9] = [ 0,0,0,0,1,1,1,0,0,0,0]
        circle[10]= [ 0,0,0,0,0,0,0,0,0,0,0]    
              

        #g
        file = h5py.File(path+r'\header.mat','r')
        nframes=int(file[r'/vid/nframes'][0][0])
        #print(nframes)
        nframes = 2
        width=int(file[r'/vid/width/'][0][0])
        height=int(file[r'/vid/height/'][0][0])

        filenumber=file[r'/vid/filenumber/'][:].flatten().astype('int')
        offset=file[r'/vid/offset'][:].flatten().astype('int')

        frame=np.zeros((height,width), dtype= np.int16)
        ave_arr = np.zeros((height,width), dtype= np.float32)
        nframes = 10
            
        gfilename = str(filenumber[0]) + '.glimpse'
        gfile_path = path+r'\\'+gfilename
        image_g = np.fromfile(gfile_path, dtype=(np.dtype('>i2') , (height,width)))

        try:
            gfilename = str(1) + '.glimpse'
            gfile_path = path+r'\\'+gfilename
            image_g_1 = np.fromfile(gfile_path, dtype=(np.dtype('>i2') , (height,width)))
            image_g = np.concatenate((image_g,image_g_1))
        except:
            pass
        image_g = image_g + 2**16

        

        
        for j in range(self.seg * 10, self.seg * 10+nframes):
            ave_arr= ave_arr + image_g[j]

        ave_arr = ave_arr/(nframes)
        frame = ave_arr

        temp1 = frame 
        temp1 =scipy.ndimage.filters.uniform_filter(temp1,size=3,mode='nearest')


        aves = np.zeros((int(height/16),int(width/16)), dtype= np.float32)
        for i in range(8,height+1,16):
            for j in range(8,width,16):
                aves[int((i-8)/16)][int((j-8)/16)] = np.round(np.amin(temp1[i-8:i+8,j-8:j+8]),1)


        aves =  scipy.ndimage.zoom(aves, 16,order=1)
        aves = scipy.ndimage.filters.uniform_filter(aves,size=21,mode='nearest')
        bac=aves

        maxf=33722
        #minf=32946

        maxf = np.max(frame)
        #print(maxf)
        minf = np.min(frame)

        frame=rescale_intensity(frame,in_range=(minf,maxf),out_range=np.ubyte)


        dframe= bm3d.bm3d(frame, 6, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
        #dframe = frame
        #plt.imshow(np.concatenate((frame,dframe),axis=1),cmap='Greys_r',vmin=0,vmax=128)
        #plt.savefig(path+r'\\ave.tif',dpi=300)
        plt.close()
        temp1=dframe

        #bac=frame-dframe
        #bac=rescale_intensity(bac,in_range=np.ubyte,out_range=(minf,maxf))
       # plt.imshow(bac,cmap='Greys_r')
       # plt.savefig(path+r'\\back.tif',dpi=300)
       # plt.close()



        left_image  = temp1[0:height, 170 * sw + sw : 170 * sw + 170]
        left_image1  = frame[0:height, 170 * sw + sw :  170 * sw + 170]





        #dcombined_image[:,220:256].fill(0)
        blobs_dog = blob_dog(left_image, min_sigma=1.5/sqrt(2),max_sigma=3.5/sqrt(2), threshold=3 ,overlap=0)
        #blobs_dog = blob_dog(left_image)
        #print(blobs_dog)
        blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

        blobs_dog=blobs_dog[blobs_dog[:,1].squeeze()<165]

        fig = plt.figure(figsize = (170/5, 512/5))   
        ax = fig.add_subplot()
        ax.imshow(left_image,cmap='Greys_r')
        for blob in blobs_dog:
            y, x, r = blob
            c = plt.Circle((x, y), r, color='white',linewidth=0.5, fill=False)
            ax.add_patch(c)
        ax.set_axis_off()
        plt.tight_layout()

        
        cpath=os.path.join(path,r'circled')
        if not os.path.exists(cpath):
            os.makedirs(cpath)
            
        #plt.savefig(cpath+f'\\circled_{mode}.tif', dpi=left_image.shape[0])
        im = Image.fromarray(left_image1)
        im.save(cpath+f'\\circled_{mode}.tif')
        self.left_image = left_image
        #plt.show()
        #plt.close()
        
        return blobs_dog
    
    def get_image(self):
        return self.left_image