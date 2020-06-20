import os
import cv2
import numpy as np
from scipy.signal import savgol_filter

class Cobb_Angle(object):
    def __init__(self):
        self.input_path='./imgs'
        self.origin_img=None
        self.show_img=None
        self.track=None
        
    def process(self,debug_sign=False):
        for file_ in self._yield_filename(self.input_path):
            self.origin_img=cv2.imread(file_)# h,w,c   g,b,r
            # self.show('origin',self.origin_img,debug_sign)
            
            self.show_img=cv2.cvtColor(self.origin_img,cv2.COLOR_BGR2GRAY)
            # self.show('gray',self.show_img,debug_sign)
    
            self.show_img=self._crop_roi(self.show_img,1.6,0.6)
            # self.show('crop',self.show_img,debug_sign)

            self.show_img=self._change_contrast(self.show_img)
            # self.show('equalizeHist',self.show_img,debug_sign)

            self.show_img=self._drop_pix(self.show_img,0.9)
            # self.show_img=self._drop_pix(self.show_img,0.9)
            # self.show('drop1',self.show_img,debug_sign)

            self.show_img=cv2.medianBlur(self.show_img,3)
            # self.show('medianBlur1',self.show_img,debug_sign)

            self.show_img=self._change_contrast(self.show_img)           
            # self.show('Hist',self.show_img,debug_sign)

            self.show_img=cv2.medianBlur(self.show_img,5)
            # self.show('medianBlur2',self.show_img,debug_sign)
            
            self.show_img=cv2.Sobel(self.show_img,cv2.CV_16S,1,0,ksize=3)
            self.show_img=cv2.convertScaleAbs(self.show_img/4)
            # self.show_img=cv2.convertScaleAbs(self.show_img)/4 #output
            # self.show('sobel',self.show_img,debug_sign)

            self.show_img=cv2.medianBlur(self.show_img,9)
            # self.show('medianBlur3',self.show_img,debug_sign)

            # self.show_img=self._drop_pix(self.show_img,0.9)
            # self.show_img=self._drop_pix(self.show_img,0.9)
            # self.show('drop2',self.show_img,debug_sign)

            _,self.show_img=cv2.threshold(self.show_img,np.mean(self.show_img),255,cv2.THRESH_BINARY)
            # self.show('binary',self.show_img,debug_sign)

            self._get_track(self.show_img)
            self._smooth_track()
            # self.show_img=self._draw_track(self.show_img)
            # self.show('track',self.show_img,debug_sign)

            self.show_keep()

    def show(self,name,img,debug_sign=False):
        if not debug_sign:
            return
        cv2.namedWindow(name,cv2.WINDOW_KEEPRATIO)
        cv2.imshow(name,img)
        
    def show_keep(self):
        key=cv2.waitKey()&0xFF

    def _crop_roi(self,img,ratio_mean,ratio_size):
        H,W=img.shape[0:2]
        mean=np.average(img)
        
        mask=np.where(img>mean/ratio_mean,True,False)
        size=np.count_nonzero(mask,axis=0)#h,w,c
        mask=np.where(size>ratio_size*H)[0]
        return img[:,mask[0]:mask[-1]+1]
    
    def _drop_pix(self,img,ratio_median):
        H,W=img.shape[0:2]
        median=np.median(img,axis=1)#h,w,c

        mask=np.where(img<median.reshape(-1,1)/ratio_median,True,False)
        img[mask]=0
        return img

    def _change_contrast(self,img):
        # img=cv2.equalizeHist(img)
        hist=cv2.calcHist([img],[0],None,[256],[0.0,255.0])
        cdf=np.cumsum(hist)
        cdf=(cdf-cdf[0])/img.size*255
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img[i,j]=cdf[img[i,j]]
        return img

    def _get_track(self,img):
        track=np.zeros((img.shape[0],2))-1
        for i in range(img.shape[0]):# 左右白点索引
            idx=np.where(img[i]==255)#(array([xxx]),)
            try:
                track[i][0],track[i][1]=idx[0][0],idx[0][-1]
            except:
                pass
        track=np.vstack([[range(img.shape[0])],np.mean(track,axis=1)]).T
        self.track=track[np.where(track[:,1]!=-1)].astype(np.int)
    
    def _draw_track(self,img):
        for i in range(1,len(self.track)):#img:(h,w)
            cv2.line(img,(self.track[i][1],self.track[i][0]),\
                        (self.track[i-1][1],self.track[i-1][0]),100,3)#w,h
        return img
        
    def _smooth_track(self):
        self.track[:,1]=savgol_filter(self.track[:,1],101,2)

    def _yield_filename(self,path):
        for home,_,files in os.walk(path):
            for file_ in files:
                if file_.split('.')[-1] in ['jpg','png']:
                    yield os.path.join(home,file_)

def main():
    cobb_angle=Cobb_Angle()
    cobb_angle.process(True)

if __name__ == "__main__":
    main()