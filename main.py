import os
import cv2
import numpy as np

def sobel_filter(img):#边缘检测
    h, w = img.shape

    for w_i in range(1, w - 1):
        for h_i in range(1, h - 1):#左右像素点差值
            x_sobel = (img[h_i - 1, w_i + 1]
                       + 2 * img[ h_i, w_i + 1]
                       + img[ h_i + 1,w_i + 1]) \
                - (img[h_i - 1,w_i - 1]
                   + 2 * img[ h_i,w_i - 1]
                   + img[h_i + 1,w_i - 1])

            img[h_i,w_i]= int(abs(x_sobel / 4))

class Cobb_Angle(object):
    def __init__(self):
        self.input_path='./imgs'
        self.origin_img=None
        self.show_img=None
        
    
    def process(self,debug_sign=False):
        for file_ in self._yield_filename(self.input_path):
            self.origin_img=cv2.imread(file_)# h,w,c   g,b,r
            self.show('origin',self.origin_img,debug_sign)
            
            self.show_img=cv2.cvtColor(self.origin_img,cv2.COLOR_BGR2GRAY)
            self.show('gray',self.show_img,debug_sign)
            # self.show_img=cv2.imread(file_,cv2.IMREAD_GRAYSCALE)
    
            self._crop_roi(self.show_img,1.6,0.6)
            self.show('crop',self.show_img,debug_sign)

            self.show_img=cv2.equalizeHist(self.show_img)
            self.show('equalizeHist',self.show_img,debug_sign)

            self._drop_pix(self.show_img,0.9)
            self._drop_pix(self.show_img,0.9)
            self.show('drop',self.show_img,debug_sign)

            self.show_img=cv2.medianBlur(self.show_img,3)
            self.show('medianBlur',self.show_img,debug_sign)

            self.show_img=cv2.equalizeHist(self.show_img)

            self.show_img=cv2.medianBlur(self.show_img,5)
            sobel_filter(self.show_img)
            
            self.show('last',self.show_img,debug_sign)

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
        mean=np.mean(img)
        
        mask=np.where(img>mean/ratio_mean,True,False)
        size=np.count_nonzero(mask,axis=0)#h,w,c
        mask=np.where(size>ratio_size*H)[0]
        self.show_img=self.show_img[:,mask[0]:mask[-1]+1]
    
    def _drop_pix(self,img,ratio_median):
        H,W=img.shape[0:2]
        median=np.median(img,axis=1)#h,w,c

        mask=np.where(img<median.reshape(-1,1)/ratio_median,True,False)
        img[mask]=0

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