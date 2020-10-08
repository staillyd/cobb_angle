import os
import cv2
import numpy as np
from scipy.signal import savgol_filter
# from scipy.optimize import least_squares
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
# from sympy import diff,symbols,solve

class Cobb_Angle(object):
    def __init__(self):
        self.input_path='./imgs'
        self.origin_img=None
        self.show_img=None
        self.track=None
        self.tmp_dir='./tmp/'
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def process_for_ui(self,file_):
        self.origin_img=cv2.imread(file_)# h,w,c   g,b,r
        self.show_img=cv2.cvtColor(self.origin_img,cv2.COLOR_BGR2GRAY)
        self.show_img=self._crop_roi(self.show_img,1.6,0.6)
        self.show_img=self._change_contrast(self.show_img)
        self.show_img=self._drop_pix(self.show_img,0.9)
        self.show_img=cv2.medianBlur(self.show_img,3)
        self.show_img=self._change_contrast(self.show_img)           
        self.show_img=cv2.medianBlur(self.show_img,5)    
        self.show_img=cv2.Sobel(self.show_img,cv2.CV_16S,1,0,ksize=3)
        self.show_img=cv2.convertScaleAbs(self.show_img/4)
        self.show_img=cv2.medianBlur(self.show_img,9)
        _,self.show_img=cv2.threshold(self.show_img,np.mean(self.show_img),255,cv2.THRESH_BINARY)
        self._get_track(self.show_img)
        self._smooth_track()
        self.show_img=cv2.cvtColor(self.show_img,cv2.COLOR_GRAY2BGR)

        self._fit_track()
        self.show_img=self._draw_fit_track(self.show_img)

        self._get_turning_and_diff_point()
        self.show_img=self._draw_turning_and_diff_point(self.show_img)

        self._get_theta_by_turnning_points()
        self.show_img=self._draw_theta_from_turnning_points(self.show_img)

        # self._get_theta_by_tangent()
        # self.show_img=self._draw_theta_from_tangent(self.show_img)
        # self.show('theta_by_tangent',self.show_img,debug_sign)

        cv2.imwrite(self.tmp_dir+'origin.jpg',self.origin_img)
        cv2.imwrite(self.tmp_dir+'result.jpg',self.show_img)
 
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

            self.show_img=cv2.cvtColor(self.show_img,cv2.COLOR_GRAY2BGR)

            self._fit_track()
            self.show_img=self._draw_fit_track(self.show_img)
            self.show('fit track',self.show_img,debug_sign)

            self._get_turning_and_diff_point()
            self.show_img=self._draw_turning_and_diff_point(self.show_img)
            self.show('turning point',self.show_img,debug_sign)

            self._get_theta_by_turnning_points()
            self.show_img=self._draw_theta_from_turnning_points(self.show_img)
            self.show('theta_by_turnning_points',self.show_img,debug_sign)

            # self._get_theta_by_tangent()
            # self.show_img=self._draw_theta_from_tangent(self.show_img)
            # self.show('theta_by_tangent',self.show_img,debug_sign)

            self.show_keep()
            plt.show()

    def show(self,name,img,debug_sign=False):
        if not debug_sign:
            return
        cv2.namedWindow(name,cv2.WINDOW_KEEPRATIO)
        cv2.imshow(name,img)
        
    def show_keep(self):
        key=cv2.waitKey()& 0xFF==ord('q')

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
        track=track[:int(29/30*img.shape[0])]#去除最底部的干扰
        self.track=track[np.where(track[:,1]!=-1)].astype(np.int)
    
    def _draw_track(self,img):
        for i in range(1,len(self.track)):#img:(h,w)
            cv2.line(img,(self.track[i][1],self.track[i][0]),\
                        (self.track[i-1][1],self.track[i-1][0]),100,3)#w,h
        return img
        
    def _smooth_track(self):
        '''平滑曲线'''
        self.track[:,1]=savgol_filter(self.track[:,1],101,2)

    def _fit_err(self,param,x,y):
        '''多项式拟合的误差'''
        return self._get_fit_vals(param,x)-y

    def _get_fit_vals(self,param,x):
        '''多项式拟合'''
        val=0
        for i in range(len(param)):
            val+=param[i]*x**i
        return val

    # def _get_sympy_derivative(self,model):#导数为0是拐点，二阶导数为0判断凹凸。没必要用导数，直接对拟合值求diff判断+-即可得到拐点
    #     '''求导数'''
    #     x=symbols('x')
    #     powers=model.steps[0][1].powers_.reshape(-1)
    #     weights=model.steps[1][1].coef_.reshape(-1);weights[0]=model.steps[1][1].intercept_.reshape(-1)
    #     y=0
    #     for i in range(len(powers)):#由于是岭回归,正则项对x求导为0,不用管
    #         y+=weights[i]*x**powers[i]
    #     solve(diff(y,x),x)

    def _fit_track(self,plt_flag=True):
        # param_init=[0,0,0,0,0,0]
        # param=least_squares(self._fit_err,param_init,args=(self.track[:,0],self.track[:,1])).x
        # x=np.array([i for i in range(self.show_img.shape[0])])#np.linspace(0, self.show_img.shape[0], 400)
        # y=self._get_fit_vals(param,x)
        # self.fit_track=np.vstack([x,y]).T
        model = make_pipeline(PolynomialFeatures(4), Ridge())
        model.fit(self.track[:,0].reshape(-1,1),self.track[:,1].reshape(-1,1))
        x=np.array([i for i in range(self.show_img.shape[0])]).reshape(-1,1)
        y=model.predict(x)
        self.fit_track=np.hstack([x,y])
        if plt_flag:
            plt.figure()
            plt.xlim([0,self.show_img.shape[0]])
            plt.ylim([0,self.show_img.shape[1]])
            plt.gca().set_aspect(1)
            plt.plot(self.track[:,0],self.track[:,1],'b')
            plt.plot(x,y,'r')
            # plt.show()

    def _draw_fit_track(self,img):
        track=self.fit_track.astype(int)
        for i in range(1,len(track)):#img:(h,w)
            cv2.line(img,(track[i][1],track[i][0]),\
                        (track[i-1][1],track[i-1][0]),(255,0,0),3)#w,h
        return img

    def _get_turning_and_diff_point(self,plt_flag=False):
        '''求拐点,并且求切线'''
        diff_1=self.fit_track[1:,1]-self.fit_track[:-1,1]
        self.turning_point=[]
        self.turning_point.append([0,self.fit_track[0,1]])
        for i in range(len(diff_1)-1):
            if diff_1[i+1]*diff_1[i]<0:
                self.turning_point.append([i,self.fit_track[i,1]])
        self.turning_point.append([len(self.track)-1,self.fit_track[len(self.track)-1][1]])

        self.max_abs_diff_points=[]
        for i in range(len(self.turning_point)-1):
            this_slice=diff_1[self.turning_point[i][0]:self.turning_point[i+1][0]]
            max_abs_diff_idx=np.argmax(abs(this_slice))+self.turning_point[i][0]
            self.max_abs_diff_points.append([max_abs_diff_idx,self.fit_track[max_abs_diff_idx,1]])#x,y,k
        self.intersection_of_max_abs_diff_points=[]
        for i in range(len(self.max_abs_diff_points)-1):
            x1=self.max_abs_diff_points[i][0]
            y1=self.max_abs_diff_points[i][1]
            k1=diff_1[x1]
            x2=self.max_abs_diff_points[i+1][0]
            y2=self.max_abs_diff_points[i+1][1]
            k2=diff_1[x2]
            intersection_x=(y2-k2*x2-y1+k1*x1)/(k1-k2)
            intersection_y=(k1*y2-k1*k2*x2-k2*y1+k1*k2*x1)/(k1-k2)
            self.intersection_of_max_abs_diff_points.append([intersection_x,intersection_y])           

        if plt_flag:
            for x,y in self.turning_point:
                plt.plot(x,y,'r*')
            # for x,y in self.max_abs_diff_points:
            #     plt.plot(x,y,'b^')
            # for x,y in self.intersection_of_max_abs_diff_points:
            #     plt.plot(x,y,'b^')
        
    def _draw_turning_and_diff_point(self,img):
        for x,y in self.turning_point:
            cv2.circle(img,(int(y),int(x)),5,(0,0,255),-1)
        # for x,y in self.max_abs_diff_points:
        #     cv2.circle(img,(int(y),int(x)),5,(255,0,0),-1)
        # for x,y in self.intersection_of_max_abs_diff_points:
        #     cv2.circle(img,(int(y),int(x)),5,(255,0,0),-1)
        return img

    def _get_theta_by_turnning_points(self):
        self.theta_by_turnning_points=[]
        for i in range(len(self.turning_point)-2):
            a=np.array([self.turning_point[i][0]-self.turning_point[i+1][0],\
                        self.turning_point[i][1]-self.turning_point[i+1][1]])
            b=np.array([self.turning_point[i+2][0]-self.turning_point[i+1][0],\
                        self.turning_point[i+2][1]-self.turning_point[i+1][1]])
            self.theta_by_turnning_points.append(np.arccos(a.dot(b)/(np.sqrt(a.dot(a)) * np.sqrt(b.dot(b))))*180/np.pi)
    
    def _draw_theta_from_turnning_points(self,img):
        for i in range(len(self.turning_point)-1):
            cv2.line(img,(int(self.turning_point[i][1]),int(self.turning_point[i][0])),\
                (int(self.turning_point[i+1][1]),int(self.turning_point[i+1][0])),(0,0,255))
        for i in range(len(self.turning_point)-2):
            cv2.putText(img,'%.2f'%(self.theta_by_turnning_points[i]),(int(self.turning_point[i+1][1]),int(self.turning_point[i+1][0])),\
                cv2.FONT_HERSHEY_COMPLEX,6e-1,(0,0,255),2)
        return img

    def _get_theta_by_tangent(self):
        self.theta_by_tangent=[]
        for i in range(len(self.max_abs_diff_points)-1):
            a=np.array([self.max_abs_diff_points[i][0]-self.intersection_of_max_abs_diff_points[i][0],\
                self.max_abs_diff_points[i][1]-self.intersection_of_max_abs_diff_points[i][1]])
            b=np.array([self.max_abs_diff_points[i+1][0]-self.intersection_of_max_abs_diff_points[i][0],\
                self.max_abs_diff_points[i+1][1]-self.intersection_of_max_abs_diff_points[i][1]])
            self.theta_by_tangent.append(np.arccos(a.dot(b)/(np.sqrt(a.dot(a)) * np.sqrt(b.dot(b))))*180/np.pi)

    def _draw_theta_from_tangent(self,img):
        for i in range(len(self.max_abs_diff_points)-1):
            cv2.line(img,(int(self.max_abs_diff_points[i][1]),int(self.max_abs_diff_points[i][0])),\
                (int(self.intersection_of_max_abs_diff_points[i][1]),int(self.intersection_of_max_abs_diff_points[i][0])),\
                    (0,0,255))
            cv2.line(img,(int(self.max_abs_diff_points[i+1][1]),int(self.max_abs_diff_points[i+1][0])),\
                (int(self.intersection_of_max_abs_diff_points[i][1]),int(self.intersection_of_max_abs_diff_points[i][0])),\
                    (0,0,255))
        for i in range(len(self.intersection_of_max_abs_diff_points)):
            cv2.putText(img,'%.2f'%(self.theta_by_tangent[i]),(int(self.intersection_of_max_abs_diff_points[i][1]),\
                int(self.intersection_of_max_abs_diff_points[i][0])),\
                cv2.FONT_HERSHEY_COMPLEX,6e-1,(0,0,255),2)
        return img

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