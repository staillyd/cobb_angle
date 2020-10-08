from PyQt5.QtWidgets import QDialog,QFileDialog,QMessageBox
from Ui_cobb_angle import Ui_Dialog
from main import Cobb_Angle
from PyQt5.QtGui import QPixmap
import cv2

class Cobb_Angle_Dialog(QDialog):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.ui=Ui_Dialog()
        self.ui.setupUi(self)
        self.cobb_angle=Cobb_Angle()

    def on_choose_file(self):
        file_abs_path,file_filter=QFileDialog.getOpenFileName(self,"打开文件",'./',"所有文件(*.*)")
        if file_abs_path.split('.')[-1] not in ['jpg','jpeg','bmp','png']:
            QMessageBox.warning(self,'请选择图片','请选择图片')
            return
        self.ui.file_msg.setPlainText(file_abs_path)
        self.cobb_angle.process_for_ui(file_abs_path)
        self.show_imgs()
        show_text='总共%d个角度:\n'%len(self.cobb_angle.theta_by_turnning_points)
        for theta in self.cobb_angle.theta_by_turnning_points:
            show_text+='%f\n'%(180-theta)
        self.ui.show_text.setPlainText(show_text)
        key=cv2.waitKey()& 0xFF==ord('q')

    def show_imgs(self):
        cv2.namedWindow('origin',cv2.WINDOW_KEEPRATIO)
        cv2.imshow('origin',cv2.imread(self.cobb_angle.tmp_dir+'origin.jpg'))

        origin_pix=QPixmap(self.cobb_angle.tmp_dir+'origin.jpg')
        show_origin_pix=origin_pix.scaledToHeight(self.ui.origin_img.height())
        self.ui.origin_img.setPixmap(show_origin_pix)

        cv2.namedWindow('result',cv2.WINDOW_KEEPRATIO)
        cv2.imshow('result',cv2.imread(self.cobb_angle.tmp_dir+'result.jpg'))

        result_pix=QPixmap(self.cobb_angle.tmp_dir+'result.jpg')
        show_result_pix=result_pix.scaledToHeight(self.ui.result_image.height())
        self.ui.result_image.setPixmap(show_result_pix)