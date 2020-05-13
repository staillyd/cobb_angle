import os
from sys import argv
import ImageProcessing
from PIL import ImageEnhance

if __name__ == '__main__':

    files = argv[1:]

    qntt_files = len(files)

    for i in range(qntt_files):

        img = ImageProcessing.Image.open(files[i])
        img = ImageProcessing.verify_bw(img)#灰度化
        img_origin = img.copy()
        img = ImageProcessing.isolate_vertebral_column(img)
        img = ImageProcessing.crop_image(img)#roi
        img = ImageProcessing.histogram_equalize(img)#对比度
        img = ImageProcessing.isolate_scoliosis(img)#行像素置0
        img = ImageProcessing.median_filter(img, 2)#中值滤波
        img = ImageProcessing.histogram_equalize(img)
        img = ImageProcessing.median_filter(img, 4)
        img = ImageProcessing.sobel_filter(img)#边缘检测
        img = ImageProcessing.median_filter(img, 8)
        img = ImageProcessing.isolate_scoliosis(img)
        img = ImageProcessing.binary(img)#二值化
        img = ImageProcessing.trace_line(img)#画轨迹
        img = ImageProcessing.merge_images(img_origin, img)
        img.save("./results/{0}".format(files[i].split("/")[-1]))
        img.show()
