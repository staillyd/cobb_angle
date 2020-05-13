from random import randint
from PIL import Image, ImageDraw
import numpy as np

def merge_images(file1, file2):
    image1 = file1
    image2 = file2

    (width1, height1) = image1.size
    (width2, height2) = image2.size

    result_width = width1 + width2
    result_height = max(height1, height2)

    result = Image.new('L', (result_width, result_height))
    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(width1, 0))
    return result.copy()

def remove_transparency(img, bg_colour=(255, 255, 255)):
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        alpha = img.convert('RGBA').split()[-1]
        bg = Image.new("RGBA", img.size, bg_colour + (255,))
        bg.paste(img, mask=alpha)
        return bg
    else:
        return img

def verify_bw(img):
    img = remove_transparency(img)
    if Image.getmodebase(img.mode) == 'L':
        return img
    else:
        return convert2mono(img)

def convert2mono(img):#转化为灰度图
    pixels = img.load()      
    width, height = img.size      

    mono_pixels = []         

    for y in range(height):
        for x in range(width):
            if len(pixels[x, y]) == 4:
                r, g, b, a = pixels[x, y]
            else:
                r, g, b = pixels[x, y]

            p = int(r * 0.2126 + g * 0.7152 + b * 0.0722)
            mono_pixels.append(p)

    mono = Image.new('L', (width, height))
    mono.putdata(mono_pixels)
    return mono

def histogram_equalize(img):# 转换对比度
    ret_image = img
    ret_image_pixels = ret_image.load()
    width, height = img.size
    count_intensity = [0 for i in range(256)]#0~255像素的 统计个数

    for x in range(width):
        for y in range(height):
            count_intensity[ret_image_pixels[x, y]] += 1

    cdf = []#累计密度  cumsum

    for i in range(256):
        if i > 0:
            cdf.append(count_intensity[i] + cdf[i - 1])
        else:
            cdf.append(count_intensity[i])

    cdf_min = min(cdf)
    lookup_table = []

    for i in range(256):#(累计密度-cdf_min)/img_area*255
        lookup_table.append(
            int(float(cdf[i] - cdf_min) / (width * height - 1) * 255)
        )

    for x in range(width):#像素值转化为   (累计密度-cdf_min)/img_area*255
        for y in range(height):
            old = ret_image_pixels[x, y]
            ret_image_pixels[x, y] = int(lookup_table[old])

    return ret_image

def median_filter(img, intensity):#中值滤波,周围intensity/2个
    newimg = img.copy()
    width, height = img.size
    img = img.load()
    intensity = int(intensity/2)

    for y in range(height):
        for x in range(width):
            p = img[x, y]
            neighbors = []#周围的像素块
            for xx in range(x - intensity, x + intensity+1):
                for yy in range(y - intensity, y + intensity+1):
                    try:
                        neighbor = img[xx, yy]
                        neighbors.append(neighbor)
                    except:
                        continue

            nlen = len(neighbors)
            if nlen:
                bw = [neighbors[i] for i in range(nlen)]
                bw.sort()
                if nlen % 2:
                    p = bw[int(len(bw) / 2)]
                else:
                    p = (bw[int(len(bw) / 2)] +
                            bw[int(len(bw) / 2 - 1)]) / 2

                newimg.putpixel((x, y), int(p))
    return newimg.copy()

def binary(img):#二值化
    bw = np.asarray(img).copy()
    avg = np.average(bw)
    bw[bw <= avg] = 0    # Black
    bw[bw > avg] = 255  # White

    binary_img = Image.fromarray(bw)
    return binary_img

def crop_image(img, tolerance=0):
    img = np.asarray(img).copy()
    mask = img > tolerance

    return Image.fromarray(img[np.ix_(mask.any(1), mask.any(0))]).copy()

def sobel_filter(img):#边缘检测
    width, height = img.size
    filtered = img.copy()

    for x in range(1, width - 1):
        for y in range(1, height - 1):#左右像素点差值
            x_sobel = (img.getpixel((x + 1, y - 1))
                       + 2 * img.getpixel((x + 1, y))
                       + img.getpixel((x + 1, y + 1))) \
                - (img.getpixel((x - 1, y - 1))
                   + 2 * img.getpixel((x - 1, y))
                   + img.getpixel((x - 1, y + 1)))

            filtered.putpixel((x, y), int(abs(x_sobel / 4)))
    return filtered.copy()


def isolate_vertebral_column(img):
    image = img.copy()
    width, height = img.size 

    blank_columns = []

    cut_point = np.average(np.asarray(img).copy())#平均像素值

    for x in range(width):#blank_columns保存 列像素值>(平均像素值/1.6)个数  <=0.6*H的列索引
        vet_lines_w_color = []
        count_intensity = 0
        for y in range(height):
            if image.getpixel((x, y)) * 1.6 > cut_point:
                count_intensity += 1

        if count_intensity <= img.size[1] * 0.6:
            blank_columns.append(x)

    for i in range(len(blank_columns)):#将列索引相差0.2×W的列都置为0
        x = blank_columns[i]

        try:
            if (blank_columns[i + 1] - blank_columns[i]) <= image.size[0] * 0.2:
                for j in range(x, blank_columns[i + 1], 1):
                    for y in range(height):
                        image.putpixel((j, y), 0)
        except:
            pass

    return image.copy()


def isolate_scoliosis(img):
    image = img.copy()                
    width, height = img.size     

    median = 50

    for y in range(height):#像素<行像素的中位数/0.9 变为0
        vet_line = []
        for x in range(width):
            vet_line.append(image.getpixel((x, y)))
        median = np.median(vet_line)#行像素的中位数
        # import ipdb; ipdb.set_trace()
        for x in range(width):
            if image.getpixel((x, y)) * 0.9 <= median:
                image.putpixel((x, y), 0)
            # else:
            #     break

    for y in range(height - 1, 0, -1):#像素<行像素的中位数/0.9 变为0
        vet_line = []
        for x in range(width - 1, 0, -1):
            vet_line.append(image.getpixel((x, y)))
        median = np.median(vet_line)
        for x in range(width - 1, 0, -1):
            if image.getpixel((x, y)) * 0.9 <= median:
                image.putpixel((x, y), 0)
            # else:
            #     break

    return image.copy()

def trace_line(img):
    image = img.copy()             
    width, height = img.size
    ten_percent_height = int(height * 0.1)
    points_list = [i for i in range(ten_percent_height, height, ten_percent_height)]

    pixels_points = []

    for p in points_list:#寻找左右的白点，取中点
        vet_line = []
        left_side_pixel = int()
        right_side_pixel = int()

        for x in range(width):
            if image.getpixel((x, p)) >= 250:
                left_side_pixel = x
                break

        for x in range(width - 1, 0, -1):
            if image.getpixel((x, p)) >= 250:
                right_side_pixel = x
                break
        
        avg_pixel_point = int(((right_side_pixel - left_side_pixel) / 2) + left_side_pixel)
        pixels_points.append(avg_pixel_point)
        
    draw = ImageDraw.Draw(image)
    for i in range(len(points_list)):#画中线
        if i + 1 >= len(points_list) - 1: break
        draw.line((pixels_points[i], points_list[i], 
                    pixels_points[i + 1], points_list[i + 1]),
                fill=128, width=10)

    return image.copy()
