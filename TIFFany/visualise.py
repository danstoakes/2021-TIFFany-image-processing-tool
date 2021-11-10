import cv2
import numpy
from .utility import clean_binary_image, convert_to_grey, convert_to_RGB, resize

def show(src, width=None, height=None):
    if isinstance(src, str):
        if width != None or height != None:
            cv2.imshow(src, resize(cv2.imread(src, 0), width, height))
        else:
            cv2.imshow(src, cv2.imread(src, 0))
    else:
        if width != None or height != None:
            cv2.imshow("Image", resize(src, width, height))
        else:
            cv2.imshow("Image", src)

def display_window(msg, src, img):
    title = "{msg} [{img_src}]"
    cv2.imshow(title.format(msg = msg, img_src = src), img)

def gaussian_blur(src, radius=5, display=False):
    im = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    if radius % 2 == 0:
        radius = radius + 1

    filterSize = (radius, radius)
    im = cv2.GaussianBlur(im, filterSize, cv2.BORDER_DEFAULT)

    if display:
        display_window("Gaussian blur", src, im)

    return im

def median_blur(src, radius=5, display=False):
    if isinstance(src, str):
        im = cv2.imread(src, cv2.IMREAD_COLOR)
    else:
        im = src
        
    if radius % 2 == 0:
        radius = radius + 1

    im = cv2.medianBlur(im, radius)

    if display:
        display_window("Median blur", src, im)

    return im

def foreground_mask(src1, src2, threshold=128):
    # threshold is for colour values, i.e. 0 to 255
    # if greater than 128 (for example), set to 1, else 0
    im1Grey = convert_to_grey(src1)
    im1Grey = median_blur(im1Grey, 5)
    im1Grey = im1Grey.astype("float32")

    grey = convert_to_grey(src2)
    grey = median_blur(grey, 5) # blur radius 5
    grey = grey.astype("float32")

    foreground = im1Grey - grey
    foreground = abs(foreground)
    foreground = cv2.normalize(foreground, foreground, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    _, foregroundMask = cv2.threshold(foreground, threshold, 255, cv2.THRESH_BINARY)
    # elementSize dictates the size of the rectangle to be detected
    foregroundMask = clean_binary_image(foregroundMask, elementSize=5)

    return foregroundMask

def contours(src1, src2, threshold=120):
    img = foreground_mask(src1, src2)
    contours, heirarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    clean = convert_to_RGB(img)

    mu = []
    for i in range(len(contours)):
        mu.append(cv2.moments(contours[i]))

    mc = []
    for i in range(len(contours)):
        x = int(mu[i]["m10"]) / int(mu[i]["m00"])
        y = int(mu[i]["m01"]) / int(mu[i]["m00"])
        mc.append((x, y))

    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > threshold:
            cv2.drawContours(clean, contours, i, (0, 255, 0), 0)

    return clean
