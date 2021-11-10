import cv2
import numpy

def clean_binary_image(im, elementSize=5):
    size = (elementSize, elementSize)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)

    out = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)

    return out

def valid_images(img1, img2):
    if not img1.shape == img2.shape:
        print(img1.shape)
        print(img2.shape)
        h1, w1, c1 = img1.shape
        h2, w2, c2 = img2.shape
        
        msg = """
        Input images should have the same dimensions.

        Received images with ({h1}, {w1}, {c1}) and ({h2}, {w2}, {c2}) respectively.
        """
        raise ValueError(msg.format(h1, w1, c1, h2, w2, c2))

    return True

def clean_binary_image(im, elementSize=5):
    size = (elementSize, elementSize)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)

    out = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)

    return out

def convert_to_grey(src):
    im = cv2.imread(src, cv2.IMREAD_COLOR)
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

def convert_to_gray(src):
    return convert_to_grey(src)

def convert_to_RGB(src):
    im = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    return cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

def convert_to_RGB(im):
    return cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

def resize_static(im, width, height):
    return cv2.resize(im, (width, height))

def resize(im, width=None, height=None):
    h, w = im.shape[:2]
    inter_type = cv2.INTER_AREA

    if width is None and height is None:
        print("Please specify a height or width or both")
        return im
    elif width is None:
        return cv2.resize(im, (int (w * (height / float(h))), height), interpolation=inter_type)
    elif height is None:
        return cv2.resize(im, (width, int(h * (width / float(w)))), interpolation=inter_type)
    else:              
        return resize_static(im, width, height)
