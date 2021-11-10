import cv2
import numpy

def clean_binary_image(img, elementSize=5):
    size = (elementSize, elementSize)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)

    out = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)

    return out

def valid_images(img1, img2):
    if not img1.shape == img2.shape:
        msg = """
        Input images should have the same dimensions.

        Received images with sizes {} and {} respectively.
        """
        raise ValueError(msg.format(img1.shape, img2.shape))

    return True

def convert_to_grey(src):
    if isinstance(src, str):
        src = cv2.imread(src, cv2.IMREAD_COLOR)
    return cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

def convert_to_gray(src):
    return convert_to_grey(src)

def convert_to_RGB(src):
    if isinstance(src, str):
        src = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    return cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)

def resize_static(img, width, height):
    return cv2.resize(img, (width, height))

def resize(img, width=None, height=None):
    h, w = img.shape[:2]
    inter_type = cv2.INTER_AREA

    if width is None and height is None:
        print("Please specify a height or width or both")
        return img
    elif width is None:
        return cv2.resize(img, (int (w * (height / float(h))), height), interpolation=inter_type)
    elif height is None:
        return cv2.resize(img, (width, int(h * (width / float(w)))), interpolation=inter_type)
    else:              
        return resize_static(img, width, height)
