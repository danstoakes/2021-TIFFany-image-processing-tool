import cv2
import tempfile
import subprocess
import platform
from .utility import clean_binary_image, convert_to_grey, convert_to_RGB, resize

def _open_image_file(path):
    system = platform.system()
    if system == "Darwin":        # macOS
        subprocess.run(["open", path])
    elif system == "Windows":     # Windows
        subprocess.run(["start", path], shell=True)
    elif system == "Linux":       # Linux
        subprocess.run(["xdg-open", path])
    else:
        print(f"Unsupported OS: {system}. Image saved at {path}")

def _save_temp_image(img, prefix=None):
    # Create a NamedTemporaryFile without auto-delete, so we can open it later
    if prefix:
        # Generate a temp file with custom prefix
        tmp_dir = tempfile.gettempdir()
        # Use prefix + random suffix + .png
        temp_path = tempfile.mktemp(suffix=".png", prefix=prefix + "_", dir=tmp_dir)
    else:
        # Default random temp file
        tmp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp_path = tmp_file.name

    cv2.imwrite(temp_path, img)
    return temp_path

def show(src, width=None, height=None, filename_prefix=None):
    if isinstance(src, str):
        img = cv2.imread(src, 0)
    else:
        img = src

    if width is not None or height is not None:
        img = resize(img, width, height)

    temp_path = _save_temp_image(img, prefix=filename_prefix)
    _open_image_file(temp_path)

def display_window(msg, src, img):
    title = f"{msg}"

    temp_path = _save_temp_image(img, title)
    _open_image_file(temp_path)

# performs a Gaussin blur on an input image
def gaussian_blur(src, radius=5, display=False):
    im = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    if radius % 2 == 0:
        radius = radius + 1

    filterSize = (radius, radius)
    im = cv2.GaussianBlur(im, filterSize, cv2.BORDER_DEFAULT)

    if display:
        display_window("gaussian_blur", src, im)

    return im

# performs a median blur on an input image
def median_blur(src, radius=5, display=False):
    if isinstance(src, str):
        im = cv2.imread(src, cv2.IMREAD_COLOR)
    else:
        im = src
        
    if radius % 2 == 0:
        radius = radius + 1

    im = cv2.medianBlur(im, radius)

    if display:
        display_window("median_blur", src, im)

    return im

# calculates the foreground mask between two input images
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

# calculates the contours of the differences between two input images
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