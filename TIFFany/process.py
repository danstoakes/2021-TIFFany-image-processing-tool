import cv2
import numpy as np
from .utility import valid_images

def calc_mse(img1, img2):
    if isinstance(img1, str):
        img1 = cv2.imread(img1, cv2.IMREAD_COLOR)

    if isinstance(img2, str):
        img2 = cv2.imread(img2, cv2.IMREAD_COLOR)

    if valid_images(img1, img2):
        difference = img1.astype(np.float64) - img2.astype(np.float64)
        error = np.sum(difference ** 2)
        error /= float(img1.shape[0] * img1.shape[1])

        return round(error, 2)

    return -1

def mse(img1, compare):
    if isinstance(img1, str):
        img1 = cv2.imread(img1, cv2.IMREAD_COLOR)

    if isinstance(compare, str):
        return calc_mse(img1, compare)
    else:
        if len(compare) <= 0:
            raise ValueError("There should be at least one image to compare against.")
        
        mse_values = []
        for img in compare:
            mse_values.append(calc_mse(img1, img))

        return mse_values

def calc_ssim(img1, img2):
    if isinstance(img1, str):
        img1 = cv2.imread(img1, cv2.IMREAD_COLOR)

    if isinstance(img2, str):
        img2 = cv2.imread(img2, cv2.IMREAD_COLOR)

    if valid_images(img1, img2):
        C1 = (0.01 * 255) ** 2 # ensures stability when the denominator is 0
        C2 = (0.03 * 255) ** 2 # 255 for the dynamic range (255 pixels for 8 bits)

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        distribution = cv2.getGaussianKernel(11, 1.5) # constants
        window = np.outer(distribution, distribution.transpose())

        mu1 = cv2.filter2D(img1, -1, window)        
        mu2 = cv2.filter2D(img2, -1, window)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu12 = mu1 * mu2

        sigma1_sq = cv2.filter2D(img1**2, -1, window) - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window) - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window) - mu12

        ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return round(ssim_map.mean(), 2)
        

def ssim(img1, compare):
    if isinstance(img1, str):
        img1 = cv2.imread(img1, cv2.IMREAD_COLOR)

    if isinstance(compare, str):
        return calc_ssim(img1, compare)
    else:
        if len(compare) <= 0:
            raise ValueError("There should be at least one image to compare against.")

        ssim_values = []
        for img in compare:
            ssim_values.append(calc_ssim(img1, img))

        return ssim_values
