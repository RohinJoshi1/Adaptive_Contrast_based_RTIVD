from math import log10, sqrt
from skimage.metrics import structural_similarity
import cv2
import numpy as np

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def SSIM(original, processed):

    # Convert images to grayscale
    before_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between two images
    (score, diff) = structural_similarity(before_gray, after_gray, full=True)
    return score

    # The diff image contains the actual image differences between the two images
    # and is represented as a floating point data type in the range [0,1]
    # so we must convert the array to 8-bit unsigned integers in the range
    # [0,255] before we can use it with OpenCV

def main():
     im_name = "3.png"
     approach = "our_approach"
     original = cv2.imread(f"/Users/rohinjoshi/Work/codes/MajorProject/playground/test/RTVD/final/input/{im_name}")
     processed = cv2.imread(f"/Users/rohinjoshi/Work/codes/MajorProject/playground/test/RTVD/final/output/{approach}/{im_name}", 1)
     value = PSNR(original, processed)
     print(f"PSNR value is {value} dB")
     value = SSIM(original=original,processed=processed)
     print(f"SSIM value is {value}")
if __name__ == "__main__":
    main()
