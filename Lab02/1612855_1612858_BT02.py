import sys
import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt
from EdgeDetector import Sobel, Prewitt, Laplace, Canny

def show(name, srcImage, dstImage, Gx = None, Gy = None):
    if (name == "Prewitt" or name == "Sobel"):
        cv.imshow("Gradient by x", Gx)
        cv.imshow("Gradient by y", Gy)
    cv.imshow(name, dstImage)


if __name__ == "__main__":
    filename = sys.argv[1]
    operator = sys.argv[2]
    srcImage = cv.imread(filename, cv.IMREAD_GRAYSCALE)

    cv.imshow("Source image", srcImage)

    if operator == "--sobel":
        Gx, Gy, dstImage = Sobel(srcImage)
        show( "Sobel", srcImage, dstImage, Gx, Gy)
    elif operator == "--prewitt":
        Gx, Gy, dstImage = Sobel(srcImage)
        show("Prewitt", srcImage, dstImage, Gx, Gy)
    elif operator == "--laplace":
        dstImage = Laplace(srcImage)
        show("Laplace", srcImage, dstImage)
    elif operator == "--canny":
        dstImage, low, high = Canny(srcImage)
        show("Canny", srcImage, dstImage)
        blurImg = cv.GaussianBlur(srcImage, (5, 5), 1.4)
        CannyByOpenCV = cv.Canny(blurImg, low, high)
        cv.imshow("Canny by OpenCV", CannyByOpenCV)
    cv.waitKey(0)
    cv.destroyAllWindows()
