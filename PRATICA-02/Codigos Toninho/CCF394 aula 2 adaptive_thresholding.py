"""
Instalando skimage

# Update pip
python -m pip install -U pip
# Install scikit-image
python -m pip install -U scikit-image

"""


from skimage.filters import threshold_local
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
# load the image and convert to grayscale and blur it slightly
img = cv2.imread("SUDOKU.PNG")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("entrada",gray)


hist,bins = np.histogram(gray.ravel(),256,[0,256])
plt.plot(hist)
plt.title("histograma usando Numpy")
plt.show()


# apply adaptive thresholding with OpenCV
neighbourhood_size = 25
constant_c = 15
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                               neighbourhood_size, constant_c)
cv2.imshow("OpenCV Mean Threshold", thresh)

# apply adaptive thresholding with scikit-image
neighbourhood_size = 29
constant_c = 5
threshold_value = threshold_local(gray, neighbourhood_size, offset=constant_c)
# np.uint8 devolve a matriz para a faixa de 8 bits
thresh = (gray < threshold_value).astype(np.uint8) * 255
cv2.imshow("Scikit-image Mean Threshold", thresh)
cv2.waitKey(0)
