# %%
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import os

os.environ["QT_QPA_PLATFORM"] = "xcb"

# %%
left_img: np.ndarray = cv.imread("Resources/1_l.png", cv.IMREAD_GRAYSCALE)
right_img: np.ndarray = cv.imread("Resources/1_r.png", cv.IMREAD_GRAYSCALE)

# left_img: np.ndarray = cv.imread("Resources/2_l.png", cv.IMREAD_GRAYSCALE)
# right_img: np.ndarray = cv.imread("Resources/2_r.png", cv.IMREAD_GRAYSCALE)

# left_img: np.ndarray = cv.imread("Resources/3_l.png", cv.IMREAD_GRAYSCALE)
# right_img: np.ndarray = cv.imread("Resources/3_r.png", cv.IMREAD_GRAYSCALE)      

# %%
cv.imshow("Left", left_img)
cv.imshow("right", right_img)
cv.waitKey(0)

stereo: cv.StereoBM = cv.StereoBM_create(numDisparities=0, blockSize=21)
depth: np.ndarray = stereo.compute(left_img, right_img)

print(depth)
plt.imshow(depth)
plt.axis("off")
plt.show()
plt.imsave("Resources/depth_map_1.png", depth)

# %%
