import numpy as np
import cv2

mask = cv2.imread("./draw.png")
print(mask.shape)
mask = mask[:, :, 0]
mask = (mask == 255) * 1
print(mask.shape)
np.save("draw.npy", mask)