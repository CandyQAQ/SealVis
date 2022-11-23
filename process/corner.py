import cv2
import numpy as np


# img = cv2.imread('extracted.jpg')
img = cv2.imread('extracted_1.jpg')
img = cv2.resize(src=img, dsize=(224, 224))
gray = cv2.cvtColor(src=img, code=cv2.COLOR_RGB2GRAY)
tomasiCorners = cv2.goodFeaturesToTrack(image=gray, maxCorners=1000, qualityLevel=0.01, minDistance=10)
tomasiCorners = np.int0(tomasiCorners)
for corner in tomasiCorners:
    x, y = corner.ravel()
    cv2.circle(img=img, center=(x, y), radius=3, color=(255, 0, 0), thickness=-1)

cv2.imwrite("corner.jpg", img)
