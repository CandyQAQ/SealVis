from PIL import Image
import numpy as np
import cv2

img2 = cv2.imread("D:\Research\Project\\back\data\Seals\\1(4)_0.jpg")
h, w, ch = img2.shape
for y in range(w):
  for x in range(h):
    print(img2[x][y])
    if img2[x][y][0]<=90 and img2[x][y][1]<=90 and img2[x][y][2]<=90:
      img2[x, y] = (255, 255, 255)
cv2.imwrite("test.jpg", img2)