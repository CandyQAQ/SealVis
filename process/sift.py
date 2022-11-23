import cv2

# img = cv2.imread('seal.jpg')
img = cv2.imread('seal_1.jpg')
img = cv2.resize(src=img, dsize=(224, 224))
sift = cv2.SIFT_create()
kp = sift.detect(img, None)
cv2.drawKeypoints(image=img, keypoints=kp, outImage=img, color=(255, 0, 0))

cv2.imwrite("sift.jpg", img)
