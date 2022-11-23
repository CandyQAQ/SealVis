import cv2


# img = cv2.imread('extracted.jpg')
img = cv2.imread('extracted_1.jpg')
img = cv2.resize(src=img, dsize=(224, 224))
w, h, c = img.shape
result = img.copy()
# imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(imgray, 127, 255, 0)
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img, contours, -1, (255, 255, 0), 3)

for i in range(h):
    for j in range(w):
        if img[i][j][0] < 127:
            cv2.circle(result, (i, j), 1, (255, 0, 0), 2)
            break
    for j in range(w-1, 0, -1):
        if img[i][j][0] < 127:
            cv2.circle(result, (i, j), 1, (255, 0, 0), 2)
            break

for i in range(w):
    for j in range(h):
        if img[j][i][0] < 127:
            cv2.circle(result, (j, i), 1, (255, 0, 0), 2)
            break
    for j in range(h-1, 0, -1):
        if img[j][i][0] < 127:
            cv2.circle(result, (j, i), 1, (255, 0, 0), 2)
            break

cv2.imwrite("outline0.jpg", result)

for i in range(h):
    for j in range(w):
        if result[i][j][1] == 0:
            cv2.circle(img, (i, j), 1, (255, 0, 0), 2)
            break
    for j in range(w-1, 0, -1):
        if result[i][j][1] == 0:
            cv2.circle(img, (i, j), 1, (255, 0, 0), 2)
            break

for i in range(w):
    for j in range(h):
        if result[j][i][1] == 0:
            cv2.circle(img, (j, i), 1, (255, 0, 0), 2)
            break
    for j in range(h-1, 0, -1):
        if result[j][i][1] == 0:
            cv2.circle(img, (j, i), 1, (255, 0, 0), 2)
            break

cv2.imwrite("outline.jpg", img)
