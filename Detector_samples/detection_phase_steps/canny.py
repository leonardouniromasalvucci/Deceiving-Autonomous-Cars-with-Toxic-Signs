import cv2 as cv

img = cv.imread('gray_img1.png',0)
edges = cv.Canny(img, 0, 0)
cv.imwrite("canny.png", edges)
