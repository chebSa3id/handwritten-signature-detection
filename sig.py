import numpy as np
import cv2
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
#new
#new1
#new2
#qqqq
#image1
#ramy
image = cv2.imread('new.jpg')
image=cv2.resize(image,(800,800))
print(image.shape)
result = image.copy()
cv2.imshow('qqq', image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([20, 38, 0])
upper = np.array([145, 255, 255])
mask = cv2.inRange(image, lower, upper)



kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

boxes = []
for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    boxes.append([x,y, x+w,y+h])

boxes = np.asarray(boxes)
left = np.min(boxes[:,0])
top = np.min(boxes[:,1])
right = np.max(boxes[:,2])
bottom = np.max(boxes[:,3])


cv2.rectangle(result, (left,top), (right,bottom), (36, 255, 12), 2)

cv2.imshow('result', result)
cv2.imwrite('result.png', result)
cv2.waitKey()
