import cv2
import cv2 as cv
import numpy as np
import random as rng
from matplotlib import pyplot as plt
import pytesseract
import tensorflow as tf

img = cv2.imread('Meter_1.png')

res = cv.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
# res = cv.equalizeHist(res)
res = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
# res = cv.bilateralFilter(res,9,75,75)
res = cv.GaussianBlur(res, (9, 9), 0)
res = cv.medianBlur(res, 5)
# res = cv.Canny(res,100,200)
rectKern = cv.getStructuringElement(cv.MORPH_RECT, (13, 5))

res = cv.adaptiveThreshold(res, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, \
                           cv.THRESH_BINARY, 11, 2)

res = cv.morphologyEx(res, cv.MORPH_BLACKHAT, rectKern)

kernel = np.ones((3, 3), np.uint8)
res = cv.morphologyEx(res, cv.MORPH_OPEN, kernel)
res = cv.dilate(res, kernel, iterations=1)

#
contours, hierarchy = cv.findContours(res, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Approximate contours to polygons + get bounding rects and circles
contours_poly = [None] * len(contours)
boundRect = [None] * len(contours)
centers = [None] * len(contours)
radius = [None] * len(contours)
goodbbox = []

# rows,cols = img.shape[:2]
# [vx,vy,x,y] = cv.fitLine(contours, cv.DIST_L2,0,0.01,0.01)
# lefty = int((-x*vy/vx) + y)
# righty = int(((cols-x)*vy/vx)+y)
# cv.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)
def rectContains(rect,pt):

    res = rect[0] < pt[0] < rect[0]+rect[2] and rect[1] < pt[1] < rect[1]+rect[3]
    return res

for i, c in enumerate(contours):
    contours_poly[i] = cv.approxPolyDP(c, 3, True)
    boundRect[i] = cv.boundingRect(contours_poly[i])

for i, c in enumerate(boundRect):
    area = boundRect[i][2] * boundRect[i][3]

    width = boundRect[i][2]
    length = boundRect[i][3]


    if area > 350 and 0.04 * img.shape[1] < boundRect[i][1] < 0.7 * img.shape[1]:
        goodbbox.append(boundRect[i])

goodbbox.sort()


drawing = np.zeros((res.shape[0], res.shape[1], 3), dtype=np.uint8)
images = []
# Draw polygonal contour + bonding rects + circles
for i in range(len(contours)):
    color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
    cv.drawContours(drawing, contours_poly, i, color)

for i in range(len(goodbbox)):
    color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
    cv.rectangle(drawing, (int(goodbbox[i][0] - 5), int(goodbbox[i][1]) - 5),
                 (int(goodbbox[i][0] + goodbbox[i][2] + 5), int(goodbbox[i][1] + goodbbox[i][3]) + 5), color, 2)
    images.append(img[int(goodbbox[i][1]):int(goodbbox[i][1] + goodbbox[i][3]),
                  int(goodbbox[i][0]):int(goodbbox[i][0] + goodbbox[i][2])])

cv.imshow("org", img)
cv.imshow("res", res)
cv.imshow("draw", drawing)
cv.waitKey()

model = tf.keras.models.load_model('saved_model', custom_objects={'softmax_v2': tf.nn.softmax})


def preProcessing(img):
    res = cv.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    res = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
    res = cv.GaussianBlur(res, (9, 9), 0)
    res = cv.medianBlur(res, 5)
    rectKern = cv.getStructuringElement(cv.MORPH_RECT, (13, 5))
    res = cv.adaptiveThreshold(res, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)

    # res = cv.morphologyEx(res, cv.MORPH_BLACKHAT, rectKern)
    # kernel = np.ones((3, 3), np.uint8)
    # res = cv.morphologyEx(res, cv.MORPH_OPEN, kernel)
    # res = cv.dilate(res, kernel, iterations=2)

    res = res / 255
    return res


meter = ""
for i in range(len(images)):
    images[i] = cv.resize(images[i], (32, 32))
    # images[i] = cv.cvtColor(images[i], cv.COLOR_BGR2GRAY)
    # images[i] = images[i].reshape(1,32,32,1)
    cv.imshow("image", images[i])
    print(images[i].shape)
    cv.waitKey()
    img = cv.resize(images[i], (32, 32), interpolation=cv.INTER_AREA)
    img = np.asarray(img)
    img = preProcessing(img)
    cv.imshow("Processsed Image", img)
    img = img.reshape(1, 32, 32, 1)

    #### PREDICT
    classIndex = int(model.predict_classes(img))
    # print(classIndex)
    predictions = model.predict(img)
    # print(predictions)
    probVal = np.amax(predictions)
    print(classIndex, probVal)
    if (probVal > 0.99):
        meter = meter + str(classIndex)
    cv.waitKey()
    print(meter)
    cv2.waitKey()