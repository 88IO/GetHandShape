#!~/miniconda/bin/python
# coding: utf-8
# Author: 881O
# 2018-7-15

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
import sys
import getopt


def version():
    print("GetHandShape .ver 0.0.1\n")


def usage():
    print("python handshape.py --input(-i)     image_file[str]")
    print("                    --threshold(-t) thread_value[int]\n")


def blankImage(h, w):
    return np.zeros((h, w, 3), np.uint8)


def getMaxArea(image):
    img, contours, hierarchy = cv2.findContours(
        image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area, max_area_contour = 0, -1
    for contour in contours:
        area = cv2.contourArea(contour)
        if (max_area < area):
            max_area = area
            max_area_contour = contour
    return max_area_contour


def getCenterPosition(max_area_contour):
    M = cv2.moments(max_area_contour)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def sortList(max_area_contour, h, w):
    step_x = w / 10
    step_y = h / 10
    tmp_x, tmp_y = max_area_contour[0][0]
    for (i, point) in enumerate(max_area_contour):
        if (abs(point[0][0] - tmp_x) >= step_x) or (abs(point[0][1] - tmp_y) >= step_y):
            return np.vstack((max_area_contour[i:len(max_area_contour)], max_area_contour[0:i]))
        tmp_x, tmp_y = point[0]
    return max_area_contour


def getLocalMaximum(y):
    max_order, max_num = 100, 0
    while (max_num != 5):
        maxId = signal.argrelmax(y, order=max_order)
        max_num = len(maxId[0])
        if (max_num < 5) or (max_num > 10):
            break
        else:
            max_order += 100
    return maxId


def getLocalMinimum(y):
    min_order, min_num = 100, 0
    while (min_num != 6):
        minId = signal.argrelmin(y, order=min_order)
        min_num = len(minId[0])
        if (min_num < 6) or (min_num > 12):
            break
        else:
            min_order += 100
    return tuple([np.delete(minId[0], [0, 5])])


def getImages(img_threshold, img_area, img_hand):
    cv2.imwrite("result/img_threshold.png", img_threshold)
    cv2.imwrite("result/img_area.png", img_area)
    cv2.imwrite("result/img_hand.png", img_hand)


def drawFigure(cx, cy, h, w, x, y, maxId, minId, img_src, img_threshold, img_area, img_hand):
    fig = plt.figure()

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(img_src)
    ax1.set_title("img_src")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(img_area)
    ax2.set_title("img_area")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(x, y)
    ax3.plot(x[maxId], y[maxId], "ro")
    ax3.plot(x[minId], y[minId], "bo")
    ax3.set_title("distance from center")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.imshow(img_hand)
    ax4.set_title("Fingertips & Roots")

    fig.savefig("result/plot.png", dpi=300)
    plt.show()


def main(img_path, threshold):
    img_src = img_hand = cv2.imread(img_path)
    height, width, channel = img_src.shape

    img_area = blankImage(height, width)

    img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    ret, img_threshold = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)

    max_area_contour = getMaxArea(img_threshold)
    cx, cy = getCenterPosition(max_area_contour)
    cv2.drawContours(img_area, max_area_contour, -1, (255, 255, 255), 5)

    cv2.circle(img_area, (cx, cy), 10, (0, 255, 255), -1)

    img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
    img_area = cv2.cvtColor(img_area, cv2.COLOR_BGR2RGB)

    max_area_contour = sortList(max_area_contour, height, width)

    x = np.arange(len(max_area_contour))
    y = np.array([np.sqrt(pow(cx - position[0][0], 2) + pow(cx - position[0][1], 2)) for position in max_area_contour])

    maxId = getLocalMaximum(y)
    minId = getLocalMinimum(y)

    [cv2.line(img_hand, (cx, cy), tuple(max_area_contour[max][0]), (0, 0, 255), 10) for max in maxId[0]]
    [cv2.line(img_hand, (cx, cy), tuple(max_area_contour[min][0]), (255, 0, 0), 10) for min in minId[0]]

    getImages(img_threshold, img_area, img_hand)

    img_hand = cv2.cvtColor(img_hand, cv2.COLOR_BGR2RGB)

    drawFigure(cx, cy, height, width, x, y, maxId, minId, img_src, img_threshold, img_area, img_hand)


if __name__ == "__main__":
    image_file = None
    threshold = 100

    try:
        opts, args = getopt.getopt(sys.argv[1:], "i:t:hv", ["input=", "threshold=", "help", "version"])
    except getopt.GetoptError:
        usage()
        sys.exit()

    for o, a in opts:
        if o in ("-v", "--version"):
            version()
            sys.exit()
        if o in ("-h", "--help"):
            usage()
            sys.exit()
        if o in ("-i", "--input"):
            image_file = a
        if o in ("-t", "--threshold"):
            threshold = int(a)

    if (image_file is None) or (not os.path.isfile(image_file)):
        print("Please set input file.\n")
        usage()
        sys.exit()

    if not os.path.isdir("result"):
        os.mkdir("result")

    main(image_file, threshold=threshold)
