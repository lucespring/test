# -*- coding: UTF-8 -*-
import cv2
import selectivesearch
import numpy as np
from matplotlib import pyplot as plt

class OpencvFilters:
    def __init__(self):
        self.filter = []

    def set_filter_gradient_3x3(self):
        self.filter = np.array([
            [ 1,  1,  1],
            [ 0,  0,  0],
            [-1, -1, -1]
            ], np.float32)
        return self.filter

    def set_filter_gradient_5x5(self):
        self.filter = np.array([
            [ 5,  5,  5,  5,  5],
            [ 3,  3,  3,  3,  3],
            [ 0,  0,  0,  0,  0],
            [-3, -3, -3, -3, -3],
            [-5, -5, -5, -5, -5]
            ], np.float32)
        return self.filter

    def set_filter_high_pass(self):
        self.filter = np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
            ], np.float32)
        return self.filter

    def set_filter_laplacian_3x3(self):
        self.filter = np.array([
            [1,  1, 1],
            [1, -8, 1],
            [1,  1, 1]
            ], np.float32)
        return self.filter

    def set_filter_laplacian_5x5(self):
        self.filter = np.array([
            [-1, -3, -4, -3, -1],
            [-3,  0,  6,  0, -3],
            [-4,  6, 20,  6, -4],
            [-3,  0,  6,  0, -3],
            [-1, -3, -4, -3, -1]
            ], np.float32)
        return self.filter

    def set_filter_gaussian(self):
        self.filter = np.array([
            [1,  2, 1],
            [2,  4, 2],
            [1,  2, 1]
            ], np.float32) / 16
        return self.filter
        # img_gaussian = cv2.filter2D(img_gray, -1, kernel_gaussian)

    def set_filter_original_1(self):
        self.filter = np.array([
            [ 0,   1,  0],
            [ 1,  -4,  1],
            [ 0,   1,  0]
            ], np.float32) / 1
        return self.filter

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(1)
#width  = 320
#width  = 480
width  = 1920
#height = 180
#height = 270
height = 1080

def main():
    my_OpencvFilters = OpencvFilters()
    # my_filter = my_OpencvFilters.set_filter_laplacian_3x3()
    # my_filter = my_OpencvFilters.set_filter_gaussian()
    my_filter = my_OpencvFilters.set_filter_original_1()

    while(True):

        # 動画ストリームからフレームを取得
        ret, img_in = cap.read()
        # カメラ画像をリサイズ
        # img_in = cv2.resize(img_in,(width,height))

        img_out = img_in
        img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)
        _, img_out = cv2.threshold(img_out, 128, 255, cv2.THRESH_BINARY)
        kernel_open = np.ones((2, 2), np.uint8)
        kernel_close = np.ones((2, 2), np.uint8)
        kernel_gradient = np.ones((2, 2), np.uint8)
        kernel_tophat = np.ones((2, 2), np.uint8)
        kernel_blackhat = np.ones((2, 2), np.uint8)
        # img_out = cv2.morphologyEx(img_out, cv2.MORPH_OPEN, kernel_open)
        # img_out = cv2.morphologyEx(img_out, cv2.MORPH_CLOSE, kernel_close)
        # img_out = cv2.morphologyEx(img_out, cv2.MORPH_GRADIENT, kernel_gradient)
        # img_out = cv2.morphologyEx(img_out, cv2.MORPH_TOPHAT, kernel_tophat)
        img_out = cv2.morphologyEx(img_out, cv2.MORPH_BLACKHAT, kernel_blackhat)
        # img_out = cv2.Canny(img_out,10,200)
        # img_out = cv2.cv2.GaussianBlur(img_out, (15, 15), 10)
        # img_out = cv2.filter2D(img_out, -1, my_filter)

#        plt.subplot(121),plt.imshow(img,cmap = 'gray')
#        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#        plt.subplot(122),plt.imshow(edges,cmap = 'gray')
#        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
#        plt.show()

        cv2.namedWindow("input image", cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL)
        cv2.namedWindow("output image", cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL)
        cv2.imshow("input image", img_in)
        cv2.imshow("output image", img_out)

        # escを押したら終了。
        if cv2.waitKey(1) == 27:
            break

    #終了
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
