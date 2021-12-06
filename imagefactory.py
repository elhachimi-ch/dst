import cv2
import imutils
import imutils.perspective
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import threshold_local

class ImageFactory:
    def __init__(self, img_link=None, url=False):
        self.__img = None
        if url:
            self.__img = imutils.url_to_image(img_link)
        else:
            self.__img = cv2.imread(img_link)

    def set_image(self, img):
        self.__img = img

    def get_image(self):
        return self.__img

    def reshape(self, new_shape_tuple):
        self.set_image(np.reshape(self.get_image(), new_shape_tuple))

    def show(self):
        plt.figure("Image")
        plt.imshow(imutils.opencv2matplotlib(self.__img))
        plt.show()

    def resize(self, new_size_tuple, auto=False):
        # (height, width)
        if auto:
            self.__img = imutils.resize(self.__img, width=new_size_tuple[1])
        else:
            self.__img = cv2.resize(self.__img, new_size_tuple)

    def crop(self, top_left_tuple=(0, 0), bottom_right_tuple=(50, 50)):
        self.set_image(self.__img[top_left_tuple[1]:bottom_right_tuple[1], top_left_tuple[0]:bottom_right_tuple[0]])

    def to_gray_scale(self):
        self.set_image(cv2.cvtColor(self.__img, cv2.COLOR_BGR2GRAY))

    def rotat_image(self, angle):
        self.set_image(imutils.rotate(self.__img, angle))

    def find_contour(self, seuil=75, auto_threshold=True, all_points=False, auto_canny=True):
        # find contours (i.e., outlines) of the foreground objects in the
        # thresholded image
        # find contours and draw them in _img and return them as a list of tuples
        if auto_threshold:
            thresh = ImageFactory()
            thresh.set_image(self.get_image())
            thresh.set_image(thresh.get_contour_canny(auto=auto_canny, seuil=seuil))
        else:
            thresh = ImageFactory()
            thresh.set_image(self.get_image())
            thresh.to_gray_scale()
            thresh.gaussian_blur()
            thresh.adaptive_treshold()
        screen_cnt = []
        if not all_points:
            cnts = cv2.findContours(thresh.get_image(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
            for c in cnts:
                # Mo7iit
                peri = cv2.arcLength(c, True)
                # trouver le contour li kay dir chkl li baghin lwsst
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)

                # if our approximated contour has four points, then we
                # can assume that we have found our screen
                if len(approx) == 4:
                    screen_cnt = approx
                    break
        else:
            screen_cnt = cv2.findContours(thresh.get_image(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            screen_cnt = imutils.grab_contours(screen_cnt)

        # loop over the contours
        for c in screen_cnt:
            # draw each contour on the output image with a 3px thick purple
            # outline, then display the output contours one at a time
            cv2.drawContours(self.__img, [c], -1, (240, 0, 159), 3)

        pts = [tuple(p[0]) for p in screen_cnt]

        return pts

    def get_contour_canny(self, auto=True, seuil=75):
        gray = ImageFactory()
        gray.set_image(self.get_image())
        gray.to_gray_scale()
        gray.gaussian_blur()
        if auto:
            return imutils.auto_canny(gray.get_image())
        return cv2.Canny(gray.get_image(), seuil, 200)

    def gaussian_blur(self, taille_mask=(5, 5)):
        self.set_image(cv2.GaussianBlur(self.get_image(), taille_mask, 0))

    def get_median_blur(self):
        self.set_image(cv2.medianBlur(self.get_image(), 5))

    def scan(self):
        self.to_gray_scale()
        # convert the warped image to grayscale, then threshold it
        # to give it that 'black and white' paper effect
        t = threshold_local(self.__img, 11, offset=10, method="gaussian")
        warped = (self.__img > t).astype("uint8") * 255
        self.set_image(warped)

    def perspective_vue_from_4_points(self, points_list=None):
        # load the notecard code image, clone it, and initialize the 4 points
        # that correspond to the 4 corners of the notecard
        if points_list is None:
            pts = [(0, 0), (self.get_shape()[1], 0), (self.get_shape()[1], self.get_shape()[0]), (0,
                                                                                                  self.get_shape()[0])]
        else:
            pts = points_list

        pts = np.array(pts)

        # loop over the points and draw them on the cloned image
        """for (x, y) in pts:
            cv2.circle(self.__img, (x, y), 5, (0, 255, 0), -1)"""

        # apply the four point tranform to obtain a "birds eye view" of
        # the notecard
        self.set_image(imutils.perspective.four_point_transform(self.get_image(), pts))

    def find_skeleton(self):
        pass

    def get_shape(self):
        return self.get_image().shape

    def write_text(self, text_to_write, text_position_tuple=(0, 0), color_gbr_tuple=(0, 255, 0), font_size=0.7):
        cv2.putText(self.__img, text_to_write, text_position_tuple,
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, color_gbr_tuple, 2)

    def merge(self, image_link, alpha=None):
        img = ImageFactory(image_link)
        img_shape = self.get_shape()
        if img.get_shape() != img_shape:
            img_shape = (img_shape[1], img_shape[0])
            img.resize(img_shape[0:2])

        if alpha is None:
            self.set_image(cv2.add(self.get_image(), img.get_image()))
        else:
            self.set_image(cv2.addWeighted(self.get_image(), alpha, img.get_image(), 1 - alpha, 0))

    def rectangle(self, top_left_tuple=(0, 0), bottom_right_tuple=(10, 10), rect_color_tuple=(0, 0, 255)):
        cv2.rectangle(self.__img, top_left_tuple, bottom_right_tuple, rect_color_tuple, 2)

    def cercle(self, centre_tuple=(0, 0), rayon=20, cer_color_tuple=(255, 0, 0)):
        cv2.circle(self.__img, centre_tuple, rayon, cer_color_tuple, -1)

    def line(self, point_a=(0, 0), point_b=(10, 10), line_color=(0, 0, 255)):
        cv2.line(self.__img, point_a, point_b, line_color, 5)

    def threshold(self, seuil=200, inversed_binary=True):
        if inversed_binary:
            self.set_image(cv2.threshold(self.__img, seuil, 255, cv2.THRESH_BINARY_INV)[1])
        self.set_image(cv2.threshold(self.__img, seuil, 255, cv2.THRESH_BINARY)[1])

    def adaptive_treshold(self):
        self.set_image(cv2.adaptiveThreshold(self.__img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2))

    def save(self, image_name_path='image.jpg'):
        cv2.imwrite(image_name_path, self.get_image())

    def get_bgr_layers(self):
        return self.get_image().split()

    def get_histogramme(self):
        return cv2.calcHist([self.get_image()], [0], None, [256], [0, 256])

    def show_histograme(self, rgb=True):
        if rgb:
            b, g, r = cv2.split(self.get_image())
            fig = plt.figure(figsize=(8, 4))
            ax = fig.add_subplot(121)
            ax.imshow(self.get_image()[..., ::-1])
            ax = fig.add_subplot(122)
            for x, c in zip([b, g, r], ["b", "g", "r"]):
                xs = np.arange(256)
                ys = cv2.calcHist([x], [0], None, [256], [0, 256])
                ax.plot(xs, ys.ravel(), color=c)
        else:
            fig = plt.figure(figsize=(8, 4))
            ax = fig.add_subplot(121)
            img = ImageFactory()
            img.set_image(self.get_image())
            img.to_rgb()
            ax.imshow(img.get_image())
            ax = fig.add_subplot(122)
            xs = np.arange(256)
            ys = cv2.calcHist([self.get_image()], [0], None, [256], [0, 256])
            ax.plot(xs, ys.ravel(), color='black')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.show()

    def to_rgb(self):
        self.__img = cv2.cvtColor(self.get_image(), cv2.COLOR_BGR2RGB)

    @staticmethod
    def find_function(func):
        print(imutils.find_function(func))