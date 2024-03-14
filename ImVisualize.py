import cv2
import numpy as np

def drawLandmarks(image_path, points_path):
    def read_pts(filename):
        return np.loadtxt(filename, comments=("version:", "n_points:", "{", "}"))


    points = read_pts(points_path)
    image = cv2.imread(image_path)
    # cv2.imshow("a", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    for X in points:
        image = cv2.circle(image, (int(X[0]), int(X[1])), 3, (0, 0, 255), -1)
    return image
