import cv2 as cv
import mediapipe as mp
import numpy as np


def cv_photo():
    img = cv.imread("images/variant-7.jpg")
    flip = cv.flip(img, flipCode=1)  # horizontal reflection
    w, h = flip.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv.getRotationMatrix2D((cX, cY), 180, 1.0)
    flip_flop = cv.warpAffine(flip, M, (w, h))  # 180 rotate
    cv.imshow('orig', img)
    cv.imshow('flip', flip)
    cv.imshow('flip_flop', flip_flop)
    cv.waitKey(0)
    cv.destroyAllWindows()


def find_circles(image):
    circles = cv.HoughCircles(
        image,
        cv.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=100
    )
    return circles


def cv_video():
    capture = cv.VideoCapture(0)
    ref_img = cv.imread('ref-point.jpg', cv.IMREAD_GRAYSCALE)
    # marker = cv.CascadeClassifier('')
    while True:
        ret, cam_img = capture.read()
        height, width = cam_img.shape[:2]

        # Initiate ORB detector
        orb = cv.ORB_create()

        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(ref_img, None)  # 209x202
        kp2, des2 = orb.detectAndCompute(cam_img, None)  # 640x480

        # create BFMatcher object
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        # Match descriptors.
        matches = bf.match(des1, des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw first n matches.
        first_n_matches = 5
        img_matches = cv.drawMatches(
            ref_img, kp1, cam_img, kp2, matches[:first_n_matches], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # central point
        cy = height/2
        cx = width/2
        cv.circle(cam_img, (int(cx), int(cy)), 3, (0, 190, 110), 2)

        # center of marker, probably
        prob_x = 0
        prob_y = 0
        for i in matches[:first_n_matches]:
            x, y = kp2[i.trainIdx].pt
            prob_x += x
            prob_y += y
        prob_y /= len(matches[:first_n_matches])
        prob_x /= len(matches[:first_n_matches])
        cv.circle(cam_img, (int(prob_x), int(prob_y)), 3, (0, 0, 255), 3)

        # show distance
        distance_to_show = np.sqrt(
            (cx-prob_x)*(cx-prob_x) + (cy-prob_y)*(cy-prob_y))
        cv.putText(cam_img, str(int(distance_to_show)),
                   (int(prob_x)-15, int(prob_y)-12), 1, 1, (0, 80, 255))
        cv.putText(cam_img, str(int(distance_to_show)),
                   (int(cx)-15, int(cy)-12), 1, 1, (0, 80, 255))
        cv.imshow("matches", img_matches)
        cv.imshow("Video", cam_img)

        k = cv.waitKey(30) & 0xFF
        if k == 27:
            break

    capture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    cv_photo()
    cv_video()
