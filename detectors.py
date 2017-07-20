#!/usr/bin/env python

import cv2
import numpy as np


class FaceDetector:


    def __init__(
            self,
            face_casc='params/haarcascade_frontalface_default.xml',
            left_eye_casc='params/haarcascade_lefteye_2splits.xml',
            right_eye_casc='params/haarcascade_righteye_2splits.xml',
            scale_factor=4):

        # resize images before detection
        self.scale_factor = scale_factor

        # load pre-trained cascades
        self.face_casc = cv2.CascadeClassifier(face_casc)
        if self.face_casc.empty():
            print 'Warning: Could not load face cascade:', face_casc
            raise SystemExit
        self.left_eye_casc = cv2.CascadeClassifier(left_eye_casc)
        if self.left_eye_casc.empty():
            print 'Warning: Could not load left eye cascade:', left_eye_casc
            raise SystemExit
        self.right_eye_casc = cv2.CascadeClassifier(right_eye_casc)
        if self.right_eye_casc.empty():
            print 'Warning: Could not load right eye cascade:', right_eye_casc
            raise SystemExit

    def detect(self, frame):

        frameCasc = cv2.cvtColor(
            cv2.resize(
                frame,
                (0, 0),
                fx=1.0 / self.scale_factor,
                fy=1.0 / self.scale_factor),
            cv2.COLOR_RGB2GRAY)
        faces = self.face_casc.detectMultiScale(
            frameCasc,
            scaleFactor=1.1,
            minNeighbors=3,
            flags=cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT) * self.scale_factor

        # if face is found: extract head region from bounding box
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 255, 0), 2)
            head = cv2.cvtColor(frame[y:y + h, x:x + w],
                                cv2.COLOR_RGB2GRAY)
            return True, frame, head, (x, y)

        return False, frame, None, (0, 0)

    def align_head(self, head):

        height, width = head.shape[:2]

        # detect left eye
        left_eye_region = head[0.2*height:0.5*height, 0.1*width:0.5*width]
        left_eye = self.left_eye_casc.detectMultiScale(
            left_eye_region,
            scaleFactor=1.1,
            minNeighbors=3,
            flags=cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT)
        left_eye_center = None
        for (xl, yl, wl, hl) in left_eye:
            # find the center of the detected eye region
            left_eye_center = np.array([0.1*width + xl + wl / 2,
                                        0.2*height + yl + hl / 2])
            break  # need only look at first, largest eye

        # detect right eye
        right_eye_region = head[0.2*height:0.5*height, 0.5*width:0.9*width]
        right_eye = self.right_eye_casc.detectMultiScale(
            right_eye_region,
            scaleFactor=1.1,
            minNeighbors=3,
            flags=cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT)
        right_eye_center = None
        for (xr, yr, wr, hr) in right_eye:
            # find the center of the detected eye region
            right_eye_center = np.array([0.5*width + xr + wr / 2,
                                         0.2*height + yr + hr / 2])
            break  # need only look at first, largest eye

        # need both eyes in order to align face
        # else break here and report failure (False)
        if left_eye_center is None or right_eye_center is None:
            return False, head

        # we want the eye to be at 25% of the width, and 20% of the height
        # resulting image should be square (desired_img_width,
        # desired_img_height)
        
        desired_eye_x = 0.25
        desired_eye_y = 0.2
        desired_img_width = 200
        desired_img_height = desired_img_width

        # get center point between the two eyes and calculate angle
        eye_center = (left_eye_center + right_eye_center) / 2
        eye_angle_deg = np.arctan2(right_eye_center[1] - left_eye_center[1],
                                   right_eye_center[0] - left_eye_center[0]) \
            * 180.0 / cv2.cv.CV_PI

        # scale distance between eyes to desired length
        eyeSizeScale = (1.0 - desired_eye_x * 2) * desired_img_width / \
            np.linalg.norm(right_eye_center - left_eye_center)

        # get rotation matrix
        rot_mat = cv2.getRotationMatrix2D(tuple(eye_center), eye_angle_deg,
                                          eyeSizeScale)

        # shift center of the eyes to be centered in the image
        rot_mat[0, 2] += desired_img_width*0.5 - eye_center[0]
        rot_mat[1, 2] += desired_eye_y*desired_img_height - eye_center[1]

        # warp perspective to make eyes aligned on horizontal line and scaled
        # to right size
        res = cv2.warpAffine(head, rot_mat, (desired_img_width,
                                             desired_img_width))

        # return success
        return True, res
