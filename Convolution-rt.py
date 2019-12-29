import cv2, dlib, sys
from statistics import mean
import NumPyCNN as numpycnn
import Gabor as gbr
import numpy as np

print(cv2.__version__)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
jkw = cv2.imread('face.jpg')
cap = cv2.VideoCapture(0)

while True:
    _,frame = cap.read()

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(grey)
    rotated270 = grey
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        crop = grey[y1:y2,x1:x2]
        grey = cv2.resize(crop,(100,100))
        l1_feature_map = numpycnn.conv(grey/255, gbr.filter3)
        l1_feature_map_i = numpycnn.conv(grey/255, gbr.filter3_i)

        magnitude = np.sqrt((l1_feature_map.T ** 2) + (l1_feature_map_i.T ** 2))
        magnitude = ( (magnitude - np.amin(magnitude) ) * 1 ) / ( np.amax(magnitude) - np.amin(magnitude) )
        
        phase = np.arctan(l1_feature_map_i.T / l1_feature_map.T)
        phase = ( (phase - np.amin(phase) ) * 1 ) / ( np.amax(phase) - np.amin(phase) )
        
        print(grey.shape, phase[2,:,:].shape)
        
        img = phase[0,:,:]
        (h,w) = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 270, 1.0)
        rotated2700 = cv2.warpAffine(img, M, (h, w))
        img = phase[1,:,:]
        (h,w) = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 270, 1.0)
        rotated2701 = cv2.warpAffine(img, M, (h, w))
        img = phase[2,:,:]
        (h,w) = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 270, 1.0)
        rotated2702 = cv2.warpAffine(img, M, (h, w))
        img = phase[3,:,:]
        (h,w) = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 270, 1.0)
        rotated2703 = cv2.warpAffine(img, M, (h, w))

        img = magnitude[0,:,:]
        (h,w) = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 270, 1.0)
        mag0 = cv2.warpAffine(img, M, (h, w))
        img = magnitude[1,:,:]
        (h,w) = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 270, 1.0)
        mag1 = cv2.warpAffine(img, M, (h, w))
        img = magnitude[2,:,:]
        (h,w) = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 270, 1.0)
        mag2 = cv2.warpAffine(img, M, (h, w))
        img = magnitude[3,:,:]
        (h,w) = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 270, 1.0)
        mag3 = cv2.warpAffine(img, M, (h, w))

    cv2.imshow("r0", l1_feature_map[:,:,0])
    cv2.imshow("r1", l1_feature_map[:,:,1])
    cv2.imshow("r2", l1_feature_map[:,:,2])
    cv2.imshow("r3", l1_feature_map[:,:,3])

    cv2.imshow("i0", l1_feature_map_i[:,:,0])
    cv2.imshow("i1", l1_feature_map_i[:,:,1])
    cv2.imshow("i2", l1_feature_map_i[:,:,2])
    cv2.imshow("i3", l1_feature_map_i[:,:,3])

    cv2.imshow("p0", rotated2700)
    cv2.imshow("p1", rotated2701)
    cv2.imshow("p2", rotated2702)
    cv2.imshow("p3", rotated2703)

    cv2.imshow("m0", mag0)
    cv2.imshow("m1", mag1)
    cv2.imshow("m2", mag2)
    cv2.imshow("m3", mag3)

    cv2.waitKey(1)
    key = cv2.waitKey(1)
    if(key == 27):
        break