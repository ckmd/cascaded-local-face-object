import cv2, dlib, sys
from statistics import mean
import numpy as np
import Gabor as gbr
import NumPyCNN as numpycnn
import matplotlib.pyplot as plt

# Output range is 0 to 1
def add_padding(lo_obj,pad):
    height = lo_obj.shape[0]
    width = lo_obj.shape[1]
    new_layer = np.zeros([height + 2*pad, width + 2*pad])
    new_layer += 0.5
    for i in range(height):
        for j in range(width):
            new_layer[i + pad][j + pad] = lo_obj[i][j]/255
    # cv2.imshow("test", new_layer)
    # cv2.waitKey(1)
    return new_layer

def convolution(image,filter_real, filter_imajiner):
    # Convolution
    l1_feature_map = numpycnn.conv(image, filter_real)
    l1_feature_map_i = numpycnn.conv(image, filter_imajiner)
    # Pooling
    l1_feature_map_pool = numpycnn.pooling(l1_feature_map, 2, 2)    
    l1_feature_map_pool_i = numpycnn.pooling(l1_feature_map_i, 2, 2)    
    # Create Magnitude
    magnitude = np.sqrt((l1_feature_map_pool.T ** 2) + (l1_feature_map_pool_i.T ** 2))
    magnitude = ( (magnitude - np.amin(magnitude) ) * 1 ) / ( np.amax(magnitude) - np.amin(magnitude) )
    # Create Phase
    # np.seterr(divide='ignore', invalid='ignore') # mengatasi runtime warning : invalid value encountered in true divide
    phase = np.arctan(l1_feature_map_pool_i.T / l1_feature_map_pool.T)
    phase = ( (phase - np.amin(phase) ) * 1 ) / ( np.amax(phase) - np.amin(phase) )
    return magnitude,phase

print("Welcome Gais, Pasang muka seganteng mungkin")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
jkw = cv2.imread('face.jpg')
cap = cv2.VideoCapture(0)

while True:
    _,frame = cap.read()

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(grey)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        landmark = predictor(grey, face)

        left_eye_x, left_eye_y = [], []
        right_eye_x, right_eye_y = [], []
        nose_x, nose_y = [], []
        left_mouth_x, left_mouth_y = [], []
        right_mouth_x, right_mouth_y = [], []

        for i in range(0,68):
            x0 = landmark.part(i).x
            y0 = landmark.part(i).y
            # crop both eye area
            if(i >= 30 and i <= 35):
                nose_x.append(x0)
                nose_y.append(y0)
            if(i >= 36 and i <= 41):
                left_eye_x.append(x0)
                left_eye_y.append(y0)
            if(i >= 42 and i <= 47):
                right_eye_x.append(x0)
                right_eye_y.append(y0)
            if(i == 48):
                left_mouth_x.append(x0)
                left_mouth_y.append(y0)
            if(i == 54):
                right_mouth_x.append(x0)
                right_mouth_y.append(y0)
            
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255),1)
            # cv2.circle(frame, (x0,y0), 3, (0,255,255),1)
            # draw line between node
            # if(i < 67):
                # if(i!=16 and i!=26 and i!=35 and i!=41 and i!=47):
                    # cv2.line(frame, (x0,y0) ,(landmark.part(i+1).x,landmark.part(i+1).y), (0,255,255),1)
        # cv2.circle(frame, (x27,y27), 3, (0,255,255),1)
        # draw rectangle in 5 local face object
        cv2.rectangle(frame, (min(left_eye_x)-9,min(left_eye_y)-9), (max(left_eye_x)+9,max(left_eye_y)+9), (0,255,255),1)
        cv2.rectangle(frame, (min(right_eye_x)-9,min(right_eye_y)-9), (max(right_eye_x)+9,max(right_eye_y)+9), (0,255,255),1)
        cv2.rectangle(frame, (min(nose_x)-9,min(nose_y)-9), (max(nose_x)+9,max(nose_y)+9), (0,255,255),1)
        cv2.rectangle(frame, (min(left_mouth_x)-9,min(left_mouth_y)-9), (max(left_mouth_x)+9,max(left_mouth_y)+9), (0,255,255),1)
        cv2.rectangle(frame, (min(right_mouth_x)-9,min(right_mouth_y)-9), (max(right_mouth_x)+9,max(right_mouth_y)+9), (0,255,255),1)

        # Crop 5 local face object
        left_eye = cv2.resize(grey[min(left_eye_y)-9:max(left_eye_y)+9,min(left_eye_x)-9:max(left_eye_x)+9],(50,25))
        right_eye = cv2.resize(grey[min(right_eye_y)-9:max(right_eye_y)+9,min(right_eye_x)-9:max(right_eye_x)+9],(50,25))
        nose = cv2.resize(grey[min(nose_y)-9:max(nose_y)+9,min(nose_x)-9:max(nose_x)+9],(65,40))
        left_mouth = cv2.resize(grey[min(left_mouth_y)-9:max(left_mouth_y)+9,min(left_mouth_x)-9:max(left_mouth_x)+9],(18,18))
        right_mouth = cv2.resize(grey[min(right_mouth_y)-9:max(right_mouth_y)+9,min(right_mouth_x)-9:max(right_mouth_x)+9],(18,18))

        # add padding, convolution, pooling, magnitude, phase in 5 local face object
        fitur_left_eye = convolution(add_padding(left_eye,8),gbr.filter2,gbr.filter2_i)
        fitur_right_eye = convolution(add_padding(right_eye,8),gbr.filter2,gbr.filter2_i)
        fitur_nose = convolution(add_padding(nose,8),gbr.filter2,gbr.filter2_i)
        fitur_left_mouth = convolution(add_padding(left_mouth,8),gbr.filter2,gbr.filter2_i)
        fitur_right_mouth = convolution(add_padding(right_mouth,8),gbr.filter2,gbr.filter2_i)

        # Flatten Layer
        input_eye = []
        input_nose = []
        input_mouth = []
        for in1, conv1 in enumerate(fitur_left_eye[0]):
            input_eye.append(conv1)
        for in1, conv1 in enumerate(fitur_left_eye[1]):
            input_eye.append(conv1)
        for in1, conv1 in enumerate(fitur_right_eye[0]):
            input_eye.append(conv1)
        for in1, conv1 in enumerate(fitur_right_eye[1]):
            input_eye.append(conv1)
        for in1, conv1 in enumerate(fitur_nose[0]):
            input_nose.append(conv1)
        for in1, conv1 in enumerate(fitur_nose[1]):
            input_nose.append(conv1)
        for in1, conv1 in enumerate(fitur_left_mouth[0]):
            input_mouth.append(conv1)
        for in1, conv1 in enumerate(fitur_left_mouth[1]):
            input_mouth.append(conv1)
        for in1, conv1 in enumerate(fitur_right_mouth[0]):
            input_mouth.append(conv1)
        for in1, conv1 in enumerate(fitur_right_mouth[1]):
            input_mouth.append(conv1)
        input_eye = np.array(np.array(input_eye).ravel())
        input_nose = np.array(np.array(input_nose).ravel())
        input_mouth = np.array(np.array(input_mouth).ravel())
        input_total = np.array([[*input_eye, *input_nose, *input_mouth]]) # sudah berupa array 2 dimensi

        cv2.waitKey(1)
        key = cv2.waitKey(1)
        if(key == 27):
            break