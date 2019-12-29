import time, cv2, dlib, sys, read_data, pickle
from statistics import mean
import numpy as np
import Gabor as gbr
import NumPyCNN as numpycnn
import matplotlib.pyplot as plt

start = time.time()

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
    np.seterr(divide='ignore', invalid='ignore') # mengatasi runtime warning : invalid value encountered in true divide
    phase = np.arctan(l1_feature_map_pool_i.T / l1_feature_map_pool.T)
    phase = ( (phase - np.amin(phase) ) * 1 ) / ( np.amax(phase) - np.amin(phase) )
    return magnitude,phase

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

print("Welcome Gais, Pasang muka seganteng mungkin")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
jkw = cv2.imread('face.jpg')
cap = cv2.VideoCapture(0)

data = read_data.data
labels = read_data.label
feature_set = data

# Defining Neural Network Synapse
np.random.seed(2)
weights = 2 * np.random.rand(5888,1000) - 1
weights2 = 2 * np.random.rand(1000,15) - 1
bias = 2 * np.random.rand(1,1000) - 1
bias2 = 2 * np.random.rand(1,15) - 1
lr = 0.05

# Defining Neural Network Pickle
weights = pickle.load(open("syn0.pickle", "rb"))
weights2 = pickle.load(open("syn1.pickle", "rb"))
bias = pickle.load(open("bias.pickle", "rb"))
bias2 = pickle.load(open("bias2.pickle", "rb"))

epoch = 2 * len(feature_set)
detected = 0
accurate = 0

for j in range(epoch):
    ri = np.random.randint(len(feature_set))
    frame = feature_set[ri]
# while True:
#     _,frame = cap.read()

    grey = frame
    faces = detector(grey)
    for face in faces:
        detected += 1
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
            
            # cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255),1)
            # cv2.circle(frame, (x0,y0), 3, (0,255,255),1)
            # draw line between node
            # if(i < 67):
                # if(i!=16 and i!=26 and i!=35 and i!=41 and i!=47):
                    # cv2.line(frame, (x0,y0) ,(landmark.part(i+1).x,landmark.part(i+1).y), (0,255,255),1)
        # cv2.circle(frame, (x27,y27), 3, (0,255,255),1)
        # draw rectangle in 5 local face object
        # cv2.rectangle(frame, (min(left_eye_x)-9,min(left_eye_y)-9), (max(left_eye_x)+9,max(left_eye_y)+9), (0,255,255),1)
        # cv2.rectangle(frame, (min(right_eye_x)-9,min(right_eye_y)-9), (max(right_eye_x)+9,max(right_eye_y)+9), (0,255,255),1)
        # cv2.rectangle(frame, (min(nose_x)-9,min(nose_y)-9), (max(nose_x)+9,max(nose_y)+9), (0,255,255),1)
        # cv2.rectangle(frame, (min(left_mouth_x)-9,min(left_mouth_y)-9), (max(left_mouth_x)+9,max(left_mouth_y)+9), (0,255,255),1)
        # cv2.rectangle(frame, (min(right_mouth_x)-9,min(right_mouth_y)-9), (max(right_mouth_x)+9,max(right_mouth_y)+9), (0,255,255),1)

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
        # for in1, conv1 in enumerate(fitur_left_eye[1]):
        #     input_eye.append(conv1)
        for in1, conv1 in enumerate(fitur_right_eye[0]):
            input_eye.append(conv1)
        # for in1, conv1 in enumerate(fitur_right_eye[1]):
        #     input_eye.append(conv1)
        for in1, conv1 in enumerate(fitur_nose[0]):
            input_nose.append(conv1)
        # for in1, conv1 in enumerate(fitur_nose[1]):
        #     input_nose.append(conv1)
        for in1, conv1 in enumerate(fitur_left_mouth[0]):
            input_mouth.append(conv1)
        # for in1, conv1 in enumerate(fitur_left_mouth[1]):
        #     input_mouth.append(conv1)
        for in1, conv1 in enumerate(fitur_right_mouth[0]):
            input_mouth.append(conv1)
        # for in1, conv1 in enumerate(fitur_right_mouth[1]):
        #     input_mouth.append(conv1)
        input_eye = np.array(np.array(input_eye).ravel())
        input_nose = np.array(np.array(input_nose).ravel())
        input_mouth = np.array(np.array(input_mouth).ravel())
        input_total = np.array([[*input_eye, *input_nose, *input_mouth]]) # sudah berupa array 2 dimensi

        l1 = sigmoid(np.dot(input_total, weights) + bias)
        z = sigmoid(np.dot(l1, weights2) + bias2)
        
        # print(z.shape, labels[ri].shape)
        # backpropagation step 1
        error = z - np.array([labels[ri]])
        # print(ri, labels[ri], z)
        # backpropagation step 2
        dcost_dpred = error
        dpred_dz = sigmoid_der(z)
        z_delta = dcost_dpred * dpred_dz

        l1_error = z_delta.dot(weights2.T)
        dpred_dl1 = sigmoid_der(l1_error)
        l1_delta = l1_error * dpred_dl1

        l1 = l1.T
        weights2 -= lr * np.dot(l1,z_delta)
        input_total = input_total.T
        # print(z_delta.shape, inputs.shape)
        weights -= lr * np.dot(input_total, l1_delta)

        for num in z_delta:
            bias2 -= lr * num

        for num in l1_delta:
            bias -= lr * num
        if(np.argmax(labels[ri]) == np.argmax(z[0])):
            accurate += 1
            current = time.time()
            print(round((current - start),1),'s',round((j/epoch*100),2),'%', np.argmax(labels[ri]), labels[ri], z[0])
            print("accurate : ", accurate / detected * 100, "%")
        cv2.waitKey(1)
        current = time.time()
        cv2.imshow("frame",frame)
        print(input_total.shape)
        print(current - start)
        key = cv2.waitKey(1)
        if(key == 27):
            break

# Save Synaplse / Model into Pickle
pickle_out = open("syn0.pickle", "wb")
pickle.dump(weights, pickle_out)
pickle_out = open("syn1.pickle", "wb")
pickle.dump(weights2, pickle_out)
pickle_out = open("bias.pickle", "wb")
pickle.dump(bias, pickle_out)
pickle_out = open("bias2.pickle", "wb")
pickle.dump(bias2, pickle_out)
pickle_out.close()

end = time.time()
print(end - start, "s")