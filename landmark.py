import cv2, dlib, sys
from statistics import mean

print(cv2.__version__)
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
        cv2.rectangle(frame, (min(left_eye_x)-9,min(left_eye_y)-9), (max(left_eye_x)+9,max(left_eye_y)+9), (0,255,255),1)
        cv2.rectangle(frame, (min(right_eye_x)-9,min(right_eye_y)-9), (max(right_eye_x)+9,max(right_eye_y)+9), (0,255,255),1)
        cv2.rectangle(frame, (min(nose_x)-9,min(nose_y)-9), (max(nose_x)+9,max(nose_y)+9), (0,255,255),1)
        cv2.rectangle(frame, (min(left_mouth_x)-9,min(left_mouth_y)-9), (max(left_mouth_x)+9,max(left_mouth_y)+9), (0,255,255),1)
        cv2.rectangle(frame, (min(right_mouth_x)-9,min(right_mouth_y)-9), (max(right_mouth_x)+9,max(right_mouth_y)+9), (0,255,255),1)

    # Membuat bounding box dengan mean, hasilnya kurang bagus karena rasio x dan y berbeda
    # mean_x, mean_y = int(mean(left_eye_x)),int(mean(left_eye_y))
    # cv2.rectangle(frame, (mean_x-17,mean_y-17), (mean_x+17,mean_y+17), (255,0,0),1)

    cv2.imshow("jokowi", frame)
    cv2.waitKey(1)
    key = cv2.waitKey(1)
    if(key == 27):
        break