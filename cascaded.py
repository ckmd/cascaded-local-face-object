import cv2, dlib, sys

print(cv2.__version__)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
jkw = cv2.imread('face.jpg')
cap = cv2.VideoCapture(0)

# while True:
    # _,frame = cap.read()
frame = cv2.imread("picture_test/person09291+60+90.jpg")
grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = detector(grey)
for face in faces:
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    landmark = predictor(grey, face)
    # cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255),4)
    for i in range(0,68):
        x0 = landmark.part(i).x
        y0 = landmark.part(i).y
        # print(x0 - x27,y0 - y27)
        cv2.circle(frame, (x0,y0), 2, (0,255,255),2)
        # if(i < 67):
        #     if(i!=16 and i!=26 and i!=35 and i!=41 and i!=47):
        #         cv2.line(frame, (x0,y0) ,(landmark.part(i+1).x,landmark.part(i+1).y), (0,255,255),1)
    # cv2.circle(frame, (x27,y27), 3, (0,255,255),1)

cv2.imshow("jokowi", frame)
cv2.imwrite("landmark_0.jpg", frame)
key = cv2.waitKey(0)
# if(key == 27):
#     break