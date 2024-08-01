#importing the libraries
import mtcnn_cv2
from mtcnn_cv2 import MTCNN
import cv2

#detector object
detector = MTCNN()

#live webcam 
video = cv2.VideoCapture(0)

if(video.isOpened() == False):
    print("Webcam is not working")

while(True):
    #reading frame from camera 
    ret , frame = video.read()

    #using detectfaces model for face detection
    faces = detector.detect_faces(frame)

    #drawing boxes
    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

     # show the resulting frame
    cv2.imshow('Real-time Face Detection', frame)

    # press 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the video capture object and close all windows
video.release()
cv2.destroyAllWindows()

