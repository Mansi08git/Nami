#import opencv library
import cv2

#load the classifier
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

#video capture object
vid = cv2.VideoCapture(0)

while(True):
    ret , frame = vid.read()
    cv2.imshow('frame',frame)
    #converting into grayscale
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #classifier to perform the face detection
    face = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)
    #rectangle box
    for (x, y, w, h) in face: cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    
vid.release()
cv2.destroyAllWindows()







