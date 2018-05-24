import cv2

eye_classifier = cv2.CascadeClassifier('haarcascade_eye.xml')
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_face_and_eyes(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(grey, 1.5, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
        roi_grey = grey[y:y+h, x:x+w]
        roi_original = image[y:y+h, x:x+w]
        eyes = eye_classifier.detectMultiScale(roi_grey, 1.1, 3)
        for ex, ey, ew, eh in eyes:
            cv2.rectangle(roi_original, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)
    return image


cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    image = detect_face_and_eyes(frame)
    cv2.imshow('Video', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
