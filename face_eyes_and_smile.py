import cv2

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_classifier = cv2.CascadeClassifier('haarcascade_smile.xml')

def detect(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect multiple faces in the image
    faces = face_classifier.detectMultiScale(grey, 1.3, 5) # image, scaling factor, min neighbours
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2) # src, start, end, color, thickness
        # We can narrow down the search for eyes and smiles to the face. So, we create a region of interest which only includes the face
        roi_grey = grey[y:y+h, x:x+w]
        roi_original = image[y:y+h, x:x+w]
        # Detect multiple eyes in the image
        eyes = eye_classifier.detectMultiScale(roi_grey, 1.3, 5) # You need to tune this. Increase the number of neighbours
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_original, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)
        smiles = smile_classifier.detectMultiScale(roi_grey, 1.7, 7) # You need to tune this. Increase the number of neighbours
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_original, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)

    return image
def main():
    cap = cv2.VideoCapture(1)
    while True:
        _, frame = cap.read()
        frame = detect(frame)
        cv2.imshow("Window", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
