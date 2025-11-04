import cv2
from cv2 import CascadeClassifier

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# set to 0 to signify webcam
video = cv2.VideoCapture(0)

# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=35, detectShadows=False)

while True:
    ret, frame = video.read()

    # apply gray frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    # Apply background subtraction
    fgmask = fgbg.apply(blur_frame)

    # Apply morphological operations to remove noise
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, None)

    # Find contours
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 10000:
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(frame, "Motion", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Motion Detection", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
