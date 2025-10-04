
import cv2
import dlib
import numpy as np
import tensorflow as tf
from imutils import face_utils
from pygame import mixer


model = tf.keras.models.load_model("drowsiness_cnn.h5")
class_names = ["Closed", "No_Yawn", "Open", "Yawn"]


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


mixer.init()
mixer.music.load("alarm.wav")

def prepare_region(roi):
    roi = cv2.resize(roi, (64,64))
    roi = roi.astype("float32")/255.0
    roi = np.expand_dims(roi, axis=(0,-1))
    return roi

cap = cv2.VideoCapture(0)
eye_counter = 0
mouth_counter = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        (x1,y1,w1,h1) = cv2.boundingRect(np.array([leftEye]))
        (x2,y2,w2,h2) = cv2.boundingRect(np.array([rightEye]))
        leftROI = gray[y1:y1+h1, x1:x1+w1]
        rightROI = gray[y2:y2+h2, x2:x2+w2]

        # Mouth
        mouth = shape[mStart:mEnd]
        (mx,my,mw,mh) = cv2.boundingRect(np.array([mouth]))
        mouthROI = gray[my:my+mh, mx:mx+mw]

        # Predictions
        if leftROI.size and rightROI.size:
            pred_left = class_names[np.argmax(model.predict(prepare_region(leftROI)))]
            pred_right = class_names[np.argmax(model.predict(prepare_region(rightROI)))]
            if pred_left == "Closed" and pred_right == "Closed":
                eye_counter += 1
                if eye_counter > 15:
                    cv2.putText(frame,"ALERT! EYES CLOSED",(10,30),
                                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
                    if not mixer.music.get_busy(): mixer.music.play()
            else:
                eye_counter = 0

        if mouthROI.size:
            pred_mouth = class_names[np.argmax(model.predict(prepare_region(mouthROI)))]
            if pred_mouth == "Yawn":
                mouth_counter += 1
                if mouth_counter > 10:
                    cv2.putText(frame,"ALERT! YAWNING",(10,60),
                                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
                    if not mixer.music.get_busy(): mixer.music.play()
            else:
                mouth_counter = 0

    cv2.imshow("Day 4 - Final System", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()
