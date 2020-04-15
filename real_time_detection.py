import cv2 as cv
import time

import call_model
import utils

cam = cv.VideoCapture(0)
model = call_model.load_model()

while (1):

    ret, frame = cam.read()
    try:
        r, res = utils.detect_real_time_phase(frame)
        classes = call_model.get_real_time_prediction(r, model)
        img = cv.rectangle(frame, (res[0][2], res[0][0]), (res[0][3], res[0][1]), (0, 255, 0), 5)
        img = cv.putText(frame, classes, (res[0][3] - 10, res[0][1] - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                         cv.LINE_AA)
    except:
        print("Don't cover the camera")

    cv.imshow("Real Time Signs Detection", frame)
    time.sleep(0.1)
    if cv.waitKey(33) == ord('a'):
        exit()

cam.release()
cv.destroyAllWindows()
