import cv2
import time
import pandas
from datetime import datetime

first_frame = None


video = cv2.VideoCapture(0)

status_list=[None, None]
times=[]
df = pandas.DataFrame(columns=["Start","End"])

# waiting for camera to adjust brightness for the first_frame
check, frame = video.read()
time.sleep(1)

while True:
    check, frame = video.read()

    status=0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11,11), 0)

    if first_frame is None:
        first_frame = gray
        continue

    delta_frame = cv2.absdiff(first_frame, gray)
    print(first_frame)
    print(gray)
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    (cntrs,_)=cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contours in cntrs:
        if cv2.contourArea(contours) < 10_000:
            continue
        else:
            status = 1
            (x, y, w, h) = cv2.boundingRect(contours)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0))

    status_list.append(status)
    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())
    elif status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())
    # cv2.imshow("FirstFrame", first_frame)
    # cv2.imshow("Gray", gray)
    # cv2.imshow("Delta", delta_frame)
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) == ord('q'):
        if status == 1:
            times.append(datetime.now())
        break

for i in range(0, len(times),2):
    df = df.append({"Start": times[i], "End": times[i+1]}, ignore_index=True)

df.to_csv("Motion.csv")

video.release()
cv2.destroyAllWindows()

