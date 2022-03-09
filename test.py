from pathlib import Path
import os
from typing import List
import cv2
import numpy as np
import sys
import time

root = Path(os.path.dirname(__file__)).resolve()
path = root.joinpath("data/dataset/s10").resolve()
data_path = root.joinpath("../dataset/lipnet/train/s10").resolve()

sample = list(path.glob("*.npy"))[0]

data = np.load(sample)

print(data.shape)

# Read video
print(str(data_path.joinpath(sample.stem + ".mpg")))

# cv2.namedWindow()
window_name = "default"

for i in data:
    cv2.imshow(window_name, i)

    if cv2.waitKey(1) == ord("w"):
        continue
    else:
        time.sleep(0.5)

    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
# video = cv2.VideoCapture(str(data_path.joinpath(sample.stem + ".mpg")))

# # Exit if video not opened.
# if not video.isOpened():
#     print("Could not open video")
#     sys.exit()

# Read first video.
# count = 0
# while video.isOpened():
#     print(data[count])
#     if count != 0:
#         break
#     # Read a new frame
#     ret, frame = video.read()

#     if not ret:
#         break

#     # Start timer
#     timer = cv2.getTickCount()

#     # Calculate Frames per second (FPS)
#     fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

#     # Draw bounding box
#     if ret:
#         coor = data[count]
#         # Tracking success
#         img = cv2.rectangle(
#             frame,
#             pt1=(coor[0], coor[1]),
#             pt2=(coor[2], coor[3]),
#             color=(255, 0, 0),
#             thickness=2,
#         )
#     else:
#         # Tracking failure
#         cv2.putText(
#             frame,
#             "Tracking failure detected",
#             (100, 80),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.75,
#             (0, 0, 255),
#             2,
#         )

#     # cv2.putText(
#     #     frame,
#     #     "FPS : " + str(int(fps)),
#     #     (100, 50),
#     #     cv2.FONT_HERSHEY_SIMPLEX,
#     #     0.75,
#     #     (50, 170, 50),
#     #     2,
#     # )

#     # Display result
#     cv2.imshow("Tracking", frame)

#     # Exit if ESC pressed
#     if cv2.waitKey(1) == ord("q"):
#         break
