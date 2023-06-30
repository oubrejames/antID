import numpy as np
import cv2 
from matplotlib import pyplot as plt
from roboflow import Roboflow

def crop_to_face(img):
    # Locate ant face
    detections = model.predict(img, confidence=60, overlap=30).json()
    if detections['predictions']:
        detection = detections['predictions'][0]

        # Get bounding box
        x1 = int(detection['x']) - int(detection['width'] / 2)
        x2 = int(detection['x']) + int(detection['width'] / 2)
        y1 = int(detection['y']) - int(detection['height'] / 2)
        y2 = int(detection['y']) + int(detection['height'] / 2)
        crop_img = img[y1:y2, x1:x2]
        return crop_img
    else:
        return img

# Get Face Detection model from Roboflow
rf = Roboflow(api_key="WZMvKYOhn8xpuDVHz6JX")
project = rf.workspace("antid").project("ant-face-detect")
model = project.version(1).model

# Read some ant images
ant0_0 = cv2.imread("../labeled_images/ant_0/ant_48_im_0.jpg")
ant0_123 = cv2.imread("../labeled_images/ant_0/ant_48_im_20.jpg")
ant1_0 = cv2.imread("../labeled_images/ant_1/ant_49_im_0.jpg")
ant1_123 = cv2.imread("../labeled_images/ant_1/ant_49_im_123.jpg")

# Crop images to face
ant0_0 = crop_to_face(ant0_0)
ant0_123 = crop_to_face(ant0_123)
ant1_0 = crop_to_face(ant1_0)
ant1_123 = crop_to_face(ant1_123)

# # Show cropped images
# cv2.imshow("image00", ant0_0)
# cv2.imshow("image0123", ant0_123)
# cv2.imshow("image10", ant1_0)
# cv2.imshow("image1123", ant1_123)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints with ORB
kp00 = orb.detect(ant0_0,None)
# compute the descriptors with ORB
kp00, des00 = orb.compute(ant0_0, kp00)

# find the keypoints with ORB
kp0123 = orb.detect(ant0_123,None)
# compute the descriptors with ORB
kp0123, des0123 = orb.compute(ant0_123, kp0123)

# find the keypoints with ORB
kp10 = orb.detect(ant1_0,None)
# compute the descriptors with ORB
kp10, des10 = orb.compute(ant1_0, kp10)

# find the keypoints with ORB
kp1123 = orb.detect(ant1_123,None)
# compute the descriptors with ORB
kp1123, des1123 = orb.compute(ant1_123, kp1123)

# draw only keypoints location,not size and orientation
img00 = cv2.drawKeypoints(ant0_0, kp00, None, color=(0,255,0), flags=0)
plt.imshow(img00), plt.show()

# draw only keypoints location,not size and orientation
img0123 = cv2.drawKeypoints(ant0_123, kp0123, None, color=(0,255,0), flags=0)
plt.imshow(img0123), plt.show()

# draw only keypoints location,not size and orientation
img10 = cv2.drawKeypoints(ant1_0, kp10, None, color=(0,255,0), flags=0)
plt.imshow(img10), plt.show()

# draw only keypoints location,not size and orientation
img1123 = cv2.drawKeypoints(ant1_123, kp1123, None, color=(0,255,0), flags=0)
plt.imshow(img1123), plt.show()

cv2.FlannBasedSegment({'algorithm':1, 'trees':10}, {}).knnMatch(des10, des1123, k=2)

match_points = []
best_score = 0
for p, q in matches:
    if p.distance < 0.1*q.distance:
        match_points.append(p)
        
keypoints = 0
if len(kp10) < len(kp1123):
    keypoints = len(kp10)
else:
    keypoints = len(kp1123)

if len(match_points) / keypoints *100 > best_score:
    best_score = len(match_points) / keypoints *100

print(best_score)