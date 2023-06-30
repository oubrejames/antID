#####################################################################
############# WILL BREAK THIS INTO MULTIPLE FILES LATER #############
#####################################################################

#####################################################################
################### CROP IMAGE TO HEAD OF ANT #######################
#####################################################################

from roboflow import Roboflow
import cv2

rf = Roboflow(api_key="WZMvKYOhn8xpuDVHz6JX")
project = rf.workspace("antid").project("ant-face-detect")
model = project.version(1).model

# infer on a local image
im = cv2.imread("/home/oubre/ants/antid_ws/labeled_images/ant_5/ant_53_im_190.jpg")
detections = model.predict(im, confidence=60, overlap=30).json()
detection = detections['predictions'][0]

x1 = int(detection['x']) - int(detection['width'] / 2)
x2 = int(detection['x']) + int(detection['width'] / 2)
y1 = int(detection['y']) - int(detection['height'] / 2)
y2 = int(detection['y']) + int(detection['height'] / 2)
box = (x1, x2, y1, y2)

# Draw the bounding box on the image
img = cv2.imread("/home/oubre/ants/antid_ws/labeled_images/ant_5/ant_53_im_190.jpg")
crop_img = img[y1:y2, x1:x2]

cv2.imshow("image", crop_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#####################################################################
######################### Apply Filters #############################
#####################################################################