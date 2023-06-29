from roboflow import Roboflow
import cv2

rf = Roboflow(api_key="WZMvKYOhn8xpuDVHz6JX")
project = rf.workspace("antid").project("ant-face-detect")
model = project.version(1).model

# infer on a local image
detections = model.predict("/home/oubre/ants/antid_ws/labeled_images/ant_5/ant_53_im_190.jpg", confidence=40, overlap=30).json()
detection = detections['predictions'][0]


x1 = int(detection['x']) - int(detection['width'] / 2)
x2 = int(detection['x']) + int(detection['width'] / 2)
y1 = int(detection['y']) - int(detection['height'] / 2)
y2 = int(detection['y']) + int(detection['height'] / 2)
box = (x1, x2, y1, y2)

# Draw the bounding box on the image
img = cv2.imread("/home/oubre/ants/antid_ws/labeled_images/ant_5/ant_53_im_190.jpg")
cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
