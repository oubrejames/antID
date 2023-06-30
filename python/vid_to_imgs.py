import cv2
import numpy as np
from matplotlib import pyplot as plt
# Opens the Video file
cap= cv2.VideoCapture('labeled_vids/ant_0.avi')
i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imwrite('labeled_imgs/ant_0/im'+str(i)+'.jpg',frame)
    i+=1
 
cap.release()
cv2.destroyAllWindows()