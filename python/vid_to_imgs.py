import cv2
import numpy as np
from roboflow import Roboflow
import os
from ultralytics import YOLO


model= YOLO("YOLO_V8/runs/detect/yolov8s_v8_25e5/weights/best.pt")

def get_head_box(im):
    # detections = model.predict(im, confidence=60, overlap=30).json()
    detections = model(im, conf=0.8)
    bboxes = detections[0].boxes
    num_detections = bboxes.xyxy.size(dim=0)

    if num_detections > 0:
        detection = detections[0]
        bbox = detection.boxes
        x1 = bbox.xyxy[0][0] #int(detection['x']) - int(detection['width'] / 2)
        x2 = bbox.xyxy[0][2] #int(detection['x']) + int(detection['width'] / 2)
        y1 = bbox.xyxy[0][1] #int(detection['y']) - int(detection['height'] / 2)
        y2 = bbox.xyxy[0][3] #int(detection['y']) + int(detection['height'] / 2)
        box = [[x1, y1], [x2, y2]]
        return box
    else:
        return None

def main():
    # Initialize counter for image names
    img_cnt = 0
    
    # Loop through all videos in labeled_videos
    videos_directory = "../labeled_vids"
    print("Processing videos in " + videos_directory)
    for ant_video in os.listdir(videos_directory):
        path_to_video = os.path.join(videos_directory, ant_video)

        # Open video with OpenCV
        cap= cv2.VideoCapture(path_to_video)

        # Create path to save images
        ant_id = path_to_video.split("/")[-1].split(".")[0]
        path_to_imgs = "../labeled_images/" + ant_id
        os.mkdir(path_to_imgs)

        # Loop through all frames in video
        while(1):
            ret, frame = cap.read()
            if frame is None:
                break
            
            # Detect ant head
            head_box = get_head_box(frame)

            if head_box is None:
                continue

            # Crop image to head
            y1 = int(head_box[0][1])
            y2 = int(head_box[1][1])
            x1 = int(head_box[0][0])
            x2 = int(head_box[1][0])
            head_crop_img = frame[y1:y2,x1:x2]

            # Get ant label
            path_to_label = path_to_imgs + "/im" + str(img_cnt) + ".jpg"

            # Save cropped image to labeled_images
            cv2.imwrite(path_to_label ,head_crop_img)
            img_cnt += 1
          
        cap.release()
        print("Finished processing video: " + ant_video)
        print("Saved " + str(img_cnt) + " images to labeled_images/" + ant_id)
        print("------------------------------------------------------------\n")

if __name__ == "__main__":
    main()