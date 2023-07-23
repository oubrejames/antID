import cv2
import numpy as np
from roboflow import Roboflow
import os
from ultralytics import YOLO

# Import yolo model
model= YOLO("YOLO_V8/runs/detect/yolov8s_v8_25e6/weights/best.pt")

def get_head_box(im):
    """Take a video frame and return the bounding box of the ant head if it exists.

    Args:
        im (CV::MAT): Input image frame

    Returns:
        list: Two points idescribing the bounding box of the ant head
    """

    detections = model(im, conf=0.8)
    bboxes = detections[0].boxes
    num_detections = bboxes.xyxy.size(dim=0)

    if num_detections > 0:
        detection = detections[0]
        bbox = detection.boxes
        x1 = bbox.xyxy[0][0]
        x2 = bbox.xyxy[0][2]
        y1 = bbox.xyxy[0][1]
        y2 = bbox.xyxy[0][3]
        box = [[x1, y1], [x2, y2]]
        return box
    else:
        return None

def main():
    """
    This script loops through all videos in a directory of labeled videos, performs object detection on
    each frame, crops the image to an ant head, and saves the cropped image to a directory of labeled 
    images.
    """

    # Initialize counter for image names
    img_cnt = 0

    # Loop through all videos in labeled_videos
    videos_directory = "../labeled_vids"
    # videos_directory = "../unseen_vids" # Two video directories to make training and testing data

    print("Processing videos in " + videos_directory)
    for ant_video in os.listdir(videos_directory):
        path_to_video = os.path.join(videos_directory, ant_video)

        # Create path to save images
        ant_id = path_to_video.split("/")[-1].split(".")[0]
        path_to_imgs = "../labeled_images/" + ant_id
        # path_to_imgs = "../unseen_images/" + ant_id # Must change images_directory for unseen_images

        # Check if directory already exists
        if os.path.exists(path_to_imgs):
            print("Images for " + ant_id + " already exist. Skipping...")
            continue

        os.mkdir(path_to_imgs)

        # Open video with OpenCV
        cap= cv2.VideoCapture(path_to_video)

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