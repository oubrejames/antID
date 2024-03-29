import cv2
import numpy as np
from roboflow import Roboflow
import os
from ultralytics import YOLO
import torch

# Choose witch model to use (true for bodies false for heads)
ant_body_flag = True

# Flag for if you are adding to the unseen dataset for testing
unseen_test_flag = True

# Import yolo model and choose correct directories for images and videos
videos_directory = "../labeled_vids"
if ant_body_flag:
    model= YOLO("YOLO_Body/runs/detect/yolov8s_v8_25e3/weights/best.pt")
    labelled_image_dir = "../labeled_images_bodies/"
    if unseen_test_flag:
        labelled_image_dir = "../unseen_body_imgs_bad_crop/"
        videos_directory = "../unseen_vids/"
else:
    model= YOLO("YOLO_V8/runs/detect/yolov8s_v8_25e6/weights/best.pt")
    labelled_image_dir = "../labeled_images/"
    if unseen_test_flag:
        labelled_image_dir = "../unseen_images/"
        videos_directory = "../unseen_vids/"

# Put on GPU 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)






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

def get_bbox(im, model = model):
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
        center_point = [abs(x2-x1)/2 + x1, abs(y2-y1)/2 + y1]

        x1 = center_point[1] - 1726/2
        x2 = center_point[1] + 1726/2
        y1 = center_point[1] - 610/2
        y2 = center_point[1] + 610/2

        if x1 < 0:
            difference = - x1
            x1 += difference
            x2 += difference

        if x2 > 2048:
            difference = x2 - 2048
            x1 -= difference
            x2 -= difference

        if y1 < 0:
            difference = - y1
            y1 += difference
            y2 += difference

        if y2 > 1536:
            difference = y2 - 1536
            y1 -= difference
            y2 -= difference

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
    print("Processing videos in " + videos_directory)
    for ant_video in os.listdir(videos_directory):
        path_to_video = os.path.join(videos_directory, ant_video)

        # Create path to save images
        ant_id = path_to_video.split("/")[-1].split(".")[0]
        path_to_imgs = labelled_image_dir + ant_id

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

            if ant_body_flag:
                # Detect ant body
                bbox = get_bbox(frame) # TODO revert this back
            else:
                # Detect ant head
                bbox = get_head_box(frame)

            if bbox is None:
                continue

            # Crop image to head
            y1 = int(bbox[0][1])
            y2 = int(bbox[1][1])
            x1 = int(bbox[0][0])
            x2 = int(bbox[1][0])
            head_crop_img = frame[y1:y2,x1:x2]

            # Get ant label
            path_to_label = path_to_imgs + "/im" + str(img_cnt) + ".jpg"

            # Save cropped image to labeled_images
            cv2.imwrite(path_to_label ,head_crop_img)
            img_cnt += 1
          
        cap.release()
        print("Finished processing video: " + ant_video)
        print("Saved " + str(img_cnt) + " images to labeled_images/" + ant_id)
        img_cnt = 0
        print("------------------------------------------------------------\n")

if __name__ == "__main__":
    main()