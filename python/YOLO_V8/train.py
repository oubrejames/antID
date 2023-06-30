from ultralytics import YOLO

!nvidia-smi
!yolo task=detect mode=train model=yolov8s.pt data={dataset.location}/ant_head_v8.yaml epochs=25 imgsz=800 plots=True