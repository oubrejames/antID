
from ultralytics import YOLO
 
# Load the model.
model = YOLO('yolov8s.pt')
 
# Training.
results = model.train(
   data='ant-face-detect.v1-ant_head_v1.yolov8/data.yaml',
   imgsz=800,
   epochs=25,
   batch=8,
   name='yolov8s_v8_25e'
)