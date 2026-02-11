import sys
import os

local_path = r"C:\ultralytics-main" 

sys.path.insert(0, local_path)

from ultralytics import YOLO
def main():
    model = YOLO('yolov11n.pt')  
    results = model.train(
        data='C:\ultralytics-main\ultralytics-main\ultralytics\cfg\models\11\yolo11_CPAM_backbone_FARM.yaml',   
        epochs=100,              
        imgsz=640,               
        batch=16,                
        device=0,                
        name='my_custom_model',  
        patience=50              
    )

if __name__ == '__main__':
    main()