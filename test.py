from ultralytics import YOLO
import cv2
import os


weights_path = 'C:\ultralytics-main\ultralytics\weights\best.pt' 
source_path = 'C:\ultralytics-main\ultralytics\test_image.jpg' 
save_dir = 'C:\ultralytics-main\predict_result'

def run_inference():

    if not os.path.exists(weights_path):
        print(f"ERROR:Couldn't find {weights_path}")
        return


    print(f"Loading Model: {weights_path}...")
    model = YOLO(weights_path)
    results = model.predict(
        source=source_path,
        save=True,
        project=os.path.dirname(save_dir), 
        name=os.path.basename(save_dir),   
        conf=0.5,  #
        iou=0.45,
        exist_ok=True 
    )

    print(f"\nSave the result in {os.path.join(os.path.dirname(save_dir), os.path.basename(save_dir))}")

if __name__ == '__main__':
    run_inference()