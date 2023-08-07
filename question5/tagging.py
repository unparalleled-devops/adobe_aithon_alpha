from ultralytics import YOLO
import os
from pathlib import Path


class Tagging:
    def __init__(self, source):
        self.model = YOLO("yolov8m.pt")
        self.source = source

    def tag_images(self):
        """Tags each image using YOLOv8

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing image path and identified entities.
        """
        yolo_output_list = []
        for root, _, files in os.walk(self.source):
            for file in files:
                if file.endswith(".jpg"):
                    img_path = os.path.join(root, file)
                    yolo_result = self.model.predict(img_path, conf=0.2)
                    print(yolo_result)


    
        
