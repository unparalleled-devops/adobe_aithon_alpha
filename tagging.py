import os
import time
import numpy as np
from ultralytics import YOLO
from PIL import Image


class Tagging:
    def __init__(self, source):
        self.model = YOLO("yolov8m.pt")
        self.source = source

    def tag_images(self):
        """Tags each image using YOLOv8

        Returns:
            None
        """

        return self.model.predict(self.source, conf=0.2)
