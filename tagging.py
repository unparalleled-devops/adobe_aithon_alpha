import drive_functions as dr
from ultralytics import YOLO


class Tagging:
    def __init__(self, source):
        self.model = YOLO("yolov8m.pt")
        self.source = dr.get_content_path(source)

    def tag_images(self, conf=0.39):
        """Tags each image using YOLOv8

        Returns:
            None
        """

        if dr.is_on_drive():
            dr.mount(self.source)

        return self.model.predict(self.source, conf=conf)
