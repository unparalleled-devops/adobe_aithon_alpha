from ultralytics import YOLO
model = YOLO("yolov8m.pt")
import os
import numpy as np

listings = os.listdir("All_Images")
source = "All_Images"
results = model(listings,stream=True)
results = model.predict(source,conf=0.4)


class Yolov8Module:
    def __init__(self,results):
        self.results = results
    
    def Boxed_images(results):
        count = 1
        if not os.path.exists("Boxed_image"):
            os.makedirs("Boxed_image")
        for result in results:
    
            from PIL import Image 
            Boxed_image = Image.fromarray(result.plot()[:,:,::-1])
            Boxed_image.save(f"Boxed_image/Box_{str(count)}.png")
            
                
            count = count + 1
Yolov8Module.Boxed_images(results)


