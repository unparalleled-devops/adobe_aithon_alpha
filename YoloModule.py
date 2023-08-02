
# Importing all the modules
from ultralytics import YOLO
import os
import numpy as np
from PIL import Image 
import time 
from concurrent.futures import ProcessPoolExecutor
'''
Over here we setting up the model variable which will be using the YOLO module
'results' acts as an inference of the model which is trained on the yolov8m.pt dataset

'''
model = YOLO("yolov8m.pt")

source = "All_Images"
results = model(source,stream=True)
results = model.predict(source,conf=0.2)



# Yolov8Module class is being initialized which now we will use through out our projects for various tasks
class Yolov8Module:
    def __init__(self,results):
        self.results = results
    
    '''
      We are defining the Boxed_image method of the class Yolov8Module which will be giving us the image 
      with the boxes around it 
    '''
   

    def Boxed_images(results):
        counter = 1 # this is a counter for naming the boxed images in the loop
        if not os.path.exists("Boxed_image"):
            os.makedirs("Boxed_image")
        start_time = time.time() 
       
         # Get the current time before starting the loop
        for result in results:
            # Convert result.plot() to a NumPy array and apply BGR conversion
            boxed_image_array = np.array(result.plot()[:, :, ::-1])
            boxed_image = Image.fromarray(boxed_image_array.astype(np.uint8))
            boxed_image.save(f"Boxed_image/Box_{str(counter)}.png")
            counter += 1  # Update the counter for the next image
        
        end_time = time.time()  # Get the current time after finishing the loop
        elapsed_time = end_time - start_time
    
        print(f"Time taken: {elapsed_time:.4f} seconds")

Yolov8Module.Boxed_images(results)



