import os
import time

import numpy as np
from PIL import Image

import tagging


def create_boxed_images(results):
    """Create a boxed image for each result.

    Creates a boxed image for each result that the model predicted.
    Writes files to a separate directory.

    Returns:
        None
    """

    # resulting image_path
    if not os.path.exists("boxed_images"):
        os.makedirs("boxed_images")

    # Time marker to see how long it takes
    start_time = time.time()

    for result in results:
        base_path = os.path.basename(result.path)

        # Convert result.plot() to a NumPy array and apply BGR conversion
        # by flipping the third axis
        image_array = result.plot()
        boxed_image = Image.fromarray(np.flip(image_array, 2))

        boxed_image.save(f"boxed_images/boxed_{base_path}")

    end_time = time.time()  # Get the current time after finishing the loop
    elapsed_time = end_time - start_time

    print(f"Time taken: {elapsed_time:.4f} seconds")


# driver code for question 1
def main():
    tag = tagging.Tagging("data/All_Images")
    results = tag.tag_images()
    create_boxed_images(results)


if __name__ == "__main__":
    main()
