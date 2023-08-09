import math
import os

import pandas as pd
from PIL import Image


class Interestingness:

    @staticmethod
    def _calculate_interestingness_score(box, image_shape):
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        conf = round(box.conf[0].item(), 2)

        y, x = image_shape
        x0, y0 = abs(cords[2] - cords[0]), abs(cords[3] - cords[1])

        x_dist = ((y - x0) / 2) ** 2
        y_dist = ((y - y0) / 2) ** 2

        d = x_dist + y_dist
        scored_dist = 100 * math.exp(-d)

        area_of_img = x0 * y0
        scored_area = 100 * (area_of_img / (x * y))

        scored_area_dist = 50 * math.sin((scored_area + scored_dist) / 20)

        return 200 * conf + scored_area_dist

    def process_images(self, results):
        rows, cols = len(results[0].names), len(results)
        data = [[[] for _ in range(cols)] for _ in range(rows)]
        df = pd.DataFrame(data)

        for i in range(cols):
            result = results[i]

            for box in result.boxes:
                int_score = self._calculate_interestingness_score(box, result.orig_shape)
                df.iat[int(box.cls[0].item()), i].append(int_score)

            for j in range(rows):
                if not sum(df.iloc[j, i]) == 0:
                    df.iloc[j, i] = float(max(df.iloc[j, i])) + float(1.5 * sum(df.iloc[j, i]) / (len(df.iloc[j, i])))
                else:
                    df.iloc[j, i] = float(0)

            df[i] = pd.to_numeric(df[i])
            res_dir = result.names[df[i].argmax()]

            if not os.path.exists(f"results/{res_dir}/"):
                os.makedirs(f"results/{res_dir}/")

            im = Image.open(f"{result.path}")
            im.save(f"results/{res_dir}/{res_dir}_{os.path.basename(result.path)}")

        return df
