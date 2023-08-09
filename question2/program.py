import os.path
import pandas as pd
import tagging


def write_report(results):
    for result in results:

        # The results variable contains a list of classes of all images loaded.
        # The dictionary "count" is made to contain names of all entities present as keys
        # and their quantities in the image as values.
        count = {}

        for box in result.boxes:
            # class_id contains the name of the element that has been identified.
            class_id = result.names[box.cls[0].item()]

            # if the name of the element exists as a key in the dictionary,
            # increment the value by 1. if it does not, create that key with value 1
            if class_id in count:
                count[class_id] += 1
            else:
                count[class_id] = 1

        df = pd.DataFrame({"entity": [*count.keys()], "count": [*count.values()]})

        base_path = os.path.basename(result.path)
        base_name = base_path[:base_path.rindex('.')]

        if not os.path.exists(f"results/"):
            os.makedirs(f"results/")

        df.to_csv(f"results/{base_name}.csv", index=False)


def main() -> None:
    tag = tagging.Tagging("data/All_Images")
    results = tag.tag_images()

    write_report(results)


if __name__ == "__main__":
    main()
