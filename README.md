# adobe_aithon_alpha
Repository for Binary Bots Team Alpha's codebase for the Adobe AI-Thon 2023

# Problem Statements

1.	Given an image, find the entities in the image and mark their position and likeliness (using the YOLOv8 model).
2.	Count number of entities and group them by name. Report the count for each entity.
3.	Group a set of images based on the entities found in them.
4.	Given an example set of ideal advertisement images. Find the missing or extra entities in other images.
5.	For each entity in every image find the closest matching entities from all the images. (Crop out the matching entities and put one source entity in the folder, rest matching images in its sub folders) for all images.

Create an express page and express your learnings.

# Structure
- `tagging`: code related to tagging entities, shared across as a module to be used
- `reporting`: code related to report generation via txt files and/or CSV files.
- `question<N>`: application code for each of the given question where N is the question number

# Tools
- YOLOv8 model -- [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- OpenAI CLIP -- [openai/CLIP](https://github.com/openai/CLIP)
