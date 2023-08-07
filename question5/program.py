import os
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List, Dict

def tokenize_text(text: List[str], tokenizer):
    # Tokenize the input text using the provided tokenizer
    return tokenizer(text)

def cosine_similarity(a, b):
    '''
    In the code provided, the CLIP model from OpenAI is used to encode images and texts into high-dimensional feature vectors. 
    These feature vectors are essentially numerical representations of the images and texts in a multi-dimensional space. 
    The cosine similarity measures the cosine of the angle between these feature vectors, which provides a measure of their similarity.
    
    After processing the image and text through the CLIP model, we get the image feature vector and the text feature vector. 
    These vectors are represented as NumPy arrays.

    '''
    '''
    Cosine Similarity: The cosine similarity between two vectors, A and B, 
                        is calculated as the dot product of the vectors divided by the product of their magnitudes
    '''
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b) # Compute the cosine similarity between vectors a and b

    '''
    The cosine similarity ranges between -1 and 1. 
    A similarity of 1 means the vectors point in the same direction (i.e., they are identical or perfectly similar), 
    while a similarity of -1 means they point in completely opposite directions (i.e., they are completely dissimilar). 
    A similarity of 0 indicates orthogonality (i.e., they are independent or have no similarity).
    '''

def find_closest_matches(tagged_images_folder: str, all_images_folder: str):
    '''
    Find the top 3 most similar images cropped for each entity in the tagged images.

    Args:
        tagged_images_folder (str): Path to the folder containing tagged images with bounding boxes.
        all_images_folder (str): Path to the folder containing all the images.

    Returns:
        None
    '''
    
    # Load the CLIP model from OpenAI
    model, preprocess = clip.load("ViT-B/32", jit=False)

    # Create a simple tokenizer
    tokenizer = clip.simple_tokenizer.SimpleTokenizer()

    # Get a list of all the entity folders from the tagged images folder
    entity_folders = [folder for folder in os.listdir(tagged_images_folder) if os.path.isdir(os.path.join(tagged_images_folder, folder))]

    for entity_folder in entity_folders:
        entity_images_folder = os.path.join(tagged_images_folder, entity_folder)
        entity_cropped_folder = os.path.join(all_images_folder, entity_folder)

        # If the entity cropped folder doesn't exist, create it
        os.makedirs(entity_cropped_folder, exist_ok=True)

        # Get a list of all the images in the entity folder
        image_files = [file for file in os.listdir(entity_images_folder) if file.endswith((".jpg",".jpeg",".png"))]

        for image_file in tqdm(image_files):
            image_path = os.path.join(entity_images_folder, image_file)
            with Image.open(image_path) as image:
                # Get the tagged entity for the current image
                entity = os.path.splitext(image_file)[0]

                entity_similarities = []

                # Compare the current entity with entities in all other images using CLIP
                for root, _, files in os.walk(all_images_folder):
                    for file in files:
                        if file.endswith((".jpg",".jpeg","png")) and file != image_file:
                            img_path = os.path.join(root, file)
                            with Image.open(img_path) as entity_image:
                                # Tokenize the entity text
                                text = tokenize_text([entity], tokenizer)
                                # Preprocess the image
                                image_input = preprocess(entity_image).unsqueeze(0)

                                # Calculate similarity
                                image_features = model.encode_image(image_input)
                                text_features = model.encode_text(text)
                                similarity = cosine_similarity(image_features, text_features)
                                entity_similarities.append((img_path, similarity))

                # Sort the entities based on similarity and save the top-3 matches
                entity_similarities.sort(key=lambda x: x[1], reverse=True)
                for i, (img_path, _) in enumerate(entity_similarities[:3]):
                    top_match_img = Image.open(img_path)
                    top_match_img.save(os.path.join(entity_cropped_folder, f"top{i+1}-{entity}.jpg"))

if __name__ == "__main__":
    # Replace these with the tagged images folder and all images folder paths
    tagged_images_folder = r"C:\Users\amrit\Aryan\CS\python\Adobe hack-a-thon\Round 2 prep\Aithon\adobe_aithon_alpha\question1\boxed_images"
    all_images_folder = r"C:\Users\amrit\Aryan\CS\python\Adobe hack-a-thon\Round 2 prep\Aithon\adobe_aithon_alpha\All_Images"

    find_closest_matches(tagged_images_folder, all_images_folder)
