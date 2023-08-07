import os
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List, Dict
import concurrent.futures
import pickle

def tokenize_text(text: List[str], tokenizer):
    # Same as before
    pass

def cosine_similarity(a, b):
    # Same as before
    pass

def process_image(image_path, entity, tokenizer, preprocess, model):
    # Process a single image and calculate similarity
    with Image.open(image_path) as image:
        text = tokenize_text([entity], tokenizer)
        image_input = preprocess(image).unsqueeze(0)
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text)
        similarity = cosine_similarity(image_features, text_features)
        return image_path, similarity

def compute_embeddings(all_images_folder, model, tokenizer, preprocess):
    # Pre-compute embeddings for all images and cache them
    image_files = [file for file in os.listdir(all_images_folder) if file.endswith(".jpg")]

    embeddings = {}
    for image_file in tqdm(image_files):
        image_path = os.path.join(all_images_folder, image_file)
        with Image.open(image_path) as image:
            image_input = preprocess(image).unsqueeze(0)
            image_features = model.encode_image(image_input).detach().numpy()  # Use detach() before numpy()
            embeddings[image_file] = image_features

    with open('embeddings_cache.pkl', 'wb') as f:
        pickle.dump(embeddings, f)

def find_closest_matches(tagged_images_folder: str, all_images_folder: str):
    model, preprocess = clip.load("ViT-B/32", jit=False)
    tokenizer = clip.simple_tokenizer.SimpleTokenizer()

    # Check if embeddings are cached
    if os.path.exists('embeddings_cache.pkl'):
        with open('embeddings_cache.pkl', 'rb') as f:
            embeddings = pickle.load(f)
    else:
        compute_embeddings(all_images_folder, model, tokenizer, preprocess)
        with open('embeddings_cache.pkl', 'rb') as f:
            embeddings = pickle.load(f)

    entity_folders = [folder for folder in os.listdir(tagged_images_folder) if os.path.isdir(os.path.join(tagged_images_folder, folder))]

    for entity_folder in entity_folders:
        entity_images_folder = os.path.join(tagged_images_folder, entity_folder)
        entity_cropped_folder = os.path.join(all_images_folder, entity_folder)
        os.makedirs(entity_cropped_folder, exist_ok=True)

        image_files = [file for file in os.listdir(entity_images_folder) if file.endswith(".jpg")]

        # Initialize a dictionary to store similarity scores for each entity
        entity_similarities = {}

        # Process images and calculate similarity for each entity
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_path = {executor.submit(process_image, os.path.join(entity_images_folder, file), os.path.splitext(file)[0], tokenizer, preprocess, model): file for file in image_files}
            for future in tqdm(concurrent.futures.as_completed(future_to_path), total=len(future_to_path)):
                img_path, similarity = future.result()
                entity = os.path.splitext(os.path.basename(img_path))[0]

                # Add similarity score to the dictionary
                if entity not in entity_similarities:
                    entity_similarities[entity] = []
                entity_similarities[entity].append((img_path, similarity))

        # Save the top 3 similar images for each entity in its respective folder
        for entity, similarities in entity_similarities.items():
            # Sort similarities for the current entity
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Create a folder for the current entity if it doesn't exist
            entity_output_folder = os.path.join(entity_cropped_folder, entity)
            os.makedirs(entity_output_folder, exist_ok=True)

            # Save the top 3 similar images in the entity folder
            for i, (img_path, _) in enumerate(similarities[:3]):
                top_match_img = Image.open(img_path)
                top_match_img.save(os.path.join(entity_output_folder, f"top{i+1}-{os.path.basename(img_path)}"))


if __name__ == "__main__":
    tagged_images_folder = r"C:\Users\amrit\Aryan\CS\python\Adobe hack-a-thon\Round 2 prep\Aithon\adobe_aithon_alpha\question1\boxed_images"
    all_images_folder = r"C:\Users\amrit\Aryan\CS\python\Adobe hack-a-thon\Round 2 prep\Aithon\adobe_aithon_alpha\All_Images"

    find_closest_matches(tagged_images_folder, all_images_folder)




