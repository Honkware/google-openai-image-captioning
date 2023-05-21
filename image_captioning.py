import os
import sys
import argparse
import torch
import openai
from google.cloud import vision_v1p3beta1 as vision
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


def process_images(path, gcreds, oapikey):
    # Set Google Cloud credentials
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = gcreds

    # Set OpenAI API credentials
    openai.api_key = oapikey

    client = vision.ImageAnnotatorClient()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    gpt_cache = {}

    def describe_image(image_path):
        # Perform label detection using Google Cloud Vision API
        with open(image_path, 'rb') as image_file:
            response = client.label_detection(image=vision.Image(content=image_file.read()), max_results=20)
        labels = [label.description for label in response.label_annotations]
        labels_sentence = 'Image: ' + ', '.join(labels)
        prompt = f"Concise caption for this image: {labels_sentence}"

        if prompt not in gpt_cache:
            # Generate captions using OpenAI GPT-3.5 Turbo
            messages = [{"role": "system", "content": "Generate a concise and accurate dataset caption for images."},
                        {"role": "user", "content": prompt}]
            response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, max_tokens=100,
                                                    n=5, temperature=0.5)
            gpt_cache[prompt] = [choice.message['content'].strip() for choice in response.choices]

        descriptions = gpt_cache[prompt]

        with Image.open(image_path) as image:
            # Extract image features using CLIP model
            image_input = processor(images=image, return_tensors="pt", padding=True).to(device)
            image_embedding = model.get_image_features(**image_input)
            text_inputs = processor(text=descriptions, return_tensors="pt", padding=True).to(device)
            text_embeddings = model.get_text_features(**text_inputs)

            image_embedding_flat = image_embedding.cpu().detach().numpy().flatten()
            similarities = [1 - torch.cosine_similarity(image_embedding_flat, text_embedding.cpu().detach().numpy().flatten(), dim=0).item()
                            for text_embedding in text_embeddings]

        return descriptions[similarities.index(max(similarities))]

    # Get image files from the specified path
    image_files = [filename for filename in os.listdir(path) if filename.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for filename in image_files:
        image_path = os.path.join(path, filename)
        description = describe_image(image_path)
        print(f'Image: {filename}\nDescription:\n{description}\n---------------------')


def parse_arguments():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('path', type=str, help='Path to the folder containing images')
    parser.add_argument('--gcreds', type=str, help='Google Cloud credentials file path')
    parser.add_argument('--oapikey', type=str, help='OpenAI API key')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    process_images(args.path, args.gcreds, args.oapikey)
