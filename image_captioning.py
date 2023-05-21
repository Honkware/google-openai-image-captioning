import os
import torch
import openai
from google.cloud import vision_v1p3beta1 as vision
from transformers import CLIPProcessor, CLIPModel

def describe_image(image_path, google_credentials, openai_api_key):
    # Set Google Cloud credentials
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_credentials

    # Create a client for the Vision API
    client = vision.ImageAnnotatorClient()

    # Load the CLIP model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Perform label detection using Google Cloud Vision API
    response = client.label_detection(image=vision.Image(content=open(image_path, 'rb').read()), max_results=10)
    labels = [label.description for label in response.label_annotations]
    labels_sentence = 'Image: ' + ', '.join(labels)

    # Generate image description using GPT-3.5-turbo
    messages = [
        {"role": "user", "content": f"Caption image concisely: {labels_sentence}"}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=100,
        n=1,
        stop=".",
        temperature=0,
        api_key=openai_api_key,
    )

    description = response.choices[0].message['content'].strip()

    return description
