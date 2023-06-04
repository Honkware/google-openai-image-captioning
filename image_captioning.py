import os
import torch
from PIL import Image
import openai
from google.cloud import vision_v1p3beta1 as vision
from transformers import CLIPProcessor, CLIPModel

def describe_image(i, g, o):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = g
    c = vision.ImageAnnotatorClient()
    image = Image.open(i)
    d = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(d)
    p = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    r = c.label_detection(image=vision.Image(content=open(i, 'rb').read()), max_results=50)
    l = ' '.join([label.description for label in r.label_annotations])
    msgs = [
        {"role": "system", "content": "You are a creative assistant, skilled at crafting vivid and detailed descriptions."},
        {"role": "user", "content": f"Using your imaginative capabilities, describe this image as vividly and detailed as possible using these elements: {l}"}
    ]
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=msgs,
        max_tokens=100,
        n=1,
        stop=".",
        temperature=0.5,
        api_key=o,
    )
    pd = res.choices[0].message['content'].strip()
    inputs = p(images=image, text=pd, return_tensors="pt", padding="max_length", max_length=77)
    inputs = {k: v.to(d) for k, v in inputs.items()}
    outputs = m(**inputs)
    if_norm = torch.nn.functional.normalize(outputs[0], p=1, dim=-1)
    tf_norm = torch.nn.functional.normalize(outputs[1], p=1, dim=-1)
    ss = (if_norm @ tf_norm.T).squeeze()
    if ss.item() > 0.5:
        fd = pd
    else:
        msgs = [
            {"role": "system", "content": "You are a meticulous assistant, with the ability to refine descriptions into accurate representations."},
            {"role": "user", "content": f"The description is vivid and imaginative. Now, let's refine it to ensure it is an accurate representation of the image."}
        ]
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=msgs,
            max_tokens=100,
            n=1,
            stop=".",
            temperature=0.5,
            api_key=o,
        )
        fd = res.choices[0].message['content'].strip()
    return fd
