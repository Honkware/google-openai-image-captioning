import gradio as gr
import argparse
from image_captioning import describe_image
from PIL import Image

def caption_image(image, google_credentials, openai_api_key):
    pil_image = Image.fromarray(image)
    pil_image.save("image.jpg")
    description = describe_image("image.jpg", google_credentials, openai_api_key)
    return description

def main(google_credentials, openai_api_key, share):
    iface = gr.Interface(
        fn=lambda image: caption_image(image, google_credentials, openai_api_key),
        inputs=gr.components.Image(label="Upload Image"),
        outputs=gr.components.Textbox(label="Image Description"), 
        title="Google-OpenAI Image Captioning",
        description="An image captioning tool using Google Cloud Vision API for label detection and OpenAI CLIP for image understanding.",
    )

    iface.launch(share=share)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Google-OpenAI Image Captioning")
    parser.add_argument("google_credentials", help="Google credentials")
    parser.add_argument("openai_api_key", help="OpenAI API key")
    parser.add_argument("--share", action="store_true", help="Create a shareable link")

    args = parser.parse_args()

    main(args.google_credentials, args.openai_api_key, args.share)
