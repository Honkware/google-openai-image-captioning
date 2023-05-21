import os
import sys
from image_captioning import describe_image

def main(folder_path, google_credentials, openai_api_key):
    # Process each image in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)

            # Describe the image
            description = describe_image(image_path, google_credentials, openai_api_key)

            # Print the description
            print(f'Image: {filename}')
            print('Description:')
            print(description)
            print('---------------------')

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python generate.py <folder_path> <google_credentials> <openai_api_key>")
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
