# Google-OpenAI Image Captioning

A minimal and efficient image captioning tool using Google Cloud Vision API for label detection and OpenAI CLIP for image understanding.

## Setup

1. **Install required packages:**

```
pip install -r requirements.txt
```

2. **Set up Google Cloud Vision API credentials:**

- Create a Google Cloud project and enable the Vision API by following the instructions [here](https://cloud.google.com/vision/docs/before-you-begin).
- Download the JSON key file for your service account and save it to your local machine.

3. **Get an OpenAI API key:**

- Sign up for an OpenAI account [here](https://beta.openai.com/signup/).
- Obtain your API key from the OpenAI [dashboard](https://beta.openai.com/dashboard/).

## Usage

Run the script with the following command:

```
python generate.py <folder_path> <google_credentials> <openai_api_key>
```

Replace `<folder_path>` with the path to the folder containing your images, `<google_credentials>` with the path to your Google Cloud JSON key file, and `<openai_api_key>` with your OpenAI API key.

The script will process each image in the folder and print a concise description for each image.

## Example Output

```
Image: primate.jpeg
Description: A primate observes a plant in its natural environment, surrounded by other organisms and terrestrial plants
---------------------
```
