# Google-OpenAI Image Captioning

A minimal and efficient image captioning tool using Google Cloud Vision API for label detection and OpenAI CLIP for image understanding.

## Setup

1. **Install required packages:**

```
pip install -r requirements.txt
```

Optionally, you can also install Gradio for a graphical user interface:

```
pip install gradio
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

## Optional: Gradio Interface

If you have installed Gradio, you can use the `gradio_gen.py` script to launch a web interface for the image captioning tool. Run the script with the following command:

```
python gradio_gen.py <google_credentials> <openai_api_key>
```

Replace `<google_credentials>` with the path to your Google Cloud JSON key file, and `<openai_api_key>` with your OpenAI API key.

To create a shareable link, add the `--share` option:

```
python gradio_gen.py <google_credentials> <openai_api_key> --share
```
