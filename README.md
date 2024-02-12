# Image Anonymizer

This application is designed to anonymize regions of interest, such as faces or license plates, in images. It utilizes YOLO (You Only Look Once) object detection models to detect objects in images and applies a Gaussian blur to anonymize them.

## How it Works

1. **Upload Image**: Users can upload an image containing regions of interest.
2. **Select Models**: Users can select the models for blurring.
3. **Detect Objects**: The application detects objects in the uploaded image using the selected YOLO object detection models.
4. **Anonymize**: Detected objects are anonymized by applying a Gaussian blur effect to the corresponding regions in the image.
5. **Download**: Users can download the anonymized image with the blurred regions.

## Usage

### Website

The application is publicly available at [Image Anonymizer](https://image-anonymizer.streamlit.app/). You can visit the website to blur regions of interest in images.

### Running Locally

To run the application locally, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/ippen/image-anonymizer.git
```

2. Navigate to the project directory:

```bash
cd image-anonymizer
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit application:

```bash
streamlit run image_anonymizer.py
```

5. Access the application in your web browser at `http://localhost:8501`.

## Components

### `image_anonymizer.py`

This Python script contains the main functionality of the application. It includes functions for detecting objects in images, anonymizing the detected objects, loading the YOLO models, and the main Streamlit application.

### `requirements.txt`

This file lists all the Python libraries and their versions required to run the application. You can install these dependencies using pip:

```bash
pip install -r requirements.txt
```

## Models

The YOLO models used for object detection are stored in `data/models/`. Users can select the models for blurring in the application interface.