import streamlit as st
from PIL import Image, ImageFilter, ImageDraw
import io
from ultralytics import YOLO

@st.cache_data
def get_detections(image_file: str, _models: dict, selected_model, device="cpu"):
    """
    Get the detections (bounding boxes) from the YOLO models.

    Args:
        image_file (str): Path to the input image file.
        models (dict): Dictionary containing YOLO models.
        selected_model (str): Name of the selected model. Argument just to trigger the cache when the selected model changes.
        device (str): Device to run the models on.

    Returns:
        dict: Dictionary containing detections for each model.
    """
    image = Image.open(image_file)
    detections = {}
    for model_name, model in _models.items():
        # Perform inference
        results = model.predict(image, imgsz=640, device=device, verbose=False)
        detections[model_name] = results[0].boxes.data
    return detections


def anonymize_image(image: Image.Image, detections: dict, blur_radius: float, blur_shapes: dict):
    """
    Anonymize the regions in the input image by blurring the regions defined by bounding boxes.

    Args:
        image (PIL.Image.Image): Input image.
        detections (dict): Dictionary containing bounding box coordinates and labels for each model.
        blur_radius (float): Radius of the Gaussian blur filter.
        blur_shapes (dict): Dictionary containing the shape of blur effect for each model.

    Returns:
        PIL.Image.Image: Anonymized image.
    """
    for model_name, model_detections in detections.items():
        blur_shape = blur_shapes.get(model_name, "rectangle")  # Default to rectangle if not specified
        for bbox in model_detections:
            x1, y1, x2, y2 = map(int, bbox[:4])
            if blur_shape == "ellipse":
                mask = Image.new('L', image.size, 0)
                draw = ImageDraw.Draw(mask)
                draw.ellipse((x1, y1, x2, y2), fill=255)
                blurred = image.filter(ImageFilter.GaussianBlur(blur_radius))
                image.paste(blurred, mask=mask)
            else:  # rectangle
                region = image.crop((x1, y1, x2, y2))
                blurred_region = region.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                image.paste(blurred_region, (x1, y1, x2, y2))
    return image

@st.cache_resource
def load_models(selected_models):
    """
    Load the YOLO models from the checkpoints.

    Returns:
        dict: Dictionary containing YOLO models.
    """
    models = {}
    for model_name in selected_models:
        if model_name == 'faces':
            model_path = "data/models/face/v8n_face_v1.pt"
        elif model_name == 'licence plates':
            model_path = "data/models/lp/v8n_lp_v1.pt"
        else:
            raise ValueError("Invalid model name")
        models[model_name] = YOLO(model_path)
    return models

def main():
    st.title("Image Anonymizer")
    st.write("Easily anonymize regions of interest in images by uploading them and adjusting the blur strength slider. " +
             "Models for blurring can be individually selected, allowing you to blur faces, license plates, or both. " +
             "Anonymized images can be downloaded via the 'Download Anonymized Image' button.")
    st.markdown(
        """
        <span style='font-size: 16px; padding-right: 10px; vertical-align: middle;'>Made by Lars Ippen</span>
        [<img src="https://upload.wikimedia.org/wikipedia/commons/8/81/LinkedIn_icon.svg" width=30 height=30 style='background-color: white; padding: 3px; border-radius: 3px;'>](https://www.linkedin.com/in/lars-ippen)
        <span style='padding-right: 4px;'></span>
        [<img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width=30 height=30 style='background-color: white; padding: 3px; border-radius: 3px;'>](https://github.com/ippen)
        """, unsafe_allow_html=True
    )

    # Select the models for blurring
    selected_models = st.multiselect("Select models for blurring", ['faces', 'licence plates'], default=['faces', 'licence plates'])

    # Display a file uploader widget
    uploaded_file = st.file_uploader("Choose a file", type=["jpeg", "jpg", "png", "webp"])

    blur_strength = st.slider("Blur Strength", min_value=0.0, max_value=100.0, value=10.0, step=0.1, format="%.1f", help="Set the strength of the blur effect. 0: no blur, 100: maximum blur.")

    if uploaded_file:
        # Load YOLO models
        yolo_models = load_models(selected_models)

        # Always recalculate detections for a new uploaded image
        with st.spinner('Detecting objects...'):
            # Get detections
            detections = get_detections(uploaded_file, yolo_models, selected_models)

        with st.spinner('Anonymizing...'):
            # Define blur shapes for each model
            blur_shapes = {'faces': 'ellipse', 'licence plates': 'rectangle'}

            # Anonymize regions and dynamically set blur strength
            image = Image.open(uploaded_file)
            anonymized_image = anonymize_image(image, detections, blur_strength, blur_shapes)

        # Display the anonymized image
        st.image(anonymized_image, caption="Anonymized Image", use_column_width=True)

        # Download button for the anonymized image
        buffered = io.BytesIO()
        anonymized_image.save(buffered, format="PNG")
        anonymized_image_data = buffered.getvalue()
        uploaded_file_name = uploaded_file.name.split(".")[0]
        st.download_button(
            label="Download Anonymized Image",
            data=anonymized_image_data,
            file_name=uploaded_file_name+"_anonymized.png",
            mime="image/png",
        )

if __name__ == "__main__":
    main()
