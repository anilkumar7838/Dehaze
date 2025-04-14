import os
import torch
from PIL import Image
import streamlit as st
from model import load_model
from runtestModel import RuntestModel
from io import BytesIO

# Paths
MODEL_PATH = "D:\\Dehaze\\trained Model\\Proposed_method_best.pth"  # Update with your model path
INPUT_FOLDER = "D:\\Dehaze\\test\\haze"  # Folder containing input hazy images
OUTPUT_FOLDER = "D:\\Dehaze\\test\\gt"  # Folder to save the dehazed images

# Make sure input/output folders exist
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load the model
@st.cache_resource
def load_dehazing_model():
    print("Loading model...")
    model = load_model(MODEL_PATH)
    model.eval()  # Set the model to evaluation mode
    return model

# Dehaze image using RuntestModel and save to output folder
def dehaze_image_and_save(model, uploaded_image):
    # Save the uploaded image in INPUT_FOLDER
    input_image_path = os.path.join(INPUT_FOLDER, "uploaded_image.png")
    uploaded_image.save(input_image_path)

    # Run the dehazing model using RuntestModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results= RuntestModel(model, INPUT_FOLDER, device)

    # Assuming results contain tuples (input_image, output_tensor)
    output_image = results[0][1]  # Get the dehazed image tensor
    output_image_pil = tensor_to_pil(output_image)  # Convert tensor to PIL image

    # Save the dehazed image in OUTPUT_FOLDER
    output_image_path = os.path.join(OUTPUT_FOLDER, "dehazed_image.png")
    output_image_pil.save(output_image_path)

    return uploaded_image, output_image_pil, output_image_path

# Function to convert tensor to PIL image
def tensor_to_pil(tensor):
    tensor = tensor.squeeze(0)  # Remove batch dimension
    tensor = tensor.permute(1, 2, 0)  # Change from CxHxW to HxWxC
    tensor = tensor.detach().cpu().clamp(0, 1)  # Ensure tensor values are between 0 and 1
    return Image.fromarray((tensor.numpy() * 255).astype('uint8'))  # Convert to PIL image

# Streamlit app UI
def main():
    st.title("Dehazing Image App")

    # Upload Image
    uploaded_file = st.file_uploader("Upload a hazy image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        uploaded_image = Image.open(uploaded_file)
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

        # Load the model
        model = load_dehazing_model()

        # Dehaze the image using RuntestModel and save to output folder
        input_image, output_image, output_path = dehaze_image_and_save(model, uploaded_image)

        # Display the dehazed image
        st.image(output_image, caption="Dehazed Image", use_container_width=True)

        # Download button for the dehazed image
        buffered = BytesIO()
        output_image.save(buffered, format="PNG")
        st.download_button(
            label="Download Dehazed Image",
            data=buffered.getvalue(),
            file_name="dehazed_image.png",
            mime="image/png"
        )

if __name__ == "__main__":
    main()
