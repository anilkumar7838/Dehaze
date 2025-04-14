import os
import torch
from PIL import Image
from model import load_model
from runtestModel import RuntestModel

# Paths
MODEL_PATH = "D:\\Dehaze\\trained Model\\Proposed_method_best.pth"  # Update with your model path
INPUT_FOLDER = "D:\\Dehaze\\test\\haze"  # Folder containing input hazy images
OUTPUT_FOLDER = "D:\\Dehaze\\test\\gt"  # Folder to save the dehazed images

def tensor_to_pil(tensor):
    # Convert tensor to a PIL image
    tensor = tensor.squeeze(0)  # Remove the batch dimension if present
    tensor = tensor.permute(1, 2, 0)  # Change from CxHxW to HxWxC
    tensor = tensor.detach().cpu().clamp(0, 1)  # Ensure tensor values are between 0 and 1
    return Image.fromarray((tensor.numpy() * 255).astype('uint8'))  # Convert to numpy and then to a PIL image

def main():
    # Load the model
    print("Loading model...")
    model = load_model(MODEL_PATH)
    model.eval()  # Set the model to evaluation mode

    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Check if the input folder exists
    if not os.path.exists(INPUT_FOLDER):
        raise FileNotFoundError(f"Input folder not found: {INPUT_FOLDER}")

    # Create the output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Run the model on the test dataset
    print("Running the dehazing model on test images...")
    results = RuntestModel(model, INPUT_FOLDER, device)

    # Save the results
    for i, (input_image, output_tensor) in enumerate(results):
        input_path = os.path.join(INPUT_FOLDER, f"input_{i + 1}.png")
        output_path = os.path.join(OUTPUT_FOLDER, f"dehazed_{i + 1}.png")

        try:
            # Convert tensor to PIL image
            output_image = tensor_to_pil(output_tensor)
            print(f"Saving dehazed image {i + 1}...")
            output_image.save(output_path)
        except Exception as e:
            print(f"Failed to save dehazed image {i + 1}: {e}")

    print("Dehazing completed for all images.")

if __name__ == "__main__":
    main()
