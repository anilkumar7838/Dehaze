import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def RuntestModel(model, test_data_hazy, device):
    # Verify the directory exists
    if not os.path.isdir(test_data_hazy):
        raise FileNotFoundError(f"Directory {test_data_hazy} does not exist")

    # Load hazy images from the directory
    hazy_image_paths = [os.path.join(test_data_hazy, img_name) for img_name in os.listdir(test_data_hazy)
                        if img_name.lower().endswith(('jpg', 'jpeg', 'png'))]
    
    if not hazy_image_paths:
        raise ValueError(f"No images found in the directory {test_data_hazy}")

    results = []

    # Define transforms for the images
    img_transforms = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images to 256x256
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize as needed
    ])

    model.eval()  # Set the model to evaluation mode
    model = model.to(device)  # Ensure the model is on the correct device

    for img_path in hazy_image_paths:
        # Load the hazy image
        img_hazy = Image.open(img_path).convert("RGB")

        # Apply transforms to the image
        img_hazy_transformed = img_transforms(img_hazy)

        # Add batch dimension
        img_hazy_transformed = img_hazy_transformed.unsqueeze(0).to(device)

        # Perform inference using the model
        with torch.no_grad():
            output = model(img_hazy_transformed)

        # Convert tensors to numpy arrays for visualization
        img_hazy_np = (img_hazy_transformed.squeeze().cpu().numpy().transpose(1, 2, 0) * 0.5) + 0.5  # De-normalize
        output_np = (output.squeeze().cpu().numpy().transpose(1, 2, 0) * 0.5) + 0.5  # De-normalize

        # Clip values for display
        img_hazy_np = img_hazy_np.clip(0, 1)
        output_np = output_np.clip(0, 1)

        # Display hazy and dehazed images
        # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        # axes[0].imshow(img_hazy_np)
        # axes[0].set_title('Hazy Image')
        # axes[0].axis('off')
        # axes[1].imshow(output_np)
        # axes[1].set_title('Dehazed Image')
        # axes[1].axis('off')
        # plt.show()

        # Append input and output tensors to results list
        results.append((img_hazy_transformed, output))

    return results
