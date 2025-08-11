import torch
import torchvision.transforms as transforms
from PIL import Image
from mntsr import MntsrModel
import os
from glob import glob

# Base paths based on script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # mntsr folder
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # final folder

# Load trained model
model_path = os.path.join(SCRIPT_DIR, "mntsr_trained.pth")
model = MntsrModel(channels=64, num_srb=3, upscale_factor=2)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True), strict=False)
model.eval()  # Set model to evaluation mode

# Define input and output folders
input_folder = os.path.join(BASE_DIR, "craft_result")  # Folder containing CRAFT output images
output_folder = os.path.join(BASE_DIR, "mntsr_result")  # Folder to save deblurred images

print(f"Script directory: {SCRIPT_DIR}")
print(f"Base directory: {BASE_DIR}")
print(f"Input folder: {input_folder}")
print(f"Output folder: {output_folder}")
print(f"Model path: {model_path}")

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get all image paths from the input folder
image_paths = glob(os.path.join(input_folder, "*.png"))
print(f"Found {len(image_paths)} images to process")

# Special case: If the input folder contains a 'detected_text.png' file, prioritize it
detected_text_path = os.path.join(input_folder, "detected_text.png")
if os.path.exists(detected_text_path):
    print(f"Found detected_text.png - prioritizing this file")
    image_paths = [detected_text_path] + [p for p in image_paths if p != detected_text_path]

if len(image_paths) == 0:
    print("No images found! Check input directory.")
    # Save an empty placeholder if no images found
    placeholder = Image.new('RGB', (100, 100), (255, 255, 255))
    placeholder.save(os.path.join(output_folder, "enhanced_image.png"))
    print("Created placeholder image")
    exit(0)

# Process each image in the folder
for image_path in image_paths:
    try:
        # Load the image
        image = Image.open(image_path)
        print(f"Processing: {os.path.basename(image_path)} | Size: {image.size} | Mode: {image.mode}")

        # Convert Grayscale to RGB if Needed
        if image.mode != "RGB":
            print(f"Converting grayscale image to RGB: {os.path.basename(image_path)}")
            image = image.convert("RGB")

        # Transform Image for Model
        transform = transforms.Compose([
            transforms.ToTensor()  # Convert image to tensor (C, H, W) format
        ])
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension (B, C, H, W)

        print(f"Input Tensor Shape: {input_tensor.shape}")

        # Pass through Model
        with torch.no_grad():
            output_tensor = model(input_tensor)  # First upscaling
            output_tensor = torch.clamp(output_tensor, 0, 1)  # Ensure pixel range is [0,1]
        
        # Convert Output to Image
        output_image = transforms.ToPILImage()(output_tensor.squeeze(0))

        # Save High-Resolution Output
        base_name = os.path.basename(image_path)
        output_filename = os.path.join(output_folder, f"deblurred_{base_name}")
        output_image.save(output_filename)
        print(f"Saved deblurred image: {output_filename}")
        
        # If this is the detected_text.png file, save a specially named version for OCR
        if base_name == "detected_text.png":
            special_output = os.path.join(output_folder, "deblurred_detected_text.png")
            output_image.save(special_output)
            print(f"Saved special OCR image: {special_output}")

    except Exception as e:
        print(f"Error processing {os.path.basename(image_path)}: {e}")
        import traceback
        print(traceback.format_exc())

# Create a combined/enhanced image for OCR
try:
    # Use the first processed image as the enhanced image or combine them
    if len(image_paths) > 0:
        # Try to use deblurred_detected_text.png if it exists
        special_output = os.path.join(output_folder, "deblurred_detected_text.png")
        if os.path.exists(special_output):
            enhanced_image = Image.open(special_output)
            enhanced_path = os.path.join(output_folder, "enhanced_image.png")
            enhanced_image.save(enhanced_path)
            print(f"Saved enhanced image from detected_text: {enhanced_path}")
        else:
            # Otherwise use first enhanced image
            first_output = os.path.join(output_folder, f"deblurred_{os.path.basename(image_paths[0])}")
            if os.path.exists(first_output):
                enhanced_image = Image.open(first_output)
                enhanced_path = os.path.join(output_folder, "enhanced_image.png")
                enhanced_image.save(enhanced_path)
                print(f"Saved enhanced image: {enhanced_path}")
            else:
                print(f"Warning: Could not find first output image at {first_output}")
except Exception as e:
    print(f"Error creating enhanced image: {e}")

print("MNTSR processing complete!")