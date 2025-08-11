import torch
import cv2
import numpy as np
from craft import CRAFT  # Import CRAFT
from collections import OrderedDict
import os
from sklearn.cluster import DBSCAN  # For clustering text regions
import sys
import io
import shutil

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Define copy_state_dict function
def copy_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # Remove 'module.' prefix if present
        new_state_dict[name] = v
    return new_state_dict

# Set model path
model_path = r"C:\Users\bindh\OneDrive\Desktop\final\craft\craft_mlt_25k (1).pth"

# Get the parent directory (final folder)
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Set the result folder to craft_result inside "final"
result_folder = os.path.join(parent_folder, "craft_result")

# Define the additional result folder
additional_result_folder = os.path.join(parent_folder, "final", "craft_result")
os.makedirs(additional_result_folder, exist_ok=True)

# Ensure folder exists
os.makedirs(result_folder, exist_ok=True)

print(f"Parent folder: {parent_folder}")
print(f"Result folder: {result_folder}")
print(f"Additional result folder: {additional_result_folder}")
print(f"Model path: {model_path}")
print(f"Model exists: {os.path.exists(model_path)}")

# Load CRAFT model
def load_craft_model(model_path):
    try:
        print("Loading CRAFT model...")
        net = CRAFT()  # Initialize model
        print("CRAFT initialized")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return None
            
        # Load model weights
        print("Loading model weights...")
        state_dict = torch.load(model_path, map_location="cpu")
        print("Model weights loaded")
        
        # Copy state dict
        state_dict = copy_state_dict(state_dict)
        print("State dict copied")
        
        # Load state dict into model
        net.load_state_dict(state_dict)
        print("State dict loaded into model")
        
        # Set model to evaluation mode
        net.eval()
        print("Model set to evaluation mode")
        
        return net
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Load the model
print("Attempting to load the model...")
craft_model = load_craft_model(model_path)
if craft_model is None:
    print("Failed to load the model. Exiting.")
    exit(1)
print("Model loaded successfully")

# Function to merge bounding boxes in a group
def merge_boxes_in_group(boxes):
    """
    Merge bounding boxes in a group into a single box.
    :param boxes: List of bounding boxes [x_min, y_min, x_max, y_max].
    :return: A single bounding box [x_min, y_min, x_max, y_max].
    """
    if not boxes:
        return None

    # Find the outer bounds of all boxes
    x_min = min([box[0] for box in boxes])
    y_min = min([box[1] for box in boxes])
    x_max = max([box[2] for box in boxes])
    y_max = max([box[3] for box in boxes])

    return [x_min, y_min, x_max, y_max]

# Function to expand bounding boxes
def expand_boxes(boxes, expand_pixels=10):
    """
    Expand bounding boxes by a specified number of pixels.
    :param boxes: List of bounding boxes [x_min, y_min, x_max, y_max].
    :param expand_pixels: Number of pixels to expand the boxes in all directions.
    :return: List of expanded bounding boxes.
    """
    expanded_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        # Expand the box
        x_min = max(0, x_min - expand_pixels)  # Ensure the box doesn't go out of the image
        y_min = max(0, y_min - expand_pixels)
        x_max = x_max + expand_pixels
        y_max = y_max + expand_pixels
        expanded_boxes.append([x_min, y_min, x_max, y_max])
    return expanded_boxes

# Function to group bounding boxes using DBSCAN
def group_bounding_boxes(boxes, eps=50, min_samples=1):
    """
    Group bounding boxes using DBSCAN clustering.
    :param boxes: List of bounding boxes [x_min, y_min, x_max, y_max].
    :param eps: Maximum distance between two samples for them to be considered in the same neighborhood.
    :param min_samples: Minimum number of samples in a neighborhood for a point to be considered a core point.
    :return: List of grouped bounding boxes.
    """
    if not boxes:
        return []

    # Calculate the center of each bounding box
    centers = np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in boxes])

    # Use DBSCAN to cluster the centers
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(centers)
    labels = db.labels_

    # Group boxes based on labels
    groups = {}
    for i, label in enumerate(labels):
        if label not in groups:
            groups[label] = []
        groups[label].append(boxes[i])

    # Merge boxes in each group
    grouped_boxes = []
    for label in groups:
        if label == -1:  # Noise points (not part of any cluster)
            for box in groups[label]:
                grouped_boxes.append(box)  # Add individual boxes as-is
        else:
            merged_box = merge_boxes_in_group(groups[label])
            grouped_boxes.append(merged_box)

    return grouped_boxes

# Process an image
def process_image(image_path):
    print(f"Processing image: {image_path}")
    print(f"Image exists: {os.path.exists(image_path)}")
    
    # Load image
    image = cv2.imread(image_path)

    if image is None:
        print(f" ERROR: Unable to load image from {image_path}. Check the path!")
        return None
    else:
        print(f"Image loaded successfully: {image.shape}")

    orig_image = image.copy()  # Keep a copy for visualization
    
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print("Image converted to RGB")
        
        image = cv2.resize(image, (1280, 720))  # Resize for uniformity
        print("Image resized")
        
        image = np.transpose(image, (2, 0, 1))  # Change to CxHxW format
        print("Image transposed")
        
        image = torch.from_numpy(image).float().unsqueeze(0)  # Add batch dimension
        print("Image converted to tensor")
        
        image = image / 255.0  # Normalize
        print("Image normalized")

        # Perform text detection
        print("Performing text detection...")
        with torch.no_grad():
            y, _ = craft_model(image)
        print("Text detection completed")

        print(f"Detection output shape: {y.shape}")  # Debugging output

        # Convert output to numpy array
        score_text = y[0, :, :, 0].cpu().numpy()
        print("Score text extracted")
        
        score_text = cv2.resize(score_text, (orig_image.shape[1], orig_image.shape[0]))  # Resize back to original size
        print("Score text resized")

        # Apply threshold to get detected regions
        _, score_text_bin = cv2.threshold(score_text, 0.5, 255, cv2.THRESH_BINARY)
        score_text_bin = score_text_bin.astype(np.uint8)
        print("Threshold applied")

        # Find contours
        contours, _ = cv2.findContours(score_text_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Found {len(contours)} contours")

        # Draw bounding boxes on the original image
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append([x, y, x + w, y + h])  # Store bounding box coordinates as [x_min, y_min, x_max, y_max]

        print(f"Created {len(boxes)} bounding boxes")

        # Expand bounding boxes to include full word edges
        expanded_boxes = expand_boxes(boxes, expand_pixels=10)  # Adjust expand_pixels as needed
        print(f"Expanded {len(expanded_boxes)} bounding boxes")

        # Group bounding boxes into text regions
        grouped_boxes = group_bounding_boxes(expanded_boxes, eps=50, min_samples=1)
        print(f"Grouped into {len(grouped_boxes)} text regions")

        # Draw grouped bounding boxes on the original image
        for box in grouped_boxes:
            x_min, y_min, x_max, y_max = box
            cv2.rectangle(orig_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Save output image with bounding boxes
        output_path = os.path.join(result_folder, "detected_text.png")
        print(f"Saving output image to: {output_path}")
        success = cv2.imwrite(output_path, orig_image)

        if success:
            print(f" Processed image saved at: {output_path}")
        else:
            print(f" ERROR: Failed to save image at {output_path}. Check folder permissions!")

        return grouped_boxes
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Save cropped text regions
def save_cropped_text_regions(image_path, boxes, output_folder):
    print(f"Saving cropped text regions to: {output_folder}")
    
    # Load the original image
    image = cv2.imread(image_path)
    if image is None:
        print(f" ERROR: Unable to load image from {image_path}. Check the path!")
        return

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Crop and save each text region
    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = box
        cropped_text = image[y_min:y_max, x_min:x_max]  # Crop the text region

        # Save the cropped text region
        output_path = os.path.join(output_folder, f"text_region_{i}.png")
        success = cv2.imwrite(output_path, cropped_text)

        if success:
            print(f" Cropped text region {i} saved at: {output_path}")
        else:
            print(f" ERROR: Failed to save cropped text region {i} at {output_path}.")

    print(f" Saved {len(boxes)} cropped text regions in '{output_folder}'.")

# Function to copy files from one directory to another
def copy_results(src_folder, dest_folder):
    if not os.path.exists(src_folder):
        print(f"Source folder does not exist: {src_folder}")
        return
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder, exist_ok=True)
    
    for filename in os.listdir(src_folder):
        src_file = os.path.join(src_folder, filename)
        dest_file = os.path.join(dest_folder, filename)
        shutil.copy(src_file, dest_file)
        print(f"Copied {filename} to {dest_folder}")

# Main execution
if __name__ == "__main__":
    try:
        # Use the path to the image in the parent directory
        image_path = os.path.join(parent_folder, "input_image.png")
        print(f"Looking for image at: {image_path}")
        
        if not os.path.exists(image_path):
            print(f" ERROR: Input image not found at {image_path}")
            # Try alternate location
            alt_image_path = os.path.join(os.path.dirname(__file__), "input_image.png")
            print(f"Trying alternate path: {alt_image_path}")
            if os.path.exists(alt_image_path):
                image_path = alt_image_path
                print(f"Using alternate image path: {image_path}")
            else:
                print(" ERROR: Could not find input image in any location")
                exit(1)
        
        # Process the image
        grouped_boxes = process_image(image_path)

        if grouped_boxes is not None and len(grouped_boxes) > 0:
            print(f" Detected {len(grouped_boxes)} text regions.")
            save_cropped_text_regions(image_path, grouped_boxes, output_folder=result_folder)

            # Copy results to the additional result folder
            copy_results(result_folder, additional_result_folder)

            print(" Processing completed successfully")
        else:
            print(" No text regions detected or processing failed")
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)