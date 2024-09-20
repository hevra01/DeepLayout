import json
import os
import numpy as np
from utils import trim_tokens


# Function to convert a tensor for one image into a structured JSON with reshaping and bounding box normalization
def tensor_to_json(image_tensor, dataset, width=512, height=512):
    # Initialize the JSON structure
    json_structure = {
        "full_prompt": "",
        "caption": "",
        "width": width,
        "height": height,
        "bg_prompt": "",
        "annos": []
    }

    # Trim tokens (remove special tokens like BOS, EOS, PAD)
    image_tensor = trim_tokens(image_tensor, dataset.bos_token, dataset.eos_token, dataset.pad_token)
    
    # Reshape the tensor to get [category_id, x, y, width, height] per object
    image_tensor = image_tensor[: len(image_tensor) // 5 * 5].reshape(-1, 5)

    # Initialize the list to store annotations
    annos = []

    # Extract bounding box coordinates
    box = image_tensor[:, 1:].astype(np.float32)
    
    # Normalize bounding boxes:
    # Scale x, y to be within the image dimensions
    box[:, [0, 1]] = box[:, [0, 1]] / (dataset.size - 1) * width
    
    # Scale width and height and adjust to image dimensions
    box[:, [2, 3]] = (box[:, [2, 3]] - 1) / dataset.size * height
    

    # Iterate over the objects in the reshaped tensor
    for i in range(len(image_tensor)):
        category_id = image_tensor[i][0]  # First value is the category_id

        # Extract bounding box (x1, y1, x2, y2)
        x, y, w, h = box[i]

        # check if category_id is within the range of COCO categories
        if category_id in dataset.contiguous_category_id_to_json_id:
            category_id = dataset.contiguous_category_id_to_json_id[category_id]
            caption = dataset.categories[category_id]["name"]
        else:
            caption = "unknown"

        # Create the annotation object
        annotation = {
            "bbox": [float(x), float(y), float(w), float(h)],  # Convert x2, y2 back to width, height for JSON format
            "mask": [],
            "category_name": "",  # Empty category name for now
            "caption": caption,
            "certainties": 0.0  # Placeholder for certainties
        }

        # Add annotation to the list of annotations
        annos.append(annotation)

    # Assign annotations to the JSON structure
    json_structure["annos"] = annos

    return json_structure


# Function to convert all image tensors and save as JSON
def process_all_images(tensors, dataset, output_dir):
    # Iterate over all image tensors
    for i, image_tensor in enumerate(tensors):
        # Process each image tensor and generate JSON
        json_data = tensor_to_json(image_tensor, dataset, width=512, height=512)

        # Create the directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
        # Save the JSON data to a file
        output_file = f"{output_dir}/{i}.json"
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=4)
        print(f"Saved JSON for image {i} to {output_file}")