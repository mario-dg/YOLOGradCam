import tempfile
import requests
from pathlib import Path
from heatmap import yolo_heatmap

# Download and cache sample images from Ultralytics
def get_sample_images():
    sample_images = {
        "zidane.jpg": "https://ultralytics.com/images/zidane.jpg",
        "bus.jpg": "https://ultralytics.com/images/bus.jpg",
    }
    
    cache_dir = Path("cache/sample_images")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = {}
    for name, url in sample_images.items():
        img_path = cache_dir / name
        if not img_path.exists():
            # Download the image
            try:
                response = requests.get(url, timeout=10)
                with open(img_path, "wb") as f:
                    f.write(response.content)
                print(f"Downloaded {name}")
            except Exception as e:
                print(f"Failed to download {name}: {e}")
                continue
        
        image_paths[name] = str(img_path)
    
    return image_paths

# Function to download default model
def download_default_model():
    cache_dir = Path("cache/models")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = cache_dir / "yolov8n.pt"
    
    if not model_path.exists():
        try:
            url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
            response = requests.get(url, timeout=30)
            with open(model_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded default model: yolov8n.pt")
        except Exception as e:
            print(f"Failed to download default model: {e}")
            return None
    
    return str(model_path)


# Define a function for the Gradio interface
def process_image(
    image_input,
    weight_file,
    device,
    method,
    layer_str,
    backward_type,
    conf_threshold,
    ratio,
    show_result,
    renormalize,
    task,
    img_size
):
    # Handle case when no image is provided
    if image_input is None:
        return None, "No image provided. Please upload an image or select a sample."
    
    # Parse layers from string to list of integers
    try:
        layers = [int(l.strip()) for l in layer_str.split(',')]
    except:
        return None, "Error parsing layers. Use comma-separated integers."
    
    # Process the weight file
    if weight_file is None:
        # Try to use default model
        weight_path = download_default_model()
        if weight_path is None:
            return None, "No model file selected and default model download failed."
    else:
        weight_path = weight_file
    
    # Create temporary directory for output
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Initialize the model
        model = yolo_heatmap(
            weight=weight_path,
            device=device,
            method=method,
            layer=layers,
            backward_type=backward_type,
            conf_threshold=conf_threshold,
            ratio=ratio,
            show_result=show_result,
            renormalize=renormalize,
            task=task,
            img_size=img_size
        )
        
        # Process the image - directly pass the image input
        # This can be a file path string or a loaded image (numpy array)
        result_image = model(image_input)
        
        if result_image is None:
            return None, "Error processing image."
            
        return result_image, "Processing completed successfully."
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg

# Function to load sample image
def load_sample_image(image_name):
    sample_images = get_sample_images()
    if image_name in sample_images:
        # Return the image path, not the loaded image
        return sample_images[image_name]
    return None
