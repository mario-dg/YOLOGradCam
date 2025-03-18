import gradio as gr
import tempfile
import torch
from heatmap import yolo_heatmap
from utils import *

# Add this function to process multiple CAM methods and backward types
def process_multiple_configs(
    image_input,
    weight_file,
    device,
    methods,  # List of selected methods
    layer_str,
    backward_types,  # List of selected backward types
    conf_threshold,
    ratio,
    show_result,
    renormalize,
    task,
    img_size
):
    # Handle case when no image is provided
    if image_input is None:
        return None, None, "No image provided. Please upload an image or select a sample."
    
    # Handle case when no methods are selected
    if not methods or len(methods) == 0:
        return None, None, "Please select at least one CAM method."
    
    # Handle case when no backward types are selected
    if not backward_types or len(backward_types) == 0:
        return None, None, "Please select at least one backward type."
    
    # Parse layers from string to list of integers
    try:
        layers = [int(l.strip()) for l in layer_str.split(',')]
    except:
        return None, None, "Error parsing layers. Use comma-separated integers."
    
    # Process the weight file
    if weight_file is None:
        # Try to use default model
        weight_path = download_default_model()
        if weight_path is None:
            return None, None, "No model file selected and default model download failed."
    else:
        weight_path = weight_file
    
    # Create temporary directory for output
    temp_dir = tempfile.mkdtemp()
    
    results = []
    result_labels = []
    errors = []
    
    # Process each combination of method and backward type
    for method in methods:
        for backward_type in backward_types:
            try:
                # Initialize the model for this configuration
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
                
                # Process the image
                result_image = model(image_input)
                
                if result_image is not None:
                    results.append(result_image)
                    result_labels.append(f"{method} - {backward_type}")
                else:
                    errors.append(f"Error processing image with {method} and {backward_type}.")
                    
            except Exception as e:
                import traceback
                error_msg = f"Error with {method}, {backward_type}: {str(e)}"
                print(error_msg)
                print(traceback.format_exc())
                errors.append(error_msg)
    
    if not results:
        return None, None, "Failed to generate any visualizations. Errors: " + " ".join(errors)
    
    return results, result_labels, ("Processing completed with some errors: " + " ".join(errors)) if errors else "Processing completed successfully."

# Add this function to format results for gallery display
def process_and_format_for_gallery(
    image_input, weight_file, device, methods, layer_str,
    backward_types, conf_threshold, ratio, show_result,
    renormalize, task, img_size
):
    results, labels, status = process_multiple_configs(
        image_input, weight_file, device, methods, layer_str,
        backward_types, conf_threshold, ratio, show_result,
        renormalize, task, img_size
    )
    
    if results is None:
        return None, status
    
    # Format for gallery - list of (image, label) tuples
    gallery_items = [(img, label) for img, label in zip(results, labels)]
    
    return gallery_items, status

# Modify the create_gradio_interface function to include comparison tab
def create_gradio_interface():
    # Pre-fetch sample images
    sample_images = get_sample_images()
    sample_image_names = list(sample_images.keys())
    
    # Pre-download default model
    default_model = download_default_model()
    
    with gr.Blocks(title="YOLO Heatmap Visualization") as app:
        gr.Markdown("# YOLO Heatmap Visualization Tool")
        
        with gr.Tabs():
            with gr.TabItem("Single Heatmap"):
                gr.Markdown("Upload an image and adjust parameters to generate a heatmap visualization for YOLO models.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # Image selection
                        with gr.Tab("Upload Image"):
                            input_image = gr.Image(label="Upload Input Image")
                        
                        with gr.Tab("Sample Images"):
                            sample_image_gallery = gr.Gallery(
                                label="Select a sample image",
                                value=[
                                    (sample_images[name], name) for name in sample_image_names if name in sample_images
                                ],
                                columns=3,
                                object_fit="contain",
                                height="auto"
                            )
                            selected_sample = gr.Textbox(visible=False)
                            
                            def select_sample(evt: gr.SelectData):
                                return sample_image_names[evt.index]
                            
                            sample_image_gallery.select(
                                select_sample,
                                outputs=selected_sample
                            )
                        
                        # Model selection
                        with gr.Group():
                            gr.Markdown("### Model Selection")
                            weight_file = gr.File(label="Upload Model Weight File (.pt)", file_types=[".pt"])
                            
                            if default_model:
                                gr.Markdown(f"Default model (yolov8n.pt) is available if no model is uploaded.")
                        
                        device = gr.Dropdown(
                            label="Device", 
                            choices=["cuda:0", "cuda:1", "cuda", "cpu"], 
                            value="cuda" if torch.cuda.is_available() else "cpu"
                        )
                        method = gr.Dropdown(
                            label="CAM Method",
                            choices=[
                                "GradCAMPlusPlus", "GradCAM", "XGradCAM", "EigenCAM", 
                                "HiResCAM", "LayerCAM", "RandomCAM", "EigenGradCAM", "KPCA_CAM"
                            ],
                            value="GradCAMPlusPlus"
                        )
                        layer_str = gr.Textbox(label="Layers (comma-separated)", value="10, 12, 14, 16, 18")
                        
                        task = gr.Dropdown(
                            label="Task",
                            choices=["detect", "segment", "pose", "obb", "classify"],
                            value="detect"
                        )
                        
                        backward_type_choices = {
                            "detect": ["class", "box", "all"],
                            "segment": ["class", "box", "segment", "all"],
                            "pose": ["class", "box", "pose", "all"],
                            "obb": ["class", "box", "obb", "all"],
                            "classify": ["all"]
                        }
                        
                        backward_type = gr.Dropdown(
                            label="Backward Type",
                            choices=backward_type_choices["detect"],
                            value="all"
                        )
                        
                        # Update backward_type choices when task changes
                        def update_backward_type(task_val):
                            return gr.Dropdown(choices=backward_type_choices[task_val])
                        
                        task.change(update_backward_type, inputs=[task], outputs=[backward_type])
                        
                        conf_threshold = gr.Slider(
                            label="Confidence Threshold",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.2,
                            step=0.05
                        )
                        
                        ratio = gr.Slider(
                            label="Ratio",
                            minimum=0.01,
                            maximum=1.0,
                            value=0.02,
                            step=0.01
                        )
                        
                        img_size = gr.Slider(
                            label="Image Size",
                            minimum=128,
                            maximum=1280,
                            value=640,
                            step=32
                        )
                        
                        show_result = gr.Checkbox(label="Show Detection Results", value=True)
                        renormalize = gr.Checkbox(label="Renormalize Heatmap in Bounding Boxes", value=False)
                        
                        process_btn = gr.Button("Generate Heatmap", variant="primary")
                        
                    with gr.Column(scale=2):
                        output_image = gr.Image(label="Heatmap Visualization")
                        status_text = gr.Textbox(label="Status", interactive=False)
                
                # Process button functionality
                process_btn.click(
                    fn=process_image,
                    inputs=[
                        input_image, weight_file, device, method, layer_str,
                        backward_type, conf_threshold, ratio, show_result,
                        renormalize, task, img_size
                    ],
                    outputs=[output_image, status_text]
                )
            
            # Add new comparison tab
            with gr.TabItem("Compare Heatmaps"):
                gr.Markdown("Compare multiple heatmap methods and backward types side by side.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # Image selection
                        with gr.Tab("Upload Image"):
                            comp_input_image = gr.Image(label="Upload Input Image")
                        
                        with gr.Tab("Sample Images"):
                            comp_sample_image_gallery = gr.Gallery(
                                label="Select a sample image",
                                value=[
                                    (sample_images[name], name) for name in sample_image_names if name in sample_images
                                ],
                                columns=3,
                                object_fit="contain",
                                height="auto"
                            )
                            comp_selected_sample = gr.Textbox(visible=False)
                            
                            def comp_select_sample(evt: gr.SelectData):
                                return sample_image_names[evt.index]
                            
                            comp_sample_image_gallery.select(
                                comp_select_sample,
                                outputs=comp_selected_sample
                            )
                        
                        # Model selection
                        with gr.Group():
                            gr.Markdown("### Model Selection")
                            comp_weight_file = gr.File(label="Upload Model Weight File (.pt)", file_types=[".pt"])
                            
                            if default_model:
                                gr.Markdown(f"Default model (yolov8n.pt) is available if no model is uploaded.")
                        
                        comp_device = gr.Dropdown(
                            label="Device", 
                            choices=["cuda:0", "cuda:1", "cuda", "cpu"], 
                            value="cuda" if torch.cuda.is_available() else "cpu"
                        )
                        
                        # Multi-select for methods
                        comp_methods = gr.Dropdown(
                            label="CAM Methods (Select multiple)",
                            choices=[
                                "GradCAMPlusPlus", "GradCAM", "XGradCAM", "EigenCAM", 
                                "HiResCAM", "LayerCAM", "RandomCAM", "EigenGradCAM", "KPCA_CAM"
                            ],
                            value=["GradCAMPlusPlus", "EigenCAM"],
                            multiselect=True
                        )
                        
                        comp_layer_str = gr.Textbox(label="Layers (comma-separated)", value="10, 12, 14, 16, 18")
                        
                        comp_task = gr.Dropdown(
                            label="Task",
                            choices=["detect", "segment", "pose", "obb", "classify"],
                            value="detect"
                        )
                        
                        # Multi-select for backward types
                        comp_backward_types = gr.Dropdown(
                            label="Backward Types (Select multiple)",
                            choices=backward_type_choices["detect"],
                            value=["class", "all"],
                            multiselect=True
                        )
                        
                        # Update backward_type choices when task changes
                        def update_comp_backward_types(task_val):
                            return gr.Dropdown(
                                choices=backward_type_choices[task_val],
                                multiselect=True
                            )
                        
                        comp_task.change(update_comp_backward_types, inputs=[comp_task], outputs=[comp_backward_types])
                        
                        comp_conf_threshold = gr.Slider(
                            label="Confidence Threshold",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.2,
                            step=0.05
                        )
                        
                        comp_ratio = gr.Slider(
                            label="Ratio",
                            minimum=0.01,
                            maximum=1.0,
                            value=0.02,
                            step=0.01
                        )
                        
                        comp_img_size = gr.Slider(
                            label="Image Size",
                            minimum=128,
                            maximum=1280,
                            value=640,
                            step=32
                        )
                        
                        comp_show_result = gr.Checkbox(label="Show Detection Results", value=True)
                        comp_renormalize = gr.Checkbox(label="Renormalize Heatmap in Bounding Boxes", value=False)
                        
                        compare_btn = gr.Button("Compare Heatmaps", variant="primary")
                        
                    with gr.Column(scale=2):
                        comp_output_gallery = gr.Gallery(
                            label="Heatmap Comparisons",
                            columns=2,
                            object_fit="contain",
                            height="auto"
                        )
                        comp_status_text = gr.Textbox(label="Status", interactive=False)
                
                # Function to handle sample image selection
                def use_sample_image(sample_name):
                    if not sample_name:
                        return None
                    
                    sample_images = get_sample_images()
                    if sample_name in sample_images:
                        return sample_images[sample_name]
                    return None
                
                # Connect sample image selection to input image for comparison tab
                comp_selected_sample.change(
                    use_sample_image,
                    inputs=[comp_selected_sample],
                    outputs=[comp_input_image]
                )
                
                # Compare button functionality
                compare_btn.click(
                    fn=process_and_format_for_gallery,
                    inputs=[
                        comp_input_image, comp_weight_file, comp_device, comp_methods, comp_layer_str,
                        comp_backward_types, comp_conf_threshold, comp_ratio, comp_show_result,
                        comp_renormalize, comp_task, comp_img_size
                    ],
                    outputs=[comp_output_gallery, comp_status_text]
                )
        
        # Function to handle sample image selection for both tabs
        def use_sample_image(sample_name):
            if not sample_name:
                return None
            
            sample_images = get_sample_images()
            if sample_name in sample_images:
                return sample_images[sample_name]
            return None
        
        # Connect sample image selection to input image for single heatmap tab
        selected_sample.change(
            use_sample_image,
            inputs=[selected_sample],
            outputs=[input_image]
        )
    
    return app

# Update the launch call in main to use the new interface
if __name__ == "__main__":
    app = create_gradio_interface()
    app.launch()
