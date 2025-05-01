import os
import io
import base64
import numpy as np
from PIL import Image
import tifffile
import torch
from torchvision.utils import save_image
from torchvision import transforms
import streamlit as st
from src.ML_TEMPERATURE_PREDICTION.components.pix2pix_model import Generator
from src.ML_TEMPERATURE_PREDICTION.logging import logger
from src.ML_TEMPERATURE_PREDICTION.config.configuration import ConfigurationManager
from src.ML_TEMPERATURE_PREDICTION.utils.common import read_yaml

# Set page configuration
st.set_page_config(
    page_title="RGB-Thermal Image Translation",
    page_icon="🔥",
    layout="wide"
)

def load_model(model_path, direction):
    """
    Load the trained generator model
    
    Args:
        model_path: Path to the model checkpoint
        direction: Direction of translation (rgb2thermal or thermal2rgb)
        
    Returns:
        Generator model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set input and output channels based on direction
    if direction == 'rgb2thermal':
        input_channels = 3
        output_channels = 1
    else:  # thermal2rgb
        input_channels = 1
        output_channels = 3
    
    # Initialize model
    model = Generator(in_channels=input_channels, out_channels=output_channels).to(device)
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['generator'])
    model.eval()
    
    return model, device

def process_image(uploaded_file, model, device, direction, image_size):
    """
    Process the uploaded image using the model
    
    Args:
        uploaded_file: Uploaded file from Streamlit
        model: Generator model
        device: Device to run the model on
        direction: Direction of translation
        image_size: Size to resize the image to
        
    Returns:
        Input image, output image
    """
    # Read uploaded file
    if direction == 'rgb2thermal':
        # Process RGB image
        input_image = Image.open(uploaded_file).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:  # thermal2rgb
        # Process thermal image (16-bit TIFF)
        thermal_data = tifffile.imread(uploaded_file)
        
        # Normalize thermal image to 0-1 range
        thermal_data = thermal_data.astype(np.float32)
        thermal_min = np.min(thermal_data)
        thermal_max = np.max(thermal_data)
        if thermal_max > thermal_min:
            thermal_data = (thermal_data - thermal_min) / (thermal_max - thermal_min)
        else:
            thermal_data = np.zeros_like(thermal_data)
        
        # Convert to PIL image for transforms
        input_image = Image.fromarray((thermal_data * 255).astype(np.uint8))
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    # Apply transforms and add batch dimension
    input_tensor = transform(input_image).unsqueeze(0).to(device)
    
    # Generate output
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    # Convert tensors to PIL images for display
    # Denormalize from [-1, 1] to [0, 1]
    input_display = (input_tensor[0].cpu() + 1) / 2
    output_display = (output_tensor[0].cpu() + 1) / 2
    
    return input_display, output_display

def main():
    st.title("RGB-Thermal Image Translation")
    st.write("Upload an image to translate between RGB and thermal domains using Pix2Pix")
    
    # Load configuration
    try:
        config_manager = ConfigurationManager()
        params = read_yaml("params.yaml")
        direction = params.DIRECTION
        image_size = params.IMAGE_SIZE
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        return
    
    # Direction selection
    direction = st.radio(
        "Select translation direction:",
        ["rgb2thermal", "thermal2rgb"],
        index=0 if direction == "rgb2thermal" else 1
    )
    
    # Model path
    model_path = os.path.join("artifacts", "model_trainer", "checkpoints", "best.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join("artifacts", "model_trainer", "checkpoints", "latest.pth")
    
    if not os.path.exists(model_path):
        st.error("No trained model found. Please train the model first.")
        return
    
    # Load model
    try:
        model, device = load_model(model_path, direction)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
    
    # File uploader
    file_help = "Upload a RGB image (JPG/PNG)" if direction == "rgb2thermal" else "Upload a thermal image (TIFF)"
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "tiff"], help=file_help)
    
    if uploaded_file is not None:
        try:
            # Process the image
            input_display, output_display = process_image(uploaded_file, model, device, direction, image_size)
            
            # Display images
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Input Image")
                if direction == "rgb2thermal":
                    st.image(transforms.ToPILImage()(input_display), caption="RGB Input", use_column_width=True)
                else:
                    # For thermal display, convert to 3 channels and normalize for better visualization
                    thermal_display = input_display.repeat(3, 1, 1)
                    st.image(transforms.ToPILImage()(thermal_display), caption="Thermal Input", use_column_width=True)
            
            with col2:
                st.subheader("Output Image")
                if direction == "rgb2thermal":
                    # For thermal output, convert to 3 channels for better visualization
                    thermal_output = output_display.repeat(3, 1, 1)
                    st.image(transforms.ToPILImage()(thermal_output), caption="Thermal Output", use_column_width=True)
                    
                    # Add download button for TIFF
                    output_tiff = (output_display.numpy() * 65535).astype(np.uint16)
                    
                    # Save TIFF to bytes
                    tiff_bytes = io.BytesIO()
                    tifffile.imwrite(tiff_bytes, output_tiff)
                    tiff_bytes.seek(0)
                    
                    # Download button
                    st.download_button(
                        label="Download Thermal Image (TIFF)",
                        data=tiff_bytes,
                        file_name="thermal_output.tiff",
                        mime="image/tiff"
                    )
                else:
                    st.image(transforms.ToPILImage()(output_display), caption="RGB Output", use_column_width=True)
                    
                    # Add download button for JPG
                    img_bytes = io.BytesIO()
                    save_image(output_display, img_bytes, format='JPEG')
                    img_bytes.seek(0)
                    
                    # Download button
                    st.download_button(
                        label="Download RGB Image (JPG)",
                        data=img_bytes,
                        file_name="rgb_output.jpg",
                        mime="image/jpeg"
                    )
        
        except Exception as e:
            st.error(f"Error processing image: {e}")
            logger.exception(e)

if __name__ == "__main__":
    main()