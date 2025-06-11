import os
import uuid
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
import base64
import io

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_pymongo import PyMongo
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for no GUI environments

# Import from the ML_TEMPERATURE_PREDICTION package
from src.ML_TEMPERATURE_PREDICTION.components.pix2pix_model import Generator
from src.ML_TEMPERATURE_PREDICTION.utils.common import read_yaml
from temperature_measure import ThermalImageAnalyzer

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure MongoDB
# app.config["MONGO_URI"] = os.environ.get("MONGO_URL", "mongodb://localhost:27017/thermal_predictions")
mongo = PyMongo(app)

# Load configuration
CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")
config = read_yaml(CONFIG_FILE_PATH)
params = read_yaml(PARAMS_FILE_PATH)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set model parameters based on configuration
if params.DIRECTION == 'rgb2thermal':
    input_channels = 3
    output_channels = 1
else:
    input_channels = 1
    output_channels = 3

# Initialize model
model = Generator(in_channels=input_channels, out_channels=output_channels).to(device)

# Load model weights
model_path = os.path.join("artifacts", "model_trainer", "checkpoints", "best.pth")
if not os.path.exists(model_path):
    model_path = os.path.join("artifacts", "model_trainer", "checkpoints", "latest.pth")

if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['generator'])
    model.eval()
    print(f"Model loaded successfully from {model_path}")
else:
    print(f"No model found at {model_path}. Please train the model first.")

# Define transforms for RGB input
transform = transforms.Compose([
    transforms.Resize(params.IMAGE_SIZE),
    transforms.CenterCrop(params.IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def encode_image_to_base64(image_path):
    """Convert an image file to base64 encoding"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def save_figure_to_bytes(figure):
    """Save a matplotlib figure to bytes and return as base64"""
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def validate_image_file(file):
    """Validate if the uploaded file is a valid image"""
    # Check file extension
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
    file_extension = file.filename.lower().split('.')[-1] if '.' in file.filename else ''
    
    if file_extension not in allowed_extensions:
        return False, 'Invalid file extension. Allowed: PNG, JPG, JPEG, GIF, BMP, TIFF'
    
    # Check content type
    if not file.content_type or not file.content_type.startswith('image/'):
        return False, 'Invalid content type. Please upload an image file'
    
    # Check file size (optional - add max size limit)
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset to beginning
    
    max_size = 10 * 1024 * 1024  # 10MB limit
    if file_size > max_size:
        return False, 'File too large. Maximum size: 10MB'
    
    if file_size == 0:
        return False, 'Empty file uploaded'
    
    return True, 'Valid image file'

@app.route('/predict', methods=['POST'])
def predict():
    """Generate a thermal prediction from an RGB image"""
    try:
        # Check if RGB image is provided
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        rgb_file = request.files['image']
        if rgb_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Validate the uploaded file
        is_valid, message = validate_image_file(rgb_file)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        # Get optional user_id and monitoring_id
        user_id = request.form.get('user_id', 'anonymous')
        monitoring_id = request.form.get('monitoring_id', str(uuid.uuid4()))
        
        # Create a prediction ID
        prediction_id = str(uuid.uuid4())
        
        # Create temporary working directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the RGB image to temp directory
            rgb_path = os.path.join(temp_dir, f"rgb_input.jpg")
            rgb_file.save(rgb_path)
            
            # Additional image validation after saving
            try:
                # Try to open and validate the image
                rgb_image = Image.open(rgb_path).convert('RGB')
                # Verify it's a valid image by getting its size
                width, height = rgb_image.size
                if width == 0 or height == 0:
                    return jsonify({'error': 'Invalid image file - image has no dimensions'}), 400
                
                # Check minimum image dimensions (optional)
                min_dimension = 32  # Minimum 32x32 pixels
                if width < min_dimension or height < min_dimension:
                    return jsonify({'error': f'Image too small. Minimum size: {min_dimension}x{min_dimension} pixels'}), 400
                    
            except Exception as img_error:
                return jsonify({'error': f'Invalid image file - cannot process as image: {str(img_error)}'}), 400
            
            # Process the RGB image
            try:
                input_tensor = transform(rgb_image).unsqueeze(0).to(device)
            except Exception as transform_error:
                return jsonify({'error': f'Error processing image: {str(transform_error)}'}), 400
            
            # Generate thermal image
            try:
                with torch.no_grad():
                    output_tensor = model(input_tensor)
            except Exception as model_error:
                return jsonify({'error': f'Error generating thermal prediction: {str(model_error)}'}), 500
            
            # Denormalize output tensor
            output_tensor = (output_tensor + 1) / 2
            
            # Convert to numpy and scale to 16-bit range
            thermal_array = output_tensor.squeeze().cpu().numpy() * 65535
            thermal_array = thermal_array.astype(np.uint16)
            
            # Save thermal image to temp directory
            thermal_path = os.path.join(temp_dir, f"thermal_output.tiff")
            
            try:
                # Import tifffile here to avoid issues with matplotlib
                import tifffile
                tifffile.imwrite(thermal_path, thermal_array)
            except Exception as tiff_error:
                return jsonify({'error': f'Error saving thermal image: {str(tiff_error)}'}), 500
            
            # Analyze thermal image using ThermalImageAnalyzer
            try:
                analyzer = ThermalImageAnalyzer(
                    thermal_image_path=thermal_path,
                    rgb_image_path=rgb_path,
                    assumed_min_temp=28,
                    assumed_max_temp=30
                )
                
                # Get visualization results without displaying them
                visualizations = analyzer.get_results()
            except Exception as analyzer_error:
                return jsonify({'error': f'Error analyzing thermal image: {str(analyzer_error)}'}), 500
            
            # Save visualization images to temp directory
            thermal_colored_path = os.path.join(temp_dir, "thermal_viz.png")
            grayscale_path = os.path.join(temp_dir, "grayscale_8bit.png")
            
            try:
                import cv2
                cv2.imwrite(thermal_colored_path, visualizations['visualizations']['thermal_with_markers'])
                cv2.imwrite(grayscale_path, visualizations['visualizations']['grayscale'])
            except Exception as cv2_error:
                return jsonify({'error': f'Error saving visualization images: {str(cv2_error)}'}), 500
            
            # Save temperature map
            try:
                plt.figure(figsize=(8, 6))
                plt.imshow(visualizations['temp_scale'], cmap=analyzer.get_custom_colormap())
                plt.colorbar(label='Estimated Temperature (°C)')
                plt.axis('off')
                plt.tight_layout()
                
                temp_map_path = os.path.join(temp_dir, "temperature_map.png")
                plt.savefig(temp_map_path)
                plt.close()
            except Exception as plt_error:
                return jsonify({'error': f'Error creating temperature map: {str(plt_error)}'}), 500
            
            # Create document for MongoDB
            try:
                prediction_doc = {
                    'prediction_id': prediction_id,
                    'user_id': user_id,
                    'monitoring_id': monitoring_id,
                    'timestamp': datetime.now(),
                    'temperature_stats': {
                        'center': float(visualizations['stats']['center']),
                        'mean': float(visualizations['stats']['mean']),
                        'min': float(visualizations['stats']['min']),
                        'max': float(visualizations['stats']['max'])
                    },
                    'images': {
                        'rgb_input': encode_image_to_base64(rgb_path),
                        'thermal_colored': encode_image_to_base64(thermal_colored_path),
                        'grayscale': encode_image_to_base64(grayscale_path),
                        'temperature_map': encode_image_to_base64(temp_map_path)
                    }
                }
                
                # Insert document into MongoDB
                mongo.db.predictions.insert_one(prediction_doc)
            except Exception as db_error:
                return jsonify({'error': f'Error saving to database: {str(db_error)}'}), 500
            
            # Create image URLs for the response
            image_urls = {
                'rgb_input': f"/images/{prediction_id}/rgb_input.jpg",
                'thermal_colored': f"/images/{prediction_id}/thermal_viz.png",
                'grayscale': f"/images/{prediction_id}/grayscale_8bit.png",
                'temperature_map': f"/images/{prediction_id}/temperature_map.png"
            }
            
            # Return results
            return jsonify({
                'prediction_id': prediction_id,
                'user_id': user_id,
                'monitoring_id': monitoring_id,
                'temperature_stats': {
                    'center': float(visualizations['stats']['center']),
                    'mean': float(visualizations['stats']['mean']),
                    'min': float(visualizations['stats']['min']),
                    'max': float(visualizations['stats']['max'])
                },
                'image_urls': image_urls
            }), 200
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Unexpected server error: {str(e)}'}), 500

@app.route('/predictions/user/<user_id>', methods=['GET'])
def get_predictions_by_user(user_id):
    """Get all predictions for a specific user"""
    try:
        predictions = list(mongo.db.predictions.find(
            {'user_id': user_id},
            {'images': 0}  # Exclude image data for faster response
        ))
        
        # Convert ObjectId to string and add image URLs
        for prediction in predictions:
            prediction['_id'] = str(prediction['_id'])
            
            # Add image URLs if not already present
            if 'image_urls' not in prediction:
                prediction_id = prediction['prediction_id']
                prediction['image_urls'] = {
                    'rgb_input': f"/images/{prediction_id}/rgb_input.jpg",
                    'thermal_colored': f"/images/{prediction_id}/thermal_viz.png",
                    'grayscale': f"/images/{prediction_id}/grayscale_8bit.png",
                    'temperature_map': f"/images/{prediction_id}/temperature_map.png"
                }
        
        return jsonify(predictions), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predictions/monitoring/<monitoring_id>', methods=['GET'])
def get_predictions_by_monitoring(monitoring_id):
    """Get all predictions for a specific monitoring ID"""
    try:
        predictions = list(mongo.db.predictions.find(
            {'monitoring_id': monitoring_id},
            {'images': 0}  # Exclude image data for faster response
        ))
        
        # Convert ObjectId to string and add image URLs
        for prediction in predictions:
            prediction['_id'] = str(prediction['_id'])
            
            # Add image URLs if not already present
            if 'image_urls' not in prediction:
                prediction_id = prediction['prediction_id']
                prediction['image_urls'] = {
                    'rgb_input': f"/images/{prediction_id}/rgb_input.jpg",
                    'thermal_colored': f"/images/{prediction_id}/thermal_viz.png",
                    'grayscale': f"/images/{prediction_id}/grayscale_8bit.png",
                    'temperature_map': f"/images/{prediction_id}/temperature_map.png"
                }
        
        return jsonify(predictions), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/prediction/<prediction_id>', methods=['GET'])
def get_prediction(prediction_id):
    try:
        # Always exclude images, use image_urls instead
        prediction = mongo.db.predictions.find_one(
            {'prediction_id': prediction_id},
            {'images': 0}  # Always exclude images
        )
        
        if not prediction:
            return jsonify({'error': 'Prediction not found'}), 404
        
        # Convert ObjectId to string
        prediction['_id'] = str(prediction['_id'])
        
        # Add image URLs
        prediction['image_urls'] = {
            'rgb_input': f"/images/{prediction_id}/rgb_input.jpg",
            'thermal_colored': f"/images/{prediction_id}/thermal_viz.png",
            'grayscale': f"/images/{prediction_id}/grayscale_8bit.png",
            'temperature_map': f"/images/{prediction_id}/temperature_map.png"
        }
        
        return jsonify(prediction), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/images/<prediction_id>/<image_name>', methods=['GET'])
def get_image(prediction_id, image_name):
    """Serve image files for predictions"""
    try:
        # Find the prediction in MongoDB
        prediction = mongo.db.predictions.find_one({'prediction_id': prediction_id})
        
        if not prediction or 'images' not in prediction:
            return jsonify({'error': 'Image not found'}), 404
        
        # Map image name to the corresponding key in MongoDB
        image_key_map = {
            'rgb_input.jpg': 'rgb_input',
            'thermal_viz.png': 'thermal_colored',
            'grayscale_8bit.png': 'grayscale',
            'temperature_map.png': 'temperature_map'
        }
        
        if image_name not in image_key_map or image_key_map[image_name] not in prediction['images']:
            return jsonify({'error': 'Image not found'}), 404
        
        # Get the base64 image data
        img_data = prediction['images'][image_key_map[image_name]]
        
        # Convert base64 to binary
        img_binary = base64.b64decode(img_data)
        
        # Determine content type
        content_type = 'image/jpeg' if image_name.endswith('.jpg') else 'image/png'
        
        # Create response with the correct content type
        response = app.response_class(img_binary, content_type=content_type)
        
        return response
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'device': str(device),
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Ensure artifacts directory exists
    os.makedirs("artifacts", exist_ok=True)
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)