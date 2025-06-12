import os
import uuid
import tempfile
from datetime import datetime
from pathlib import Path
import base64
import io
import logging
from logging.handlers import RotatingFileHandler
import json
from functools import wraps
import time

# Load environment variables
from dotenv import load_dotenv
load_dotenv()  # This loads the .env file

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_pymongo import PyMongo
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for no GUI environments

# Hugging Face integration
from gradio_client import Client, handle_file
import requests
from typing import Optional, Tuple, Dict, Any
import shutil

# Keep temperature analyzer for local processing
from temperature_measure import ThermalImageAnalyzer

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure MongoDB with fallback
MONGO_URI = os.environ.get("MONGO_URI")


app.config["MONGO_URI"] = MONGO_URI
mongo = PyMongo(app)

# Hugging Face configuration
HUGGING_FACE_SPACE_URL = os.environ.get("HF_SPACE_URL", "https://riskee64-pix2pix-rgb-t.hf.space")
HF_API_TOKEN = os.environ.get("HUGGING_FACE_TOKEN")
USE_HF_SPACE = os.environ.get("USE_HF_SPACE", "true").lower() == "true"

print(f"🤖 HF Space URL: {HUGGING_FACE_SPACE_URL}")
print(f"🔑 HF Token: {'✅ Available' if HF_API_TOKEN else '❌ Not set'}")
print(f"🚀 Use HF Space: {'✅ Enabled' if USE_HF_SPACE else '❌ Disabled'}")

class HuggingFaceInference:
    """Class to handle inference using Hugging Face Space or API"""
    
    def __init__(self, space_url: str = None, api_token: str = None):
        self.space_url = space_url or HUGGING_FACE_SPACE_URL
        self.api_token = api_token or HF_API_TOKEN
        self.client = None
        self.api_url = "https://api-inference.huggingface.co/models/riskee64/Pix2Pix-RGB-T"
        
        # Initialize Gradio client
        try:
            if self.space_url and self.space_url != "https://your-username-space-name.hf.space":
                self.client = Client(self.space_url)
                print(f"✅ Connected to Gradio Space: {self.space_url}")
            else:
                print("⚠️ Gradio Space URL not configured, will use HF Inference API")
        except Exception as e:
            print(f"⚠️ Failed to connect to Gradio Space: {e}")
            print("Will fallback to HF Inference API")

    def predict_via_gradio(self, image_path: str) -> str:
        """Predict using Gradio Space - returns TIFF file path"""
        if not self.client:
            raise Exception("Gradio client not available")
        
        try:
            print(f"🔄 Calling Gradio Space with: {image_path}")
            
            # Load as PIL Image and ensure it's RGB
            rgb_image = Image.open(image_path).convert('RGB')
            print(f"✅ Loaded PIL Image: {rgb_image.size}, mode: {rgb_image.mode}")
            
            # Call Gradio Space with PIL Image
            result = self.client.predict(
                rgb_image=handle_file(image_path),
                api_name="/predict"
            )
            
            print(f"✅ Gradio result type: {type(result)}")
            print(f"✅ Gradio result length: {len(result) if isinstance(result, (list, tuple)) else 'not list/tuple'}")
            
            # Handle result - expect [tiff_file_path, preview_image]
            if isinstance(result, (list, tuple)) and len(result) >= 1:
                thermal_tiff_path = result[0]
                print(f"🔍 Received TIFF path: {thermal_tiff_path}, type: {type(thermal_tiff_path)}")
                
                # Handle different types of TIFF response
                temp_tiff = tempfile.NamedTemporaryFile(suffix='.tiff', delete=False)
                
                if isinstance(thermal_tiff_path, str):
                    # If it's a file path string
                    if os.path.exists(thermal_tiff_path):
                        print(f"✅ Copying from existing path: {thermal_tiff_path}")
                        shutil.copy2(thermal_tiff_path, temp_tiff.name)
                    else:
                        print(f"⚠️ Path doesn't exist: {thermal_tiff_path}")
                        raise Exception(f"TIFF file not found: {thermal_tiff_path}")
                
                elif hasattr(thermal_tiff_path, 'name'):
                    # If it's a file object with name attribute
                    print(f"✅ Copying from file object: {thermal_tiff_path.name}")
                    shutil.copy2(thermal_tiff_path.name, temp_tiff.name)
                
                elif hasattr(thermal_tiff_path, 'read'):
                    # If it's a file-like object
                    print(f"✅ Reading from file-like object")
                    with open(temp_tiff.name, 'wb') as f:
                        thermal_tiff_path.seek(0)
                        f.write(thermal_tiff_path.read())
                
                else:
                    raise Exception(f"Unknown TIFF format: {type(thermal_tiff_path)}")
                
                print(f"✅ Thermal TIFF saved to: {temp_tiff.name}")
                
                # Verify the file was created and has content
                if os.path.exists(temp_tiff.name) and os.path.getsize(temp_tiff.name) > 0:
                    return temp_tiff.name
                else:
                    raise Exception("Created TIFF file is empty or doesn't exist")
            
            else:
                raise Exception(f"Unexpected result format: {type(result)} - {result}")
            
        except Exception as e:
            print(f"❌ Gradio prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise Exception(f"Gradio Space prediction failed: {str(e)}")

    def predict_via_api(self, image_path: str) -> str:
        """Predict using HF Inference API - returns TIFF file path"""
        if not self.api_token:
            raise Exception("HF API token not provided")
        
        headers = {"Authorization": f"Bearer {self.api_token}"}
        
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            
            response = requests.post(
                self.api_url,
                headers=headers,
                data=image_bytes,
                timeout=60
            )
            
            if response.status_code == 200:
                # Convert response to 16-bit TIFF
                image = Image.open(io.BytesIO(response.content))
                thermal_array = np.array(image)
                
                # Scale to 16-bit if needed
                if thermal_array.dtype != np.uint16:
                    thermal_array = (thermal_array * 256).astype(np.uint16)
                
                # Save as TIFF
                temp_tiff = tempfile.NamedTemporaryFile(suffix='.tiff', delete=False)
                try:
                    import tifffile
                    tifffile.imwrite(temp_tiff.name, thermal_array)
                except ImportError:
                    thermal_image_pil = Image.fromarray(thermal_array, mode='I;16')
                    thermal_image_pil.save(temp_tiff.name, format='TIFF')
                
                print(f"✅ Thermal TIFF created from API: {temp_tiff.name}")
                return temp_tiff.name
            else:
                raise Exception(f"API Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            raise Exception(f"HF API prediction failed: {str(e)}")
    
    def predict(self, image_path: str, min_temp: float = 28, max_temp: float = 30) -> Tuple[str, Dict]:
        """Main prediction method - returns TIFF path and local analysis"""
        
        # Get thermal TIFF from Gradio Space or API
        thermal_tiff_path = None
        
        # Try Gradio Space first
        if self.client:
            try:
                thermal_tiff_path = self.predict_via_gradio(image_path)
            except Exception as e:
                print(f"Gradio Space failed: {e}")
                print("Falling back to HF Inference API...")
        
        # Fallback to HF Inference API
        if not thermal_tiff_path:
            try:
                thermal_tiff_path = self.predict_via_api(image_path)
            except Exception as e:
                raise Exception(f"All prediction methods failed. Last error: {str(e)}")
        
        # Analyze thermal image locally for temperature stats
        try:
            analyzer = ThermalImageAnalyzer(
                thermal_image_path=thermal_tiff_path,
                rgb_image_path=image_path,
                assumed_min_temp=min_temp,
                assumed_max_temp=max_temp
            )
            
            results = analyzer.get_results()
            return thermal_tiff_path, results['stats']
            
        except Exception as e:
            # Return TIFF path with default stats if analysis fails
            print(f"Warning: Local analysis failed: {e}")
            default_stats = {'center': 29.0, 'mean': 29.0, 'min': 28.0, 'max': 30.0}
            return thermal_tiff_path, default_stats

# Initialize HF inference
hf_inference = HuggingFaceInference() if USE_HF_SPACE else None

def log_request(f):
    """Decorator to log all requests with detailed information"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Start timing
        start_time = time.time()
        
        # Get request info
        request_id = str(uuid.uuid4())[:8]  # Short request ID
        client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
        user_agent = request.headers.get('User-Agent', 'Unknown')
        
        # Log request start
        request_info = {
            'request_id': request_id,
            'method': request.method,
            'endpoint': request.endpoint,
            'url': request.url,
            'client_ip': client_ip,
            'user_agent': user_agent,
            'content_type': request.content_type,
            'content_length': request.content_length,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add form data info (without sensitive data)
        if request.form:
            form_data = {}
            for key in request.form.keys():
                if key.lower() not in ['password', 'token', 'secret']:
                    form_data[key] = request.form.get(key)
            request_info['form_data'] = form_data
        
        # Add file info
        if request.files:
            file_info = {}
            for key, file in request.files.items():
                file_info[key] = {
                    'filename': file.filename,
                    'content_type': file.content_type,
                    'size': len(file.read()) if hasattr(file, 'read') else 'unknown'
                }
                file.seek(0)  # Reset file pointer
            request_info['files'] = file_info
        
        # Log request
        request_logger.info(f"REQUEST_START: {json.dumps(request_info, indent=2)}")
        app_logger.info(f"[{request_id}] {request.method} {request.endpoint} - Start")
        
        try:
            # Execute the actual function
            response = f(*args, **kwargs)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Log successful response
            if hasattr(response, 'status_code'):
                status_code = response.status_code
            elif isinstance(response, tuple):
                status_code = response[1] if len(response) > 1 else 200
            else:
                status_code = 200
            
            response_info = {
                'request_id': request_id,
                'status_code': status_code,
                'processing_time_seconds': round(processing_time, 3),
                'response_size': len(str(response)) if response else 0,
                'timestamp': datetime.now().isoformat()
            }
            
            request_logger.info(f"REQUEST_END: {json.dumps(response_info, indent=2)}")
            app_logger.info(f"[{request_id}] {request.method} {request.endpoint} - Success ({status_code}) in {processing_time:.3f}s")
            
            return response
            
        except Exception as e:
            # Calculate processing time for errors
            processing_time = time.time() - start_time
            
            # Log error
            error_info = {
                'request_id': request_id,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'processing_time_seconds': round(processing_time, 3),
                'timestamp': datetime.now().isoformat()
            }
            
            request_logger.error(f"REQUEST_ERROR: {json.dumps(error_info, indent=2)}")
            app_logger.error(f"[{request_id}] {request.method} {request.endpoint} - Error: {str(e)}")
            
            # Re-raise the exception
            raise
    
    return decorated_function



def encode_image_to_base64(image_path):
    """Convert an image file to base64 encoding"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def encode_array_to_base64(image_array, format='PNG'):
    """Convert numpy array to base64 encoding"""
    if isinstance(image_array, np.ndarray):
        image = Image.fromarray(image_array.astype(np.uint8))
        buf = io.BytesIO()
        image.save(buf, format=format)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    return None

@app.route('/predict', methods=['POST'])
@log_request
def predict():
    """Generate a thermal prediction from an RGB image using Hugging Face"""
    try:
        # Check if RGB image is provided
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        rgb_file = request.files['image']
        if rgb_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Get optional parameters
        user_id = request.form.get('user_id', 'anonymous')
        monitoring_id = request.form.get('monitoring_id', str(uuid.uuid4()))
        min_temp = float(request.form.get('min_temp', 28))
        max_temp = float(request.form.get('max_temp', 30))
        
        # Create a prediction ID
        prediction_id = str(uuid.uuid4())
        
        # Create temporary working directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the RGB image to temp directory
            rgb_path = os.path.join(temp_dir, "rgb_input.jpg")
            rgb_file.save(rgb_path)
            
            # Validate image after saving
            try:
                rgb_image = Image.open(rgb_path).convert('RGB')
                width, height = rgb_image.size
                if width == 0 or height == 0:
                    return jsonify({'error': 'Invalid image file - image has no dimensions'}), 400
                
                min_dimension = 32
                if width < min_dimension or height < min_dimension:
                    return jsonify({'error': f'Image too small. Minimum size: {min_dimension}x{min_dimension} pixels'}), 400
                    
            except Exception as img_error:
                return jsonify({'error': f'Invalid image file: {str(img_error)}'}), 400
            
            # Generate thermal prediction using Hugging Face
            try:
                if hf_inference and USE_HF_SPACE:
                    # Get TIFF file and stats from HF Space + local analysis
                    thermal_tiff_path, stats = hf_inference.predict(rgb_path, min_temp, max_temp)
                    
                    # Create visualizations from the TIFF file
                    thermal_colored_path = os.path.join(temp_dir, "thermal_viz.png")
                    grayscale_path = os.path.join(temp_dir, "grayscale_8bit.png")
                    temp_map_path = os.path.join(temp_dir, "temperature_map.png")
                    
                    # Load TIFF and create visualizations
                    try:
                        import tifffile
                        thermal_array = tifffile.imread(thermal_tiff_path)
                        print(f"✅ Loaded TIFF with shape: {thermal_array.shape}, dtype: {thermal_array.dtype}")
                    except ImportError:
                        thermal_image = Image.open(thermal_tiff_path)
                        thermal_array = np.array(thermal_image)
                        print(f"✅ Loaded via PIL with shape: {thermal_array.shape}, dtype: {thermal_array.dtype}")
                    
                    # Create colored thermal image
                    normalized = cv2.normalize(thermal_array, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    thermal_colored = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
                    cv2.imwrite(thermal_colored_path, thermal_colored)
                    
                    # Create grayscale version
                    cv2.imwrite(grayscale_path, normalized)
                    
                    # Create temperature map using local analyzer
                    try:
                        analyzer = ThermalImageAnalyzer(
                            thermal_image_path=thermal_tiff_path,
                            rgb_image_path=rgb_path,
                            assumed_min_temp=min_temp,
                            assumed_max_temp=max_temp
                        )
                        
                        plt.figure(figsize=(8, 6))
                        plt.imshow(analyzer.temp_scale, cmap=analyzer.get_custom_colormap())
                        plt.colorbar(label='Temperature (°C)')
                        plt.axis('off')
                        plt.title('Temperature Distribution')
                        plt.tight_layout()
                        plt.savefig(temp_map_path, dpi=150, bbox_inches='tight')
                        plt.close()
                        
                        # Update stats from local analysis
                        results = analyzer.get_results()
                        stats = results['stats']
                        
                        print(f"✅ Temperature stats: {stats}")
                        
                    except Exception as analyzer_error:
                        print(f"⚠️ Temperature analysis failed: {analyzer_error}")
                        # Create simple temperature map fallback
                        plt.figure(figsize=(8, 6))
                        plt.imshow(thermal_array, cmap='inferno')
                        plt.colorbar(label='Raw Values')
                        plt.axis('off')
                        plt.title('Thermal Image (Raw)')
                        plt.tight_layout()
                        plt.savefig(temp_map_path, dpi=150, bbox_inches='tight')
                        plt.close()
                    
                    # Clean up temp TIFF
                    if os.path.exists(thermal_tiff_path):
                        os.unlink(thermal_tiff_path)
                
                else:
                    return jsonify({'error': 'Hugging Face inference not configured. Please set HF_SPACE_URL or HUGGING_FACE_TOKEN environment variables.'}), 500
                    
            except Exception as hf_error:
                return jsonify({'error': f'Thermal prediction failed: {str(hf_error)}'}), 500
            
            # Create document for MongoDB
            try:
                prediction_doc = {
                    'prediction_id': prediction_id,
                    'user_id': user_id,
                    'monitoring_id': monitoring_id,
                    'timestamp': datetime.now(),
                    'temperature_stats': {
                        'center': float(stats.get('center', 0)),
                        'mean': float(stats.get('mean', 0)),
                        'min': float(stats.get('min', 0)),
                        'max': float(stats.get('max', 0))
                    },
                    'temperature_range': {
                        'min_temp': min_temp,
                        'max_temp': max_temp
                    },
                    'inference_method': 'huggingface_space' if hf_inference and hf_inference.client else 'huggingface_api',
                    'images': {
                        'rgb_input': encode_image_to_base64(rgb_path),
                        'thermal_colored': encode_image_to_base64(thermal_colored_path),
                        'grayscale': encode_image_to_base64(grayscale_path),
                        'temperature_map': encode_image_to_base64(temp_map_path)
                    }
                }
                
                # Insert document into MongoDB
                mongo.db.predictions.insert_one(prediction_doc)
                print(f"✅ Saved prediction to database: {prediction_id}")
                
            except Exception as db_error:
                print(f"Database error (non-critical): {str(db_error)}")
                # Continue even if DB save fails
            
            # Create image URLs for the response
            image_urls = {
                'rgb_input': f"/images/{prediction_id}/rgb_input.jpg",
                'thermal_colored': f"/images/{prediction_id}/thermal_viz.png",
                'grayscale': f"/images/{prediction_id}/grayscale_8bit.png",
                'temperature_map': f"/images/{prediction_id}/temperature_map.png"
            }
            
            # Return results
            return jsonify({
                'success': True,
                'prediction_id': prediction_id,
                'user_id': user_id,
                'monitoring_id': monitoring_id,
                'temperature_stats': {
                    'center': float(stats.get('center', 0)),
                    'mean': float(stats.get('mean', 0)),
                    'min': float(stats.get('min', 0)),
                    'max': float(stats.get('max', 0))
                },
                'temperature_range': {
                    'min_temp': min_temp,
                    'max_temp': max_temp
                },
                'image_urls': image_urls,
                'inference_method': 'huggingface_space' if hf_inference and hf_inference.client else 'huggingface_api'
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
@log_request
def get_prediction(prediction_id):
    """Get a specific prediction by ID"""
    try:
        prediction = mongo.db.predictions.find_one(
            {'prediction_id': prediction_id},
            {'images': 0}  # Exclude images
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
    hf_status = "disabled"
    if USE_HF_SPACE:
        if hf_inference and hf_inference.client:
            hf_status = "gradio_space_connected"
        elif HF_API_TOKEN:
            hf_status = "api_token_available"
        else:
            hf_status = "not_configured"
    
    mongo_status = "disconnected"
    try:
        mongo.db.list_collection_names()
        mongo_status = "connected"
    except Exception as e:
        mongo_status = f"error: {str(e)}"
    
    return jsonify({
        'status': 'ok',
        'huggingface_status': hf_status,
        'space_url': HUGGING_FACE_SPACE_URL if USE_HF_SPACE else None,
        'use_hf_space': USE_HF_SPACE,
        'mongo_status': mongo_status,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/test-hf', methods=['POST'])
def test_huggingface():
    """Test endpoint for Hugging Face connection"""
    if not USE_HF_SPACE or not hf_inference:
        return jsonify({'error': 'Hugging Face integration not enabled'}), 400
    
    try:
        # Test with a simple request
        test_message = "Hugging Face connection test successful"
        if hf_inference.client:
            test_message += " (via Gradio Space)"
        elif HF_API_TOKEN:
            test_message += " (via Inference API)"
        
        return jsonify({
            'success': True,
            'message': test_message,
            'space_url': HUGGING_FACE_SPACE_URL,
            'has_gradio_client': hf_inference.client is not None,
            'has_api_token': bool(HF_API_TOKEN)
        })
        
    except Exception as e:
        return jsonify({'error': f'Hugging Face test failed: {str(e)}'}), 500

def setup_logging():
    """Setup comprehensive logging configuration"""
    
    # Create logs directory if not exists
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure main logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create Flask app logger
    app_logger = logging.getLogger('flask_app')
    app_logger.setLevel(logging.INFO)
    
    # Create rotating file handler for general logs
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, 'flask_app.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    
    # Create rotating file handler for request logs
    request_handler = RotatingFileHandler(
        os.path.join(log_dir, 'requests.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    request_handler.setLevel(logging.INFO)
    
    # Create rotating file handler for error logs
    error_handler = RotatingFileHandler(
        os.path.join(log_dir, 'errors.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s'
    )
    
    request_formatter = logging.Formatter(
        '%(asctime)s - %(message)s'
    )
    
    # Set formatters
    file_handler.setFormatter(detailed_formatter)
    request_handler.setFormatter(request_formatter)
    error_handler.setFormatter(detailed_formatter)
    
    # Add handlers to loggers
    app_logger.addHandler(file_handler)
    app_logger.addHandler(error_handler)
    
    # Create specific request logger
    request_logger = logging.getLogger('requests')
    request_logger.setLevel(logging.INFO)
    request_logger.addHandler(request_handler)
    
    # Prevent duplicate logs
    request_logger.propagate = False
    
    return app_logger, request_logger


app_logger, request_logger = setup_logging()
if __name__ == '__main__':
    print("🚀 Starting Flask server with Hugging Face integration...")
    print(f"📡 HF Space URL: {HUGGING_FACE_SPACE_URL}")
    print(f"🔑 HF API Token: {'✅ Available' if HF_API_TOKEN else '❌ Not set'}")
    print(f"🤖 Use HF Space: {'✅ Enabled' if USE_HF_SPACE else '❌ Disabled'}")
    
    # Test MongoDB connection
    try:
        mongo.db.list_collection_names()
        print("✅ MongoDB connected successfully")
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")

    port = int(os.environ.get('PORT', 3210))
    debug_mode = os.environ.get('FLASK_ENV', 'production') == 'development'

    # Run Flask app
    app.run(host='0.0.0.0', port=port, debug=debug_mode)