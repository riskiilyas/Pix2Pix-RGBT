import os
import json
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torchvision.transforms as transforms
from PIL import Image
import tifffile
from tqdm import tqdm
from src.ML_TEMPERATURE_PREDICTION.entity.config_entity import ModelEvaluationConfig
from src.ML_TEMPERATURE_PREDICTION.components.pix2pix_model import Generator
from src.ML_TEMPERATURE_PREDICTION.logging import logger

class ModelEvaluation:
    """
    Model evaluation component for evaluating trained models
    """
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Set direction
        if self.config.direction == 'rgb2thermal':
            self.input_channels = 3
            self.output_channels = 1
        else:  # thermal2rgb
            self.input_channels = 1
            self.output_channels = 3
        
        # Initialize model
        self.generator = Generator(in_channels=self.input_channels, out_channels=self.output_channels, filters=64).to(self.device)        
        # Load the best model if available
        self.load_model()
    
    def load_model(self):
        """
        Load the trained model
        """
        model_path = os.path.join(self.config.model_path, 'checkpoints/best.pth')
        if not os.path.exists(model_path):
            model_path = os.path.join(self.config.model_path, 'checkpoints/latest.pth')
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.generator.load_state_dict(checkpoint['generator'])
            logger.info(f"Loaded model from epoch {checkpoint['epoch']}")
        else:
            logger.error(f"No model checkpoint found at {model_path}")
            raise FileNotFoundError(f"No model checkpoint found at {model_path}")
    
    def evaluate(self, test_loader):
        """
        Evaluate the model on the test dataset
        """
        logger.info("Starting evaluation...")
        
        self.generator.eval()
        
        # Create directory for evaluation results
        os.makedirs(os.path.join(self.config.evaluation_path, 'images'), exist_ok=True)
        
        # Metrics
        l1_losses = []
        psnr_scores = []
        ssim_scores = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
                # Get real images
                real_A = batch['A'].to(self.device)  # Input image
                real_B = batch['B'].to(self.device)  # Target image
                filenames = batch['filename']
                
                # Generate fake images
                fake_B = self.generator(real_A)
                
                # Calculate L1 loss
                l1_loss = torch.nn.L1Loss()(fake_B, real_B).item()
                l1_losses.append(l1_loss)
                
                # Rescale images from [-1, 1] to [0, 1]
                real_A_np = ((real_A + 1) / 2).cpu().numpy()
                real_B_np = ((real_B + 1) / 2).cpu().numpy()
                fake_B_np = ((fake_B + 1) / 2).cpu().numpy()
                
                # Calculate PSNR and SSIM for each image in the batch
                for i in range(real_A.size(0)):
                    # Convert to format expected by skimage metrics (H,W,C)
                    if self.output_channels == 1:
                        # For thermal images, use single channel
                        real_B_img = np.transpose(real_B_np[i], (1, 2, 0)).squeeze()
                        fake_B_img = np.transpose(fake_B_np[i], (1, 2, 0)).squeeze()
                    else:
                        # For RGB images, use 3 channels
                        real_B_img = np.transpose(real_B_np[i], (1, 2, 0))
                        fake_B_img = np.transpose(fake_B_np[i], (1, 2, 0))
                    
                    # Calculate PSNR
                    psnr_score = psnr(real_B_img, fake_B_img, data_range=1.0)
                    psnr_scores.append(psnr_score)
                    
                    # Calculate SSIM
                    if self.output_channels == 1:
                        # For grayscale
                        ssim_score = ssim(real_B_img, fake_B_img, data_range=1.0)
                    else:
                        # For RGB
                        ssim_score = ssim(real_B_img, fake_B_img, data_range=1.0, channel_axis=2, multichannel=True)
                    ssim_scores.append(ssim_score)
                    
                    # Save images for visual comparison
                    if batch_idx < 10:  # Save only a few examples
                        # Prepare images for visualization
                        if self.output_channels == 1:
                            # Repeat single channel to 3 channels for visualization
                            fake_B_vis = torch.cat([fake_B[i]] * 3, dim=0)
                            real_B_vis = torch.cat([real_B[i]] * 3, dim=0)
                        else:
                            fake_B_vis = fake_B[i]
                            real_B_vis = real_B[i]
                        
                        # Rescale to [0, 1]
                        real_A_vis = (real_A[i] + 1) / 2
                        fake_B_vis = (fake_B_vis + 1) / 2
                        real_B_vis = (real_B_vis + 1) / 2
                        
                        # Concatenate images horizontally
                        combined = torch.cat((real_A_vis, fake_B_vis, real_B_vis), 2)
                        
                        # Save the combined image
                        save_path = os.path.join(self.config.evaluation_path, 'images', f"{filenames[i]}.png")
                        transforms.ToPILImage()(combined).save(save_path)
        
        # Calculate average metrics
        avg_l1_loss = np.mean(l1_losses)
        avg_psnr = np.mean(psnr_scores)
        avg_ssim = np.mean(ssim_scores)
        
        # Log and save results
        logger.info(f"Evaluation Results:")
        logger.info(f"L1 Loss: {avg_l1_loss:.4f}")
        logger.info(f"PSNR: {avg_psnr:.4f} dB")
        logger.info(f"SSIM: {avg_ssim:.4f}")
        
        # Save metrics to file
        metrics = {
            'l1_loss': float(avg_l1_loss),
            'psnr': float(avg_psnr),
            'ssim': float(avg_ssim)
        }
        
        with open(os.path.join(self.config.evaluation_path, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        return metrics
    
    def generate_predictions(self, input_dir, output_dir, input_type='rgb'):
        """
        Generate predictions for images in a directory
        """
        logger.info(f"Generating predictions for images in {input_dir}")
        
        self.generator.eval()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get list of input files
        if input_type == 'rgb':
            input_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]
            input_transform = transforms.Compose([
                transforms.Resize(self.config.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:  # thermal
            input_files = [f for f in os.listdir(input_dir) if f.endswith('.tiff')]
            input_transform = transforms.Compose([
                transforms.Resize(self.config.image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        
        with torch.no_grad():
            for filename in tqdm(input_files, desc="Generating predictions"):
                # Load input image
                input_path = os.path.join(input_dir, filename)
                
                if input_type == 'rgb':
                    # Load RGB image
                    input_img = Image.open(input_path).convert('RGB')
                    input_tensor = input_transform(input_img).unsqueeze(0).to(self.device)
                else:
                    # Load thermal image
                    thermal_img = tifffile.imread(input_path)
                    
                    # Normalize thermal image to 0-1 range
                    thermal_img = thermal_img.astype(np.float32)
                    thermal_min = np.min(thermal_img)
                    thermal_max = np.max(thermal_img)
                    if thermal_max > thermal_min:
                        thermal_img = (thermal_img - thermal_min) / (thermal_max - thermal_min)
                    else:
                        thermal_img = np.zeros_like(thermal_img)
                    
                    # Convert to PIL image for transforms
                    thermal_pil = Image.fromarray((thermal_img * 255).astype(np.uint8))
                    input_tensor = input_transform(thermal_pil).unsqueeze(0).to(self.device)
                
                # Generate prediction
                prediction = self.generator(input_tensor)
                
                # Convert prediction back to image
                prediction = (prediction.squeeze().cpu() + 1) / 2
                
                # Save prediction
                output_filename = os.path.splitext(filename)[0]
                if self.output_channels == 1:
                    # Save thermal output as 16-bit TIFF
                    prediction_np = prediction.numpy() * 65535  # Scale to 16-bit range
                    output_path = os.path.join(output_dir, f"{output_filename}.tiff")
                    tifffile.imwrite(output_path, prediction_np.astype(np.uint16))
                else:
                    # Save RGB output as JPEG
                    output_path = os.path.join(output_dir, f"{output_filename}.jpg")
                    transforms.ToPILImage()(prediction).save(output_path)
        
        logger.info(f"Predictions saved to {output_dir}")