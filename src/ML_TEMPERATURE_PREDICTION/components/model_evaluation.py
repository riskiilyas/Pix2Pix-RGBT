import os
import json
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torchvision.transforms as transforms
from PIL import Image
import tifffile
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from src.ML_TEMPERATURE_PREDICTION.entity.config_entity import ModelEvaluationConfig
from src.ML_TEMPERATURE_PREDICTION.components.pix2pix_model import Generator
from src.ML_TEMPERATURE_PREDICTION.logging import logger

class ModelEvaluation:
    """
    Enhanced Model evaluation component for evaluating trained models with temperature analysis
    """
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Temperature range for marine environments (28-30°C)
        self.temp_min = 28.0
        self.temp_max = 30.0
        
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
    
    def thermal_to_temperature(self, thermal_array):
        """
        Convert normalized thermal values to temperature in Celsius
        Assumes thermal values are normalized between 0-1
        """
        # Map normalized values [0,1] to temperature range [28,30]°C
        temperatures = thermal_array * (self.temp_max - self.temp_min) + self.temp_min
        return temperatures
    
    def calculate_temperature_metrics(self, predicted_temps, target_temps):
        """
        Calculate temperature-specific metrics (MAE, RMSE, R²)
        """
        # Flatten arrays for metric calculation
        pred_flat = predicted_temps.flatten()
        target_flat = target_temps.flatten()
        
        # Calculate metrics
        mae = mean_absolute_error(target_flat, pred_flat)
        rmse = np.sqrt(mean_squared_error(target_flat, pred_flat))
        r2 = r2_score(target_flat, pred_flat)
        
        # Calculate temperature range compliance
        temp_range_compliance = np.mean(
            (pred_flat >= self.temp_min) & (pred_flat <= self.temp_max)
        ) * 100
        
        # Calculate mean temperatures
        mean_pred_temp = np.mean(pred_flat)
        mean_target_temp = np.mean(target_flat)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'temp_range_compliance': temp_range_compliance,
            'mean_predicted_temp': mean_pred_temp,
            'mean_target_temp': mean_target_temp,
            'temp_std_predicted': np.std(pred_flat),
            'temp_std_target': np.std(target_flat)
        }
    
    def create_temperature_visualizations(self, predicted_temps, target_temps, save_path):
        """
        Create comprehensive temperature analysis visualizations
        """
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Temperature Prediction Analysis', fontsize=16, fontweight='bold')
        
        # Flatten arrays for plotting
        pred_flat = predicted_temps.flatten()
        target_flat = target_temps.flatten()
        
        # 1. Scatter plot - Predicted vs Target
        axes[0, 0].scatter(target_flat, pred_flat, alpha=0.6, s=1)
        axes[0, 0].plot([self.temp_min, self.temp_max], [self.temp_min, self.temp_max], 'r--', lw=2)
        axes[0, 0].set_xlabel('Target Temperature (°C)')
        axes[0, 0].set_ylabel('Predicted Temperature (°C)')
        axes[0, 0].set_title('Predicted vs Target Temperature')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add correlation coefficient
        correlation = np.corrcoef(target_flat, pred_flat)[0, 1]
        axes[0, 0].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                       transform=axes[0, 0].transAxes, bbox=dict(boxstyle="round", facecolor='white'))
        
        # 2. Temperature distribution histograms
        axes[0, 1].hist(target_flat, bins=50, alpha=0.7, label='Target', color='blue', density=True)
        axes[0, 1].hist(pred_flat, bins=50, alpha=0.7, label='Predicted', color='red', density=True)
        axes[0, 1].axvline(self.temp_min, color='green', linestyle='--', label='Min Standard (28°C)')
        axes[0, 1].axvline(self.temp_max, color='green', linestyle='--', label='Max Standard (30°C)')
        axes[0, 1].set_xlabel('Temperature (°C)')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Temperature Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Error distribution
        errors = pred_flat - target_flat
        axes[0, 2].hist(errors, bins=50, alpha=0.7, color='orange', density=True)
        axes[0, 2].axvline(0, color='red', linestyle='--', label='Perfect Prediction')
        axes[0, 2].axvline(np.mean(errors), color='blue', linestyle='-', label=f'Mean Error: {np.mean(errors):.3f}°C')
        axes[0, 2].set_xlabel('Prediction Error (°C)')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].set_title('Prediction Error Distribution')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Box plot comparison
        data_for_box = [target_flat, pred_flat]
        axes[1, 0].boxplot(data_for_box, labels=['Target', 'Predicted'])
        axes[1, 0].axhline(self.temp_min, color='green', linestyle='--', alpha=0.7, label='Standard Range')
        axes[1, 0].axhline(self.temp_max, color='green', linestyle='--', alpha=0.7)
        axes[1, 0].set_ylabel('Temperature (°C)')
        axes[1, 0].set_title('Temperature Range Comparison')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Residual plot
        axes[1, 1].scatter(target_flat, errors, alpha=0.6, s=1)
        axes[1, 1].axhline(0, color='red', linestyle='--')
        axes[1, 1].set_xlabel('Target Temperature (°C)')
        axes[1, 1].set_ylabel('Residual (°C)')
        axes[1, 1].set_title('Residual Plot')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Q-Q plot for error normality
        stats.probplot(errors, dist="norm", plot=axes[1, 2])
        axes[1, 2].set_title('Q-Q Plot of Residuals')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Temperature visualization saved to {save_path}")
    
    def create_metrics_summary_plot(self, metrics, save_path):
        """
        Create a summary plot of all evaluation metrics
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Model Evaluation Metrics Summary', fontsize=16, fontweight='bold')
        
        # Current results from your evaluation
        current_results = {
            'l1_loss': 0.18755212832580914,
            'psnr': 18.895721599477806,
            'ssim': 0.6028149900483882
        }
        
        # Add temperature metrics
        temp_metrics = ['MAE', 'RMSE', 'R²', 'Temp Range Compliance (%)']
        temp_values = [metrics['mae'], metrics['rmse'], metrics['r2'], metrics['temp_range_compliance']]
        
        # Define target thresholds
        targets = {
            'L1 Loss': 0.15,
            'PSNR (dB)': 25.0,
            'SSIM': 0.75,
            'MAE (°C)': 0.5,
            'RMSE (°C)': 1.0,
            'R²': 0.8,
            'Temp Range Compliance (%)': 95.0
        }
        
        # Prepare data
        metric_names = ['L1 Loss', 'PSNR (dB)', 'SSIM', 'MAE (°C)', 'RMSE (°C)', 'R²', 'Temp Range Compliance (%)']
        current_values = [
            current_results['l1_loss'],
            current_results['psnr'],
            current_results['ssim'],
            metrics['mae'],
            metrics['rmse'],
            metrics['r2'],
            metrics['temp_range_compliance']
        ]
        target_values = [targets[name] for name in metric_names]
        
        # Create bar plot comparing current vs target
        x = np.arange(len(metric_names))
        width = 0.35
        
        bars1 = axes[0, 0].bar(x - width/2, current_values, width, label='Current', alpha=0.8)
        bars2 = axes[0, 0].bar(x + width/2, target_values, width, label='Target', alpha=0.8)
        
        axes[0, 0].set_xlabel('Metrics')
        axes[0, 0].set_ylabel('Values')
        axes[0, 0].set_title('Current vs Target Metrics')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(metric_names, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Performance radar chart
        angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        
        # Normalize values for radar chart (0-1 scale)
        normalized_current = []
        normalized_target = []
        
        for i, (curr, targ) in enumerate(zip(current_values, target_values)):
            if metric_names[i] in ['L1 Loss', 'MAE (°C)', 'RMSE (°C)']:
                # Lower is better - invert
                norm_curr = max(0, 1 - (curr / (targ * 2)))
                norm_targ = 1
            else:
                # Higher is better
                if metric_names[i] == 'Temp Range Compliance (%)':
                    norm_curr = curr / 100
                    norm_targ = targ / 100
                else:
                    norm_curr = curr / targ if targ > 0 else 0
                    norm_targ = 1
            
            normalized_current.append(norm_curr)
            normalized_target.append(norm_targ)
        
        normalized_current = np.concatenate((normalized_current, [normalized_current[0]]))
        normalized_target = np.concatenate((normalized_target, [normalized_target[0]]))
        
        ax_radar = plt.subplot(2, 2, 2, projection='polar')
        ax_radar.plot(angles, normalized_current, 'o-', linewidth=2, label='Current', color='blue')
        ax_radar.fill(angles, normalized_current, alpha=0.25, color='blue')
        ax_radar.plot(angles, normalized_target, 'o-', linewidth=2, label='Target', color='red')
        ax_radar.fill(angles, normalized_target, alpha=0.25, color='red')
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(metric_names)
        ax_radar.set_ylim(0, 1.2)
        ax_radar.set_title('Performance Radar Chart')
        ax_radar.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Temperature statistics
        temp_stats = [
            f"Mean Predicted Temp: {metrics['mean_predicted_temp']:.2f}°C",
            f"Mean Target Temp: {metrics['mean_target_temp']:.2f}°C",
            f"Pred Temp Std: {metrics['temp_std_predicted']:.2f}°C",
            f"Target Temp Std: {metrics['temp_std_target']:.2f}°C",
            f"Range Compliance: {metrics['temp_range_compliance']:.1f}%"
        ]
        
        axes[1, 0].text(0.1, 0.9, '\n'.join(temp_stats), transform=axes[1, 0].transAxes,
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.8))
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].set_title('Temperature Statistics')
        axes[1, 0].axis('off')
        
        # Performance status
        status_text = []
        for name, curr, targ in zip(metric_names, current_values, target_values):
            if name in ['L1 Loss', 'MAE (°C)', 'RMSE (°C)']:
                status = curr
            else:
                status = curr
            status_text.append(f"{name}: {status}")
        
        axes[1, 1].text(0.1, 0.9, '\n'.join(status_text), transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle="round", facecolor='lightgreen', alpha=0.8))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_title('Performance Status')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Metrics summary plot saved to {save_path}")
    
    def evaluate(self, test_loader):
        """
        Enhanced evaluation with temperature analysis
        """
        logger.info("Starting enhanced evaluation with temperature analysis...")
        
        self.generator.eval()
        
        # Create directory for evaluation results
        os.makedirs(os.path.join(self.config.evaluation_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.config.evaluation_path, 'plots'), exist_ok=True)
        
        # Metrics
        l1_losses = []
        psnr_scores = []
        ssim_scores = []
        
        # Temperature arrays for analysis
        all_predicted_temps = []
        all_target_temps = []
        
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
                
                # Convert thermal values to temperatures
                if self.output_channels == 1:  # RGB to thermal prediction
                    predicted_temps = self.thermal_to_temperature(fake_B_np)
                    target_temps = self.thermal_to_temperature(real_B_np)
                    
                    all_predicted_temps.append(predicted_temps)
                    all_target_temps.append(target_temps)
                
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
        
        # Calculate temperature metrics if applicable
        temp_metrics = {}
        if all_predicted_temps:
            all_predicted_temps = np.concatenate(all_predicted_temps, axis=0)
            all_target_temps = np.concatenate(all_target_temps, axis=0)
            
            temp_metrics = self.calculate_temperature_metrics(all_predicted_temps, all_target_temps)
            
            # Create temperature visualizations
            temp_plot_path = os.path.join(self.config.evaluation_path, 'plots', 'temperature_analysis.png')
            self.create_temperature_visualizations(all_predicted_temps, all_target_temps, temp_plot_path)
        
        # Log and save results
        logger.info(f"Evaluation Results:")
        logger.info(f"L1 Loss: {avg_l1_loss:.4f}")
        logger.info(f"PSNR: {avg_psnr:.4f} dB")
        logger.info(f"SSIM: {avg_ssim:.4f}")
        
        if temp_metrics:
            logger.info(f"Temperature Metrics:")
            logger.info(f"MAE: {temp_metrics['mae']:.4f} °C")
            logger.info(f"RMSE: {temp_metrics['rmse']:.4f} °C")
            logger.info(f"R²: {temp_metrics['r2']:.4f}")
            logger.info(f"Temperature Range Compliance: {temp_metrics['temp_range_compliance']:.1f}%")
        
        # Combine all metrics
        metrics = {
            'l1_loss': float(avg_l1_loss),
            'psnr': float(avg_psnr),
            'ssim': float(avg_ssim),
            **temp_metrics
        }
        
        # Create comprehensive metrics visualization
        metrics_plot_path = os.path.join(self.config.evaluation_path, 'plots', 'metrics_summary.png')
        self.create_metrics_summary_plot(metrics, metrics_plot_path)
        
        # Save metrics to file
        with open(os.path.join(self.config.evaluation_path, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        return metrics

    # [Rest of the methods remain the same...]
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