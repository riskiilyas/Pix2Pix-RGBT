import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import time
import numpy as np
from tqdm import tqdm
from ML_TEMPERATURE_PREDICTION.entity.config_entity import ModelTrainerConfig
from ML_TEMPERATURE_PREDICTION.components.pix2pix_model import Generator, Discriminator
from ML_TEMPERATURE_PREDICTION.logging import logger

class ModelTrainer:
    """
    Model trainer component for training Pix2Pix models
    """
    def __init__(self, config: ModelTrainerConfig):
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
        
        # Initialize models
        self.generator = Generator(in_channels=self.input_channels, out_channels=self.output_channels).to(self.device)
        self.discriminator = Discriminator(in_channels=self.input_channels, out_channels=self.output_channels).to(self.device)
        
        # Initialize optimizers
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.config.lr, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.config.lr, betas=(0.5, 0.999))
        
        # Initialize loss functions
        self.criterion_GAN = nn.MSELoss()  # For adversarial loss
        self.criterion_pixelwise = nn.L1Loss()  # For pixel-wise loss
        
        # Lambda parameter for L1 loss
        self.lambda_pixel = 100
        
        # Create directory for sample images during training
        os.makedirs(os.path.join(self.config.root_dir, 'samples'), exist_ok=True)
        
        # Initialize training variables
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        
        # Load checkpoint if available
        self.load_checkpoint()
    
    def load_checkpoint(self):
        """
        Load the latest checkpoint if available
        """
        checkpoint_path = os.path.join(self.config.root_dir, 'checkpoints/latest.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.generator.load_state_dict(checkpoint['generator'])
            self.discriminator.load_state_dict(checkpoint['discriminator'])
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
            self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_loss = checkpoint['best_val_loss']
            
            logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    def save_checkpoint(self, epoch, is_best=False):
        """
        Save a model checkpoint
        """
        os.makedirs(os.path.join(self.config.root_dir, 'checkpoints'), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
            'best_val_loss': self.best_val_loss
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(self.config.root_dir, 'checkpoints/latest.pth'))
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, os.path.join(self.config.root_dir, 'checkpoints/best.pth'))
    
    def save_some_examples(self, epoch, dataloader):
        """
        Save some example outputs during training
        """
        batch = next(iter(dataloader))
        real_A = batch['A'].to(self.device)
        real_B = batch['B'].to(self.device)
        
        self.generator.eval()
        with torch.no_grad():
            fake_B = self.generator(real_A)
            # Rescale from [-1, 1] to [0, 1]
            real_A = (real_A + 1) / 2
            real_B = (real_B + 1) / 2
            fake_B = (fake_B + 1) / 2
            
            # Concatenate images for visualization
            for i in range(min(5, real_A.size(0))):
                # For thermal images (1 channel), repeat to 3 channels for proper visualization
                if self.output_channels == 1:
                    fake_B_rgb = fake_B[i].repeat(3, 1, 1)
                    real_B_rgb = real_B[i].repeat(3, 1, 1)
                else:
                    fake_B_rgb = fake_B[i]
                    real_B_rgb = real_B[i]
                
                # Concatenate images side by side
                combined = torch.cat((real_A[i], fake_B_rgb, real_B_rgb), 2)
                save_image(combined, os.path.join(self.config.root_dir, f"samples/{epoch}_{i}.png"))
        
        self.generator.train()
    
    def train_epoch(self, train_loader):
        """
        Train the model for one epoch
        """
        self.generator.train()
        self.discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            # Get real images
            real_A = batch['A'].to(self.device)  # Input image (RGB or Thermal)
            real_B = batch['B'].to(self.device)  # Target image (Thermal or RGB)
            
            # Forward pass through discriminator to get output size
            with torch.no_grad():
                test_output = self.discriminator(real_A, real_B)
                patch_h, patch_w = test_output.size(2), test_output.size(3)
            
            # Adversarial ground truths (1 for real, 0 for fake)
            valid = torch.ones((real_A.size(0), 1, patch_h, patch_w), requires_grad=False).to(self.device)
            fake = torch.zeros((real_A.size(0), 1, patch_h, patch_w), requires_grad=False).to(self.device)
            
            # -----------------
            # Train Generator
            # -----------------
            self.optimizer_G.zero_grad()
            
            # Generate fake images
            fake_B = self.generator(real_A)
            
            # GAN loss
            pred_fake = self.discriminator(real_A, fake_B)
            loss_GAN = self.criterion_GAN(pred_fake, valid)
            
            # Pixel-wise loss
            loss_pixel = self.criterion_pixelwise(fake_B, real_B)
            
            # Total generator loss
            loss_G = loss_GAN + self.lambda_pixel * loss_pixel
            
            loss_G.backward()
            self.optimizer_G.step()
            
            # -------------------
            # Train Discriminator
            # -------------------
            self.optimizer_D.zero_grad()
            
            # Real loss
            pred_real = self.discriminator(real_A, real_B)
            loss_real = self.criterion_GAN(pred_real, valid)
            
            # Fake loss
            pred_fake = self.discriminator(real_A, fake_B.detach())
            loss_fake = self.criterion_GAN(pred_fake, fake)
            
            # Total discriminator loss
            loss_D = (loss_real + loss_fake) / 2
            
            loss_D.backward()
            self.optimizer_D.step()
            
            # Update running losses
            epoch_g_loss += loss_G.item()
            epoch_d_loss += loss_D.item()
        
        # Return average losses
        return epoch_g_loss / len(train_loader), epoch_d_loss / len(train_loader)
    
    def validate(self, val_loader):
        """
        Validate the model
        """
        self.generator.eval()
        self.discriminator.eval()
        
        val_g_loss = 0
        val_d_loss = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
                # Get real images
                real_A = batch['A'].to(self.device)  # Input image
                real_B = batch['B'].to(self.device)  # Target image
                
                # Get discriminator output size
                test_output = self.discriminator(real_A, real_B)
                patch_h, patch_w = test_output.size(2), test_output.size(3)
                
                # Adversarial ground truths
                valid = torch.ones((real_A.size(0), 1, patch_h, patch_w), requires_grad=False).to(self.device)
                fake = torch.zeros((real_A.size(0), 1, patch_h, patch_w), requires_grad=False).to(self.device)
                
                # Generate fake images
                fake_B = self.generator(real_A)
                
                # Generator loss
                pred_fake = self.discriminator(real_A, fake_B)
                loss_GAN = self.criterion_GAN(pred_fake, valid)
                loss_pixel = self.criterion_pixelwise(fake_B, real_B)
                loss_G = loss_GAN + self.lambda_pixel * loss_pixel
                
                # Discriminator loss
                pred_real = self.discriminator(real_A, real_B)
                loss_real = self.criterion_GAN(pred_real, valid)
                
                pred_fake = self.discriminator(real_A, fake_B)
                loss_fake = self.criterion_GAN(pred_fake, fake)
                
                loss_D = (loss_real + loss_fake) / 2
                
                # Update running losses
                val_g_loss += loss_G.item()
                val_d_loss += loss_D.item()
        
        # Return average losses
        return val_g_loss / len(val_loader), val_d_loss / len(val_loader)
    
    def train(self, train_loader, val_loader):
        """
        Train the model for multiple epochs
        """
        logger.info(f"Starting training from epoch {self.start_epoch} to {self.config.num_epochs}")
        
        for epoch in range(self.start_epoch, self.config.num_epochs):
            # Train for one epoch
            start_time = time.time()
            train_g_loss, train_d_loss = self.train_epoch(train_loader)
            val_g_loss, val_d_loss = self.validate(val_loader)
            end_time = time.time()
            
            # Log results
            logger.info(f"Epoch [{epoch+1}/{self.config.num_epochs}] - "
                      f"Time: {end_time - start_time:.2f}s - "
                      f"Train G Loss: {train_g_loss:.4f}, Train D Loss: {train_d_loss:.4f} - "
                      f"Val G Loss: {val_g_loss:.4f}, Val D Loss: {val_d_loss:.4f}")
            
            # Save checkpoint
            is_best = val_g_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_g_loss
                logger.info(f"New best validation loss: {self.best_val_loss:.4f}")
            
            self.save_checkpoint(epoch, is_best)
            
            # Save example outputs
            if epoch % 10 == 0:
                self.save_some_examples(epoch, val_loader)
        
        logger.info("Training completed!")
    
    def test(self, test_loader):
        """
        Test the model on the test set
        """
        logger.info("Starting testing...")
        
        # Load the best model
        best_checkpoint_path = os.path.join(self.config.root_dir, 'checkpoints/best.pth')
        if os.path.exists(best_checkpoint_path):
            checkpoint = torch.load(best_checkpoint_path, map_location=self.device)
            self.generator.load_state_dict(checkpoint['generator'])
            logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")
        
        self.generator.eval()
        
        # Create directory for test outputs
        os.makedirs(os.path.join(self.config.root_dir, 'test_results'), exist_ok=True)
        
        test_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
                # Get real images
                real_A = batch['A'].to(self.device)  # Input image
                real_B = batch['B'].to(self.device)  # Target image
                filenames = batch['filename']
                
                # Generate fake images
                fake_B = self.generator(real_A)
                
                # Calculate L1 loss
                loss = self.criterion_pixelwise(fake_B, real_B)
                test_loss += loss.item()
                
                # Rescale images from [-1, 1] to [0, 1]
                real_A = (real_A + 1) / 2
                real_B = (real_B + 1) / 2
                fake_B = (fake_B + 1) / 2
                
                # Save results
                for i in range(real_A.size(0)):
                    # For thermal images (1 channel), repeat to 3 channels for visualization
                    if self.output_channels == 1:
                        fake_B_vis = fake_B[i].repeat(3, 1, 1)
                        real_B_vis = real_B[i].repeat(3, 1, 1)
                    else:
                        fake_B_vis = fake_B[i]
                        real_B_vis = real_B[i]
                    
                    # Concatenate images side by side (input, generated, target)
                    combined = torch.cat((real_A[i], fake_B_vis, real_B_vis), 2)
                    save_image(combined, os.path.join(self.config.root_dir, f"test_results/{filenames[i]}.png"))
        
        # Report average test loss
        avg_test_loss = test_loss / len(test_loader)
        logger.info(f"Test Loss: {avg_test_loss:.4f}")
        
        return avg_test_loss