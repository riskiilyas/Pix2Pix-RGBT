# RGB-Thermal Image Translation using Pix2Pix

This project implements a Pix2Pix GAN model for translating between RGB and thermal images. The model can be trained to translate RGB images to thermal images (rgb2thermal) or thermal images to RGB images (thermal2rgb).

## Download the Pre-trained Model

You can download the pre-trained model from the link below:

[Download Pre-trained Model (best.pth)](https://drive.google.com/file/d/1-CklZwAS-zRLOK3jyUGfQcDsQC6eu1a4/view?usp=sharing)

After downloading the model, place it in the following directory:
```
artifacts/model_trainer/checkpoints/best.pth
```

## Project Structure

The project follows a modular structure:

```
ML_TEMPERATURE_PREDICTION/
├── .github/workflows/
├── artifacts/               # Generated during execution
├── config/                  # Configuration files
│   └── config.yaml
├── data/                    # Data directory
│   ├── rgb/                 # RGB images in JPG format
│   └── thermal/             # Thermal images in 16-bit TIFF format
├── input/                   # Input images for prediction
├── logs/                    # Generated log files
├── output/                  # Output predictions
├── src/                     # Source code
│   └── ML_TEMPERATURE_PREDICTION/
│       ├── components/      # Model components
│       ├── config/          # Configuration management
│       ├── constants/       # Constants
│       ├── entity/          # Data entities
│       ├── logging/         # Logging setup
│       ├── pipeline/        # Pipeline modules
│       └── utils/           # Utility functions
├── app.py                   # Streamlit web application
├── main.py                  # Main entry point
├── params.yaml              # Model parameters
├── requirements.txt         # Dependencies
└── setup.py                 # Package setup
```

## Dataset Requirements

- RGB Images: Standard RGB images in JPG format
- Thermal Images: 16-bit grayscale TIFF files
- Both RGB and thermal images should be paired with the same filename (e.g., image1.jpg and image1.tiff)
- Place RGB images in the `data/rgb/` directory and thermal images in the `data/thermal/` directory

## Installation

1. Clone the repository
2. Create a virtual environment and activate it
3. Install the dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

### Data Preparation

1. Place your paired RGB (JPG) and thermal (TIFF) images in the `data/rgb/` and `data/thermal/` directories respectively
2. Ensure each pair has the same filename (e.g., img001.jpg and img001.tiff)

### Training and Evaluation

Run the complete pipeline:

```bash
python main.py
```

Or run specific stages:

```bash
# Data ingestion (splits data into train/val/test)
python main.py --stage data_ingestion

# Model training
python main.py --stage model_training

# Model evaluation
python main.py --stage model_evaluation

# Prediction
python main.py --stage prediction
```

### Web Interface

Launch the Streamlit web application:

```bash
streamlit run app.py
```

The web interface allows you to:
- Select the translation direction (RGB to thermal or thermal to RGB)
- Upload an image for translation
- View the input and generated output
- Download the result

## Model Architecture

The model uses Pix2Pix GAN architecture with:
- Generator: U-Net with skip connections
- Discriminator: PatchGAN discriminator
- Loss: Combination of adversarial loss and L1 loss

## Customization

You can customize the model parameters in `params.yaml`:

```yaml
# Data split ratios
TRAIN_RATIO: 0.7
VAL_RATIO: 0.15
TEST_RATIO: 0.15
RANDOM_STATE: 42

# Data preprocessing
IMAGE_SIZE: 256
BATCH_SIZE: 16
NUM_WORKERS: 4

# Model parameters
DIRECTION: "rgb2thermal"  # or "thermal2rgb"
NUM_EPOCHS: 200
LEARNING_RATE: 0.0002
```

## License

This project is licensed under the MIT License.
