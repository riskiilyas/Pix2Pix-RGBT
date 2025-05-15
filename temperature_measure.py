import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os.path


class ThermalImageAnalyzer:
    """
    A class for analyzing thermal images and estimating temperature distribution.
    """
    
    def __init__(self, thermal_image_path, rgb_image_path=None, 
                 assumed_min_temp=28, assumed_max_temp=30):
        """
        Initialize the thermal image analyzer.
        
        Args:
            thermal_image_path: Path to the thermal image file (16-bit)
            rgb_image_path: Path to the corresponding RGB image (optional)
            assumed_min_temp: Estimated minimum temperature in the scene (Celsius)
            assumed_max_temp: Estimated maximum temperature in the scene (Celsius)
        """
        self.thermal_image_path = thermal_image_path
        self.rgb_image_path = rgb_image_path
        self.assumed_min_temp = assumed_min_temp
        self.assumed_max_temp = assumed_max_temp
        
        # Load thermal image and get dimensions
        self.thermal_image = self._load_thermal_image()
        self.height, self.width = self.thermal_image.shape[:2]
        self.center_x, self.center_y = self.width // 2, self.height // 2
        
        # Load RGB image (using dimensions from thermal image)
        self.rgb_image = self._load_rgb_image()
        
        # Process the thermal data
        self.temp_scale = self._calculate_temperature_scale()
        self.stats = self._calculate_temperature_stats()
        
        # Create visualizations
        self.visualizations = self._create_visualizations()
        
    def _load_thermal_image(self):
        """Load the 16-bit thermal image."""
        thermal_image = cv2.imread(self.thermal_image_path, cv2.IMREAD_ANYDEPTH)
        if thermal_image is None:
            raise FileNotFoundError(f"Could not read thermal image at {self.thermal_image_path}")
        return thermal_image
    
    def _load_rgb_image(self):
        """Load and resize the RGB image if provided."""
        if not self.rgb_image_path or not os.path.exists(self.rgb_image_path):
            return None
            
        rgb_image = cv2.imread(self.rgb_image_path)
        if rgb_image is None:
            return None
            
        # Dimensions should be defined already by thermal image loading
        # Resize RGB image to match thermal image dimensions
        return cv2.resize(rgb_image, (self.width, self.height))
    
    def _calculate_temperature_scale(self):
        """Map raw thermal values to estimated temperature range."""
        # Use percentiles to avoid outliers
        p1 = np.percentile(self.thermal_image, 1)
        p99 = np.percentile(self.thermal_image, 99)
        
        # Linear mapping from raw values to temperature range
        temp_range = self.assumed_max_temp - self.assumed_min_temp
        return (self.thermal_image - p1) / (p99 - p1) * temp_range + self.assumed_min_temp
    
    def _calculate_temperature_stats(self):
        """Calculate temperature statistics."""
        return {
            'center': self.temp_scale[self.center_y, self.center_x],
            'min': np.min(self.temp_scale),
            'max': np.max(self.temp_scale),
            'mean': np.mean(self.temp_scale)
        }
    
    def _create_visualizations(self):
        """Create various visualizations of the thermal data."""
        # Normalize thermal image to 8-bit for visualization
        normalized = cv2.normalize(self.thermal_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # Create grayscale version
        grayscale = normalized.copy()
        
        # Apply thermal colormap
        thermal_colored = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
        
        # Add markers and annotations to visualizations
        thermal_with_markers = self._add_markers(thermal_colored.copy(), color=(0, 255, 0))
        grayscale_with_markers = self._add_markers(
            cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR), 
            color=(0, 0, 255)
        )
        
        return {
            'normalized': normalized,
            'grayscale': grayscale,
            'thermal_colored': thermal_colored,
            'thermal_with_markers': thermal_with_markers,
            'grayscale_with_markers': grayscale_with_markers
        }
    
    def _add_markers(self, image, color):
        """Add center point marker and temperature annotation to image."""
        # Add center point circle
        cv2.circle(image, (self.center_x, self.center_y), 5, color, 2)
        
        # Add temperature text
        cv2.putText(
            image, 
            f"{self.stats['center']:.2f} C", 
            (self.center_x + 10, self.center_y), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            color, 
            2
        )
        return image
    
    def get_custom_colormap(self):
        """Create a custom temperature colormap."""
        colors = [(0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]  # Blue to Red
        return LinearSegmentedColormap.from_list('thermal_cmap', colors, N=256)
    
    def display_results(self):
        """Display analysis results in a clean, organized layout."""
        # Create a figure with a 3x2 grid layout
        fig = plt.figure(figsize=(16, 12))
        
        # Top row: All images (4 images in a row)
        # Temperature Map
        ax_temp_map = plt.subplot2grid((3, 4), (0, 0), colspan=1, rowspan=1)
        temp_img = ax_temp_map.imshow(self.temp_scale, cmap=self.get_custom_colormap())
        ax_temp_map.set_title("Temperature Map", fontsize=12, fontweight='bold')
        ax_temp_map.axis('off')
        cbar = plt.colorbar(temp_img, ax=ax_temp_map, fraction=0.046, pad=0.04)
        cbar.set_label('°C', fontsize=10)
        
        # RGB image
        ax_rgb = plt.subplot2grid((3, 4), (0, 1), colspan=1, rowspan=1)
        if self.rgb_image is not None:
            ax_rgb.imshow(cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2RGB))
            ax_rgb.set_title("RGB Image", fontsize=12, fontweight='bold')
        else:
            ax_rgb.text(0.5, 0.5, "RGB Not Available", 
                      ha='center', va='center', fontsize=12)
        ax_rgb.axis('off')
        
        # Grayscale thermal image
        ax_gray = plt.subplot2grid((3, 4), (0, 2), colspan=1, rowspan=1)
        ax_gray.imshow(self.visualizations['grayscale'], cmap='gray')
        ax_gray.set_title("Predicted Grayscale Thermal", fontsize=12, fontweight='bold')
        ax_gray.axis('off')

        # Thermal colored image
        ax_thermal = plt.subplot2grid((3, 4), (0, 3), colspan=1, rowspan=1)
        ax_thermal.imshow(cv2.cvtColor(self.visualizations['thermal_with_markers'], cv2.COLOR_BGR2RGB))
        ax_thermal.set_title("Predicted Thermal Image", fontsize=12, fontweight='bold')
        ax_thermal.axis('off')
        
        # Temperature histogram (spans entire second row)
        ax_hist = plt.subplot2grid((3, 4), (1, 0), colspan=4, rowspan=1)
        ax_hist.hist(self.temp_scale.ravel(), bins=40, color='skyblue', 
                  edgecolor='black', alpha=0.7)
        
        # Add lines for center and mean temperature
        ax_hist.axvline(self.stats['center'], color='r', linestyle='dashed', linewidth=2, 
                     label=f"Center: {self.stats['center']:.2f}°C")
        ax_hist.axvline(self.stats['mean'], color='g', linestyle='dashed', linewidth=2, 
                     label=f"Mean: {self.stats['mean']:.2f}°C")
        
        ax_hist.set_xlabel("Temperature (°C)", fontsize=10)
        ax_hist.set_ylabel("Frequency", fontsize=10)
        ax_hist.set_title("Temperature Distribution", fontsize=12, fontweight='bold')
        ax_hist.legend(fontsize=9)
        
        # Stats panel (spans left half of third row)
        ax_stats = plt.subplot2grid((3, 4), (2, 0), colspan=2, rowspan=1)
        ax_stats.axis('off')
        
        # Add temperature statistics as text
        stats_text = (
            f"Temperature Statistics\n"
            f"------------------------\n"
            f"Center: {self.stats['center']:.2f}°C\n"
            f"Mean: {self.stats['mean']:.2f}°C\n"
            f"Min: {self.stats['min']:.2f}°C\n"
            f"Max: {self.stats['max']:.2f}°C\n\n"
            f"Standard Range: {self.assumed_min_temp}-{self.assumed_max_temp}°C\n"
            f"------------------------\n"
            f"These are approximate estimation."
        )
        ax_stats.text(0.1, 0.5, stats_text, fontsize=11,
                     va='center', family='monospace')
        
        # Add information about the center point (spans right half of third row)
        ax_info = plt.subplot2grid((3, 4), (2, 2), colspan=2, rowspan=1)
        ax_info.axis('off')
        
        # Display image metadata if available
        file_info = os.path.basename(self.thermal_image_path)
        info_text = (
            f"Image Information\n"
            f"------------------------\n"
            f"File: {file_info}\n"
            f"Size: {self.width} x {self.height} pixels\n"
            f"Center point: ({self.center_x}, {self.center_y})\n"
            f"------------------------\n"
            f"Green/red markers indicate the\n"
            f"center measurement point."
        )
        ax_info.text(0.1, 0.5, info_text, fontsize=11,
                    va='center', family='monospace')
        
        # Add a title to the entire figure
        fig.suptitle("Thermal Image Analysis", fontsize=16, fontweight='bold', y=0.98)
        
        # Show the plot
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the main title
        plt.show()
        
    def _display_with_rgb(self):
        """Legacy visualization layout with RGB image included."""
        # This method is kept for reference but not used in the new display
        pass
    
    def _display_without_rgb(self):
        """Legacy visualization layout without RGB image."""
        # This method is kept for reference but not used in the new display
        pass
    
    def _plot_temperature_histogram(self):
        """Plot histogram of temperature distribution."""
        plt.hist(self.temp_scale.ravel(), bins=50, color='skyblue', 
                 edgecolor='black', alpha=0.7)
        
        # Add lines for center and mean temperature
        plt.axvline(self.stats['center'], color='r', linestyle='dashed', linewidth=2, 
                   label=f"Center: {self.stats['center']:.2f}°C")
        plt.axvline(self.stats['mean'], color='g', linestyle='dashed', linewidth=2, 
                   label=f"Mean: {self.stats['mean']:.2f}°C")
        
        plt.xlabel("Estimated Temperature (°C)")
        plt.ylabel("Frequency")
        plt.title("Temperature Distribution")
        plt.legend()
    
    def _print_temperature_info(self):
        """Print temperature statistics to console."""
        print("\nEstimated Temperature Information:")
        print(f"Center temperature: {self.stats['center']:.2f}°C")
        print(f"Minimum temperature: {self.stats['min']:.2f}°C")
        print(f"Maximum temperature: {self.stats['max']:.2f}°C")
        print(f"Mean temperature: {self.stats['mean']:.2f}°C")
        print("\nIMPORTANT NOTE: These are estimated values based on assumed temperature range.")
        print(f"Standard temperature range: {self.assumed_min_temp}°C to {self.assumed_max_temp}°C")
        print("These are approximations and not precise measurements.")
    
    def save_images(self, output_dir="."):
        """Save visualization images to disk."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        cv2.imwrite(f"{output_dir}/thermal_viz.png", self.visualizations['thermal_with_markers'])
        cv2.imwrite(f"{output_dir}/grayscale_8bit.png", self.visualizations['grayscale'])
        
        # Save temperature map with colormap applied
        plt.figure(figsize=(8, 6))
        plt.imshow(self.temp_scale, cmap=self.get_custom_colormap())
        plt.colorbar(label='Estimated Temperature (°C)')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/temperature_map.png")
        plt.close()
        
    def get_results(self):
        """Return analysis results as a dictionary."""
        return {
            'temp_scale': self.temp_scale,
            'visualizations': self.visualizations,
            'rgb_image': self.rgb_image,
            'stats': self.stats
        }


# Example usage
def analyze_thermal_image(thermal_path, rgb_path=None, min_temp=28, max_temp=30, save_images=False):
    """
    Analyze a thermal image and display the results.
    
    Args:
        thermal_path: Path to the thermal image file
        rgb_path: Path to the corresponding RGB image (optional)
        min_temp: Assumed minimum temperature in Celsius
        max_temp: Assumed maximum temperature in Celsius
        save_images: Whether to save visualization images to disk
    """
    analyzer = ThermalImageAnalyzer(
        thermal_path, 
        rgb_image_path=rgb_path,
        assumed_min_temp=min_temp,
        assumed_max_temp=max_temp
    )
    
    analyzer.display_results()
    
    if save_images:
        analyzer.save_images()
    
    return analyzer.get_results()


# Example execution
if __name__ == "__main__":
    name_image = "test18"

    thermal_path = f"output/{name_image}.tiff"
    rgb_path = f"input/{name_image}.jpg"
    
    # For morning/day scenes, water is typically cooler than surroundings
    results = analyze_thermal_image(
        thermal_path, 
        rgb_path=rgb_path, 
        min_temp=28, 
        max_temp=30,
        save_images=False
    )