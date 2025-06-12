# temperature_analysis.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

class TemperatureAnalyzer:
    """
    Specialized class for analyzing temperature prediction results
    """
    
    def __init__(self, temp_min=28.0, temp_max=30.0):
        self.temp_min = temp_min
        self.temp_max = temp_max
        
    def analyze_temperature_predictions(self, predicted_temps, target_temps, save_dir):
        """
        Comprehensive temperature analysis with marine-specific insights
        """
        # Flatten arrays
        pred_flat = predicted_temps.flatten()
        target_flat = target_temps.flatten()
        
        # Calculate comprehensive statistics
        stats_dict = self.calculate_comprehensive_stats(pred_flat, target_flat)
        
        # Create detailed visualizations
        self.create_detailed_temperature_plots(pred_flat, target_flat, save_dir)
        
        # Generate temperature report
        self.generate_temperature_report(stats_dict, save_dir)
        
        return stats_dict
    
    def calculate_comprehensive_stats(self, predicted, target):
        """
        Calculate comprehensive temperature statistics
        """
        errors = predicted - target
        abs_errors = np.abs(errors)
        
        # Basic statistics
        stats_dict = {
            'mean_predicted': np.mean(predicted),
            'mean_target': np.mean(target),
            'std_predicted': np.std(predicted),
            'std_target': np.std(target),
            'min_predicted': np.min(predicted),
            'max_predicted': np.max(predicted),
            'min_target': np.min(target),
            'max_target': np.max(target),
            
            # Error statistics
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'mae': np.mean(abs_errors),
            'rmse': np.sqrt(np.mean(errors**2)),
            'max_error': np.max(abs_errors),
            'min_error': np.min(abs_errors),
            
            # Correlation and agreement
            'correlation': np.corrcoef(predicted, target)[0, 1],
            'r2': stats.pearsonr(predicted, target)[0]**2,
            
            # Range compliance
            'temp_range_compliance': np.mean(
                (predicted >= self.temp_min) & (predicted <= self.temp_max)
            ) * 100,
            
            # Percentile errors
            'error_p25': np.percentile(abs_errors, 25),
            'error_p50': np.percentile(abs_errors, 50),
            'error_p75': np.percentile(abs_errors, 75),
            'error_p95': np.percentile(abs_errors, 95),
            
            # Temperature accuracy classes
            'accuracy_within_0_5C': np.mean(abs_errors <= 0.5) * 100,
            'accuracy_within_1_0C': np.mean(abs_errors <= 1.0) * 100,
            'accuracy_within_1_5C': np.mean(abs_errors <= 1.5) * 100,
        }
        
        return stats_dict
    
    def create_detailed_temperature_plots(self, predicted, target, save_dir):
        """
        Create detailed temperature analysis plots
        """
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Main scatter plot with density
        ax1 = plt.subplot(3, 4, 1)
        scatter = ax1.scatter(target, predicted, alpha=0.6, s=1, c=np.abs(predicted-target), 
                             cmap='viridis', label='Predictions')
        ax1.plot([self.temp_min, self.temp_max], [self.temp_min, self.temp_max], 'r--', lw=2, label='Perfect Prediction')
        ax1.axhline(self.temp_min, color='green', linestyle=':', alpha=0.7, label='Min Standard')
        ax1.axhline(self.temp_max, color='green', linestyle=':', alpha=0.7, label='Max Standard')
        ax1.axvline(self.temp_min, color='green', linestyle=':', alpha=0.7)
        ax1.axvline(self.temp_max, color='green', linestyle=':', alpha=0.7)
        ax1.set_xlabel('Target Temperature (°C)')
        ax1.set_ylabel('Predicted Temperature (°C)')
        ax1.set_title('Temperature Predictions vs Targets')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Absolute Error (°C)')
        
        # 2. Error distribution
        ax2 = plt.subplot(3, 4, 2)
        errors = predicted - target
        ax2.hist(errors, bins=50, alpha=0.7, density=True, color='orange')
        ax2.axvline(0, color='red', linestyle='--', label='Perfect Prediction')
        ax2.axvline(np.mean(errors), color='blue', linestyle='-', label=f'Mean Error: {np.mean(errors):.3f}°C')
        ax2.set_xlabel('Prediction Error (°C)')
        ax2.set_ylabel('Density')
        ax2.set_title('Error Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Temperature histograms
        ax3 = plt.subplot(3, 4, 3)
        ax3.hist(target, bins=50, alpha=0.7, label='Target', density=True, color='blue')
        ax3.hist(predicted, bins=50, alpha=0.7, label='Predicted', density=True, color='red')
        ax3.axvline(self.temp_min, color='green', linestyle='--', alpha=0.7, label='Standard Range')
        ax3.axvline(self.temp_max, color='green', linestyle='--', alpha=0.7)
        ax3.set_xlabel('Temperature (°C)')
        ax3.set_ylabel('Density')
        ax3.set_title('Temperature Distributions')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Box plot comparison
        ax4 = plt.subplot(3, 4, 4)
        box_data = [target, predicted]
        box_plot = ax4.boxplot(box_data, labels=['Target', 'Predicted'], patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightblue')
        box_plot['boxes'][1].set_facecolor('lightcoral')
        ax4.axhline(self.temp_min, color='green', linestyle='--', alpha=0.7)
        ax4.axhline(self.temp_max, color='green', linestyle='--', alpha=0.7)
        ax4.set_ylabel('Temperature (°C)')
        ax4.set_title('Temperature Range Comparison')
        ax4.grid(True, alpha=0.3)
        
        # 5. Residual plot
        ax5 = plt.subplot(3, 4, 5)
        ax5.scatter(target, errors, alpha=0.6, s=1)
        ax5.axhline(0, color='red', linestyle='--')
        ax5.set_xlabel('Target Temperature (°C)')
        ax5.set_ylabel('Residual (°C)')
        ax5.set_title('Residual Plot')
        ax5.grid(True, alpha=0.3)
        
        # 6. Absolute error vs target
        ax6 = plt.subplot(3, 4, 6)
        abs_errors = np.abs(errors)
        ax6.scatter(target, abs_errors, alpha=0.6, s=1, color='purple')
        ax6.axhline(0.5, color='red', linestyle='--', label='0.5°C threshold')
        ax6.axhline(1.0, color='orange', linestyle='--', label='1.0°C threshold')
        ax6.set_xlabel('Target Temperature (°C)')
        ax6.set_ylabel('Absolute Error (°C)')
        ax6.set_title('Absolute Error vs Target')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Cumulative error distribution
        ax7 = plt.subplot(3, 4, 7)
        sorted_abs_errors = np.sort(abs_errors)
        cumulative_pct = np.arange(1, len(sorted_abs_errors) + 1) / len(sorted_abs_errors) * 100
        ax7.plot(sorted_abs_errors, cumulative_pct)
        ax7.axvline(0.5, color='red', linestyle='--', label='0.5°C')
        ax7.axvline(1.0, color='orange', linestyle='--', label='1.0°C')
        ax7.set_xlabel('Absolute Error (°C)')
        ax7.set_ylabel('Cumulative Percentage (%)')
        ax7.set_title('Cumulative Error Distribution')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Q-Q plot for normality of errors
        ax8 = plt.subplot(3, 4, 8)
        stats.probplot(errors, dist="norm", plot=ax8)
        ax8.set_title('Q-Q Plot of Residuals')
        ax8.grid(True, alpha=0.3)
        
        # 9. Temperature accuracy by ranges
        ax9 = plt.subplot(3, 4, 9)
        temp_ranges = ['< 28°C', '28-29°C', '29-30°C', '> 30°C']
        range_masks = [
            target < 28.0,
            (target >= 28.0) & (target < 29.0),
            (target >= 29.0) & (target <= 30.0),
            target > 30.0
        ]
        
        mae_by_range = []
        for mask in range_masks:
            if np.any(mask):
                mae_by_range.append(np.mean(np.abs(errors[mask])))
            else:
                mae_by_range.append(0)
        
        bars = ax9.bar(temp_ranges, mae_by_range, color=['blue', 'green', 'green', 'red'], alpha=0.7)
        ax9.axhline(0.5, color='red', linestyle='--', label='Target MAE: 0.5°C')
        ax9.set_ylabel('MAE (°C)')
        ax9.set_title('Accuracy by Temperature Range')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 10. Correlation heatmap
        ax10 = plt.subplot(3, 4, 10)
        correlation_data = np.array([[1.0, np.corrcoef(target, predicted)[0, 1]], 
                                   [np.corrcoef(target, predicted)[0, 1], 1.0]])
        im = ax10.imshow(correlation_data, cmap='coolwarm', vmin=-1, vmax=1)
        ax10.set_xticks([0, 1])
        ax10.set_yticks([0, 1])
        ax10.set_xticklabels(['Target', 'Predicted'])
        ax10.set_yticklabels(['Target', 'Predicted'])
        ax10.set_title('Correlation Matrix')
        
        # Add correlation values to heatmap
        for i in range(2):
            for j in range(2):
                text = ax10.text(j, i, f'{correlation_data[i, j]:.3f}',
                               ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax10)
        
        # 11. Error statistics summary
        ax11 = plt.subplot(3, 4, 11)
        ax11.axis('off')
        
        # Calculate statistics for display
        mae = np.mean(abs_errors)
        rmse = np.sqrt(np.mean(errors**2))
        r2 = stats.pearsonr(predicted, target)[0]**2
        range_compliance = np.mean((predicted >= self.temp_min) & (predicted <= self.temp_max)) * 100
        
        stats_text = f"""
Temperature Statistics:
────────────────────
MAE: {mae:.3f}°C
RMSE: {rmse:.3f}°C
R²: {r2:.3f}
Range Compliance: {range_compliance:.1f}%

Accuracy within:
- 0.5°C: {np.mean(abs_errors <= 0.5)*100:.1f}%
- 1.0°C: {np.mean(abs_errors <= 1.0)*100:.1f}%
- 1.5°C: {np.mean(abs_errors <= 1.5)*100:.1f}%

Temperature Range:
- Min Pred: {np.min(predicted):.2f}°C
- Max Pred: {np.max(predicted):.2f}°C
- Mean Pred: {np.mean(predicted):.2f}°C
- Std Pred: {np.std(predicted):.2f}°C
        """
        
        ax11.text(0.1, 0.9, stats_text, transform=ax11.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        # 12. Performance indicators
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('off')
        
        # Performance status
        thresholds = {
            'MAE ≤ 0.5°C': mae <= 0.5,
            'RMSE ≤ 1.0°C': rmse <= 1.0,
            'R² ≥ 0.8': r2 >= 0.8,
            'Range Compliance ≥ 95%': range_compliance >= 95.0
        }
        
        status_text = "Performance Status:\n" + "─" * 20 + "\n"
        for criterion, passed in thresholds.items():
            status_icon = "✅" if passed else "❌"
            status_text += f"{status_icon} {criterion}\n"
        
        overall_score = sum(thresholds.values()) / len(thresholds) * 100
        status_text += f"\nOverall Score: {overall_score:.1f}%"
        
        color = 'lightgreen' if overall_score >= 75 else 'lightyellow' if overall_score >= 50 else 'lightcoral'
        
        ax12.text(0.1, 0.9, status_text, transform=ax12.transAxes, fontsize=11,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/detailed_temperature_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Detailed temperature analysis saved to {save_dir}/detailed_temperature_analysis.png")
    
    def generate_temperature_report(self, stats_dict, save_dir):
        """
        Generate a comprehensive temperature analysis report
        """
        report_content = f"""
# Temperature Prediction Analysis Report
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
This report provides a comprehensive analysis of the RGB-to-thermal temperature prediction model performance,
specifically evaluating predictions within the marine water temperature standard range of {self.temp_min}-{self.temp_max}°C.

## Model Performance Metrics

### Temperature Accuracy Metrics
- **Mean Absolute Error (MAE)**: {stats_dict['mae']:.4f}°C
  - Target: ≤ 0.5°C
  - Result: {stats_dict['mae']:.4f}°C

- **Root Mean Square Error (RMSE)**: {stats_dict['rmse']:.4f}°C
  - Target: ≤ 1.0°C
  - Result: {stats_dict['rmse']:.4f}°C

- **R-squared (R²)**: {stats_dict['r2']:.4f}
  - Target: ≥ 0.8
  - Result: {stats_dict['r2']:.4f}

### Temperature Range Compliance
- **Range Compliance**: {stats_dict['temp_range_compliance']:.2f}%
  - Target: ≥ 95%
  - Result: {stats_dict['temp_range_compliance']:.2f}%

### Temperature Distribution Analysis
- **Mean Predicted Temperature**: {stats_dict['mean_predicted']:.3f}°C
- **Mean Target Temperature**: {stats_dict['mean_target']:.3f}°C
- **Temperature Bias**: {stats_dict['mean_error']:.3f}°C
- **Temperature Std (Predicted)**: {stats_dict['std_predicted']:.3f}°C
- **Temperature Std (Target)**: {stats_dict['std_target']:.3f}°C

### Accuracy Classes
- **Within ±0.5°C**: {stats_dict['accuracy_within_0_5C']:.1f}%
- **Within ±1.0°C**: {stats_dict['accuracy_within_1_0C']:.1f}%
- **Within ±1.5°C**: {stats_dict['accuracy_within_1_5C']:.1f}%

### Error Statistics
- **Maximum Absolute Error**: {stats_dict['max_error']:.3f}°C
- **Minimum Absolute Error**: {stats_dict['min_error']:.3f}°C
- **Error Standard Deviation**: {stats_dict['std_error']:.3f}°C
- **95th Percentile Error**: {stats_dict['error_p95']:.3f}°C

## Marine Environmental Monitoring Suitability

### Strengths
"""

        # Add strengths based on performance
        if stats_dict['mae'] <= 0.5:
            report_content += "- Excellent temperature accuracy suitable for precise marine monitoring\n"
        if stats_dict['temp_range_compliance'] >= 90:
            report_content += "- Good compliance with marine temperature standards\n"
        if stats_dict['r2'] >= 0.7:
            report_content += "- Strong correlation between predicted and actual temperatures\n"

        report_content += "\n### Areas for Improvement\n"
        
        # Add improvements based on performance
        if stats_dict['mae'] > 0.5:
            report_content += "- Temperature accuracy needs improvement for precision marine applications\n"
        if stats_dict['temp_range_compliance'] < 95:
            report_content += "- Temperature range compliance should be enhanced\n"
        if stats_dict['r2'] < 0.8:
            report_content += "- Model correlation with ground truth requires strengthening\n"

        report_content += f"""
### Recommendations
1. **Model Training**: {"Continue current approach" if stats_dict['mae'] <= 0.5 else "Implement temperature-specific loss functions"}
2. **Data Quality**: {"Maintain current data standards" if stats_dict['r2'] >= 0.8 else "Review training data quality and diversity"}
3. **Calibration**: {"Current calibration adequate" if stats_dict['temp_range_compliance'] >= 95 else "Implement post-processing temperature calibration"}
4. **Deployment**: {"Ready for field deployment" if all([stats_dict['mae'] <= 0.5, stats_dict['rmse'] <= 1.0, stats_dict['r2'] >= 0.8]) else "Requires further development before deployment"}

## Conclusion
The temperature prediction model shows {'excellent' if stats_dict['mae'] <= 0.5 else 'good' if stats_dict['mae'] <= 1.0 else 'adequate'} performance for marine environmental monitoring applications.
{'The model meets all target criteria and is suitable for deployment.' if all([stats_dict['mae'] <= 0.5, stats_dict['rmse'] <= 1.0, stats_dict['r2'] >= 0.8, stats_dict['temp_range_compliance'] >= 95]) else 'Additional refinement is recommended before field deployment.'}

---
Report generated by DroneMEQ Temperature Analysis System
"""

        # Save report
        with open(f"{save_dir}/temperature_analysis_report.md", 'w') as f:
            f.write(report_content)
        
        print(f"Temperature analysis report saved to {save_dir}/temperature_analysis_report.md")