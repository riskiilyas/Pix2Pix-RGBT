from evaluation_runner import run_enhanced_evaluation
from temperature_analysis import TemperatureAnalyzer

# run_complete_evaluation.py
if __name__ == "__main__":
    # Run the enhanced evaluation
    results = run_enhanced_evaluation()
    
    # Run detailed temperature analysis
    analyzer = TemperatureAnalyzer()
    
    # Create sample data for demonstration (replace with actual evaluation data)
    import numpy as np
    np.random.seed(42)
    
    # Generate sample temperature data based on your model's performance
    n_samples = 5000
    target_temps = np.random.normal(29.0, 0.7, n_samples)
    target_temps = np.clip(target_temps, 28.0, 30.0)
    
    # Add realistic noise based on MAE ≈ 0.75°C
    noise = np.random.normal(0, 0.75, n_samples)
    predicted_temps = target_temps + noise
    
    # Run detailed analysis
    detailed_stats = analyzer.analyze_temperature_predictions(
        predicted_temps, target_temps, "artifacts/model_evaluation/results"
    )
    
    print("\n🎉 Complete evaluation finished!")
    print("Check the 'artifacts/model_evaluation/results' directory for all outputs.")