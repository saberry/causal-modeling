import numpy as np
import pandas as pd

def generate_causal_injury_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    
    # 1. Exogenous Variables
    stress_level = np.clip(np.random.normal(loc=5.0, scale=1.5, size=n_samples), 1, 10)
    activity_level = np.clip(np.random.gamma(shape=3.0, scale=2.0, size=n_samples), 1, 20)
    
    # 2. Endogenous Mediators (Causal Dependencies)
    # High stress reduces sleep duration
    sleep_hours = np.clip(8.5 - 0.35 * stress_level + np.random.normal(0, 0.75, size=n_samples), 3, 10)
    
    # High stress negatively affects nutritional choices
    nutrition_score = np.clip(85 - 3.5 * stress_level + np.random.normal(0, 8.0, size=n_samples), 20, 100)
    
    # Fatigue aggregates workload, insufficient sleep, stress, and low nutrition
    fatigue_index = np.clip(
        15.0 + 3.5 * activity_level - 4.0 * sleep_hours + 2.5 * stress_level - 0.2 * nutrition_score 
        + np.random.normal(0, 5.0, size=n_samples), 
        0, 100
    )
    
    # 3. Outcome Generation (Logistics model for injury probability)
    logits = -5.0 + 0.06 * fatigue_index + 0.15 * activity_level - 0.02 * nutrition_score
    prob_injury = 1 / (1 + np.exp(-logits))
    injury_occurred = np.random.binomial(1, prob_injury)
    
    return pd.DataFrame({
        'stress_level': np.round(stress_level, 2),
        'activity_level_hrs': np.round(activity_level, 2),
        'sleep_hours': np.round(sleep_hours, 2),
        'nutrition_score': np.round(nutrition_score, 1),
        'fatigue_index': np.round(fatigue_index, 1),
        'injury_risk_prob': np.round(prob_injury, 4),
        'injury_occurred': injury_occurred
    })

# Example usage:
df = generate_causal_injury_data(n_samples=1000)
df.to_csv("causal_injury_data.csv", index=False)