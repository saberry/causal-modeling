import semopy 
import pandas as pd

injury_data = pd.read_csv("causal_injury_data.csv")

model = """
    sleep_hours ~ stress_level + activity_level_hrs
    nutrition_score ~ stress_level + activity_level_hrs
    fatigue_index ~ sleep_hours + nutrition_score
    injury_risk_prob ~ fatigue_index
    injury_occurred ~ injury_risk_prob
"""

m = semopy.Model(model)
m.fit(injury_data)
g = semopy.semplot(m, "injury_path.png")
