import pandas as pd

# Load the CSV
df = pd.read_csv("./full_walkability_data.csv")

# Group by census tract and calculate the average of model_score_normalized
avg_scores = df.groupby("census_tract")["model_score_normalized"].mean().reset_index()

# Optionally rename the column
avg_scores = avg_scores.rename(columns={"model_score_normalized": "avg_model_score_normalized"})

# Save to a new CSV
avg_scores.to_csv("average_scores_by_tract.csv", index=False)