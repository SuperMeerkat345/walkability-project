import pandas as pd

# Load the CSV
df = pd.read_csv("./csvs/full_walkability_data.csv")

# Group by census tract and calculate the average of model_score_normalized
avg_scores = df.groupby("census_tract")["model_score"].mean().reset_index()

# Optionally rename the column
avg_scores = avg_scores.rename(columns={"model_score": "avg_model_score"})

# Save to a new CSV
avg_scores.to_csv("./csvs/average_scores_by_tract.csv", index=False)

print("Data transfered")