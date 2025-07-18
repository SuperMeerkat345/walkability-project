# Need to rename the column with scores from:
# label_model --> model_score

# Then categorize:
# categorize_score(model_score)

# Then normalize:
# very_low: 1
# low: 2
# medium: 3
# high: 4
# very_high:5

# Find average by Census tract and summarize 

# Then graph by census tract 
# Heatmap

# Find other heatmaps and correlate


import pandas as pd

data = pd.read_csv("./full_walkability_data.csv")

# put the stuff in "label_model" column into model_label
data.drop(
    labels = ["model_label"],
    axis=1,
    inplace=True
)
data = data.rename(columns={
    "label_model": "model_label"
})

# reorder the broken up dataframe
data = data[["index", "path", "model_label", "model_score", "model_score_normalized", "census_tract"]]

# fix broken index column
new_index_column = range(data.shape[0])
data["index"] = new_index_column

########################################
# START DATA MANIPULATION
###################################



