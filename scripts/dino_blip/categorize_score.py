def categorize_score(score):
    if score < 0.5:
        return "very_low"
    elif score < 2:
        return "low"
    elif score < 3:
        return "medium"
    elif score < 4:
        return "high"
    else: # 4+
        return "very_high"

# data["model_label"] = [categorize_score(score) for score in data["model_score"]]

normalization_dic = {
    "very_low": 1,
    "low": 2,
    "medium": 3,
    "high": 4,
    "very_high": 5
}

# data["model_score_normalized"] = [normalization_dic[label] for label in data["model_label"]]