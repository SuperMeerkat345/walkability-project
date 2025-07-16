def categorize_score(score):
    if score < 1:
        return "very_low"
    elif score < 2.5:
        return "low"
    elif score < 3.5:
        return "medium"
    elif score < 5:
        return "high"
    else: # 5+
        return "very_high"