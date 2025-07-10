import pandas as pd
from clip_runner_debug import labelImgs

COL_TO_WRITE_TO = "label_model"
SCORES_COL = "scores"

def begin_labeling():
    data = pd.read_csv('./full_validation.csv')
    paths = data["path"].tolist()

    # Process images through CLIP
    clip_labels = labelImgs(paths)

    data[COL_TO_WRITE_TO] = clip_labels[0]
    data[SCORES_COL] = clip_labels[1]

    print(clip_labels[1])

    data.to_csv('./full_validation.csv', index=False)

    print("Exiting without any problems!")
    return 1

begin_labeling()