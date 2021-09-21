import pandas as pd
from file_config import *
from udiva import UDIVA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
import numpy as np

# A baseline Random Forest model using highly correlated metadata features
# Set the parameters by cross-validation
rf_tuned_params = [{
                    'max_depth': [1, 2, 3, 5, 10],
                    'oob_score': [True],
}]

metadata = {SETS.train: UDIVA.get_metadata(SETS.train),
            SETS.val: UDIVA.get_metadata(SETS.val),
            SETS.test: UDIVA.get_metadata(SETS.test),

            }
# X[0] => training data, X[1] => validation data, y[0] => train labels, y[1] => validation labels
X = []; y = []

def return_mood_features(_set=SETS.train):
    mood_features = []
    d = metadata[_set]
    print(d['HAPPY'].shape)
    for mood, name in [[d['FATIGUE'], 'FATIGUE'], [d['GOOD'], 'GOOD'], [d['BAD'], 'BAD'], [d['HAPPY'], 'HAPPY'], [d['SAD'], 'SAD'],
                       [d['FRIENDLY'], 'FRIENDLY'], [d['UNFRIENDLY'], 'UNFRIENDLY'], [d['TENSE'], 'TENSE'], [d['RELAXED'], 'RELAXED']]:
        for part in [1, 2]:
            data = get_variation(mood)[part - 1]
            mood_features.append((data))
    return pd.DataFrame(mood_features).T

def get_variation(mood, average = True):
    average_part1 = []
    average_part2 = []
    for subject in mood:
        part_1 = []
        part_2 = []
        for session in subject:
            part_1.append(session['PART1'][0] - session['PART1'][1])
            part_2.append(session['PART2'][0] - session['PART2'][1])
        if average:
            average_part1.append(np.mean(part_1))
            average_part2.append(np.mean(part_2))
        else:
            average_part1.append(part_1)
            average_part2.append(part_2)
    return average_part1,  average_part2

if __name__ == "__main__":
    for split in [SETS.train, SETS.val, SETS.test]:
        X.append(metadata[split][["AGE", "GENDER", "NUM_SESSIONS", 'ID']])
        #X[-1] = pd.concat([X[-1], return_mood_features(split)], axis=1)
        if "OPENMINDEDNESS_Z" in metadata[split]:
            y.append(metadata[split][["OPENMINDEDNESS_Z", "CONSCIENTIOUSNESS_Z", "EXTRAVERSION_Z", "AGREEABLENESS_Z", "NEGATIVEEMOTIONALITY_Z"]])
    print(metadata[SETS.val].columns)
    # pd.DataFrame(X[0]).to_csv("../features/all_metadata_train.csv", index=False)
    # pd.DataFrame(X[1]).to_csv("../features/all_metadata_val.csv", index=False)
    # pd.DataFrame(X[2]).to_csv("../features/all_metadata_test.csv", index=False)

    train = metadata[SETS.train]
    grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=rf_tuned_params,
                               cv=3, n_jobs=-1, verbose=1, scoring='neg_mean_squared_error')
    print("RF training will start...",  X[0].shape)
    grid_search.fit(X[0], y[0])
    rf = grid_search.best_estimator_
    print("Optimal paremeters of RF regressor: ", grid_search.best_params_)
    print("Out-of-bag score is: ", rf.oob_score_)

    preds_df = pd.DataFrame(rf.predict(X[1]), columns=["OPENMINDEDNESS_Z", "CONSCIENTIOUSNESS_Z", "EXTRAVERSION_Z", "AGREEABLENESS_Z", "NEGATIVEEMOTIONALITY_Z"])
    preds_df = pd.concat([metadata[SETS.val]["ID"], preds_df], axis=1).rename(columns={"ID": "Participant ID"})
    preds_df.to_csv("../predictions/predictions.csv", index=None)
    print("Predictions on test set are as follows:", preds_df)
    print("Predictions on test set are saved in the following file (predictions/predictions.csv)")