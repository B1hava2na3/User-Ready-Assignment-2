# Load Type Classification (Steel Industry Energy Data)

Predicts the `Load_Type` (Light, Medium or Maximum load) of an industrial facility from 15-minute energy readings recorded throughout 2018.

## Problem Statement

Build a classification model that predicts `Load_Type` from the electrical readings and the timestamp. December 2018 is held out as the test set. Everything before it is used for training and validation.

## Dataset

`data/load_data.csv`: 35,041 raw readings at 15-minute intervals (1 Jan to 31 Dec 2018). After removing one duplicate timestamp, 35,040 rows remain.

| Column | Description |
|---|---|
| Date_Time | Timestamp of the reading |
| Usage_kWh | Energy consumption |
| Lagging / Leading Current Reactive Power (kVarh) | Reactive power readings |
| CO2 (tCO2) | CO2 emissions |
| Lagging / Leading Current Power Factor | Power factor readings |
| NSM | Number of seconds from midnight |
| Load_Type | Target: Light_Load, Medium_Load, Maximum_Load |

Class balance: Light 51%, Medium 28%, Maximum 21% in training. Light 59%, Medium 24%, Maximum 18% in December.

## Project Structure

```
├── data/
│   └── load_data.csv
├── load_type_classification.ipynb
├── requirements.txt
└── README.md
```

## Approach

1. **Data cleaning.**
   - Converted dates and removed one duplicate timestamp.
   - Recomputed NSM from the timestamp, because the original column had impossible values (1,623 training rows above 86,400, and 3,127 that did not match the timestamp).
   - Treated power factor values above 100 as invalid and set them to missing.
   - Capped the top 0.5% of Usage_kWh, both reactive power columns and CO2.
   - Filled missing values by forward linear interpolation, with training medians as fallback.
   - All cap limits and fallback values were learned from the training data only.
2. **EDA.** Class balance, outliers, usage by hour, load type by hour and by season, correlations.
3. **Feature engineering.** Hour, day of week, weekend flag, cyclic time features (sin/cos of NSM), total reactive power, lagging/leading ratio, and past-usage features (previous 15 minutes, previous hour, mean of the previous hour).
4. **Validation.** December 2018 is the test set. Models were compared with time-ordered cross-validation (`TimeSeriesSplit`, 5 splits) on the training months, so no future data is used to predict the past.
5. **Model selection.** Logistic Regression, Decision Tree, Random Forest, XGBoost and LightGBM were compared. LightGBM was then tuned with `RandomizedSearchCV` (15 iterations, scored on macro F1).
6. **Season feature.** After analysing the December errors of the first tuned model, a winter flag (December, January, February) was added and the model was re-tuned. See *Key Finding* below.
7. **Evaluation.** Accuracy, precision, recall, F1 and a confusion matrix on the December test set.

## Results

**Model comparison** (time-series cross-validation on the training months, before the season feature was added):

| Model | Accuracy | Macro F1 |
|---|---|---|
| LightGBM | 0.859 | 0.824 |
| XGBoost | 0.858 | 0.823 |
| Random Forest | 0.857 | 0.822 |
| Decision Tree | 0.852 | 0.816 |
| Logistic Regression | 0.811 | 0.772 |

Fold-to-fold variation is large (macro F1 standard deviation of about 0.12; for LightGBM the five folds range from 0.61 to 0.94). The earliest folds contain only January, so they see just the winter schedule. The model ranking is therefore more reliable than the absolute scores.

**Final model on the December 2018 test set:**

| | Accuracy | Macro F1 |
|---|---|---|
| Baseline (without season feature) | 0.760 | 0.656 |
| Final LightGBM (with season feature) | **0.967** | **0.964** |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Light_Load | 0.954 | 0.993 | 0.973 | 1,744 |
| Medium_Load | 0.989 | 0.903 | 0.944 | 704 |
| Maximum_Load | 0.985 | 0.964 | 0.974 | 528 |

## Key Finding

`Load_Type` follows a time-of-day schedule that changes with the season. In January and February the Maximum/Medium pattern differs from March to November:

- **Mar to Nov:** hours 13 to 16 are mostly Maximum, and hours 17 to 19 are mostly Medium.
- **Jan and Feb:** hours 13 to 16 are mostly Medium, and hours 17 to 19 are mostly Maximum.

December appears to follow the winter pattern. The baseline model did not know the season, so it applied the March to November schedule to December and confused Medium and Maximum. Adding a winter flag (December, January, February) raised test accuracy from 0.76 to 0.97.

**Important caveat:** this was discovered by inspecting the December errors, so the test set influenced a modelling decision. The 0.967 score should be read as optimistic, not as an unbiased estimate of performance on unseen data. The baseline result is kept for transparency.

## Limitations

- The winter schedule is learned from only two months (Jan and Feb) of a single year.
- December never appears in training. Treating it as winter is a hand-made assumption based on the test-set error analysis, so the final test score is optimistic.
- The season flag is a calendar rule, not something the model learns on its own.
- Medium_Load has the lowest recall (0.90).
- Cross-validation scores vary widely between folds, so absolute CV numbers are only a rough guide.
- More years of data would be needed to validate the season effect properly.

## How to Run

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt
jupyter notebook load_type_classification.ipynb
```

Then run all cells from top to bottom (in VS Code, open the notebook and choose **Run All**).

Note: the notebook sets `LOKY_MAX_CPU_COUNT` to `4` to avoid a joblib warning on Windows. Change it to your number of CPU cores if needed.

## Requirements

Python 3.10+ and the packages listed in `requirements.txt` (pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, lightgbm).
