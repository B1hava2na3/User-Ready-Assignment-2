# Load Type Classification (Steel Industry Energy Data)

Predicts the `Load_Type` (Light, Medium or Maximum load) of an industrial facility from 15-minute energy readings recorded throughout 2018.

## Problem Statement

Build a classification model that predicts `Load_Type` from the electrical readings and timestamp. The last month of the data (December 2018) is held out as the test set, and everything before it is used for training and validation.

## Dataset

`data/load_data.csv`: 35,041 readings at 15-minute intervals (1 Jan to 31 Dec 2018).

| Column | Description |
|---|---|
| Date_Time | Timestamp of the reading |
| Usage_kWh | Energy consumption |
| Lagging / Leading Reactive Power (kVarh) | Reactive power readings |
| CO2 (tCO2) | CO2 emissions |
| Lagging / Leading Power Factor | Power factor readings |
| NSM | Number of seconds from midnight |
| Load_Type | Target: Light_Load, Medium_Load, Maximum_Load |

## Project Structure

```
├── data/
│   └── load_data.csv
├── load_type_classification.ipynb
├── requirements.txt
└── README.md
```

## Approach

1. **Data cleaning:** converted dates, removed one duplicate timestamp, recomputed NSM from the timestamp (the original had impossible values), treated power factors above 100 as invalid, capped the top 0.5% of extreme readings, and filled missing values by interpolation. All limits and fallback values were learned from the training data only.
2. **EDA:** class balance, outliers, usage by hour, load type by hour and season, correlations.
3. **Feature engineering:** hour, day of week, weekend flag, cyclic time features, total reactive power, lagging/leading ratio, past-usage features (previous 15 minutes, previous hour, rolling mean), and a winter-season flag.
4. **Validation:** December 2018 is the test set. Models were compared with time-ordered cross-validation (`TimeSeriesSplit`) on the training months, so no future data is used to predict the past.
5. **Model selection:** Logistic Regression, Decision Tree, Random Forest, XGBoost and LightGBM were compared. LightGBM was tuned with `RandomizedSearchCV`.
6. **Evaluation:** accuracy, precision, recall, F1 and a confusion matrix on the December test set.

## Results

**Model comparison (cross-validation on training data):**

| Model | Accuracy | Macro F1 |
|---|---|---|
| LightGBM | 0.859 | 0.824 |
| XGBoost | 0.858 | 0.823 |
| Random Forest | 0.857 | 0.822 |
| Decision Tree | 0.852 | 0.816 |
| Logistic Regression | 0.811 | 0.772 |

**Final model on the December 2018 test set:**

| | Accuracy | Macro F1 |
|---|---|---|
| Baseline (without season feature) | 0.760 | 0.656 |
| Final LightGBM (with season feature) | **0.967** | **0.964** |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Light_Load | 0.954 | 0.993 | 0.973 |
| Medium_Load | 0.989 | 0.903 | 0.944 |
| Maximum_Load | 0.985 | 0.964 | 0.974 |

## Key Finding

`Load_Type` follows a time-of-day schedule that changes with the season. In January and February, the hours labelled Medium and Maximum are swapped compared with March to November, and December follows the winter pattern. The baseline model did not know the season and applied the March to November schedule to December, which confused Medium and Maximum. Adding a winter flag (December, January, February) raised accuracy from 0.76 to 0.97.

This problem was found while analysing the December errors, so the baseline result is kept in the notebook for transparency.

## Limitations

- The winter schedule is learned from only two months (Jan and Feb) of a single year.
- The season flag is a calendar rule, not something the model learns automatically.
- Medium_Load has the lowest recall (0.90).
- More years of data would be needed to validate the season effect properly.

## How to Run

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt
jupyter notebook load_type_classification.ipynb
```

Then run all cells from top to bottom. (In VS Code, open the notebook and choose **Run All**.)

## Requirements

Python 3.10+ and the packages listed in `requirements.txt`.