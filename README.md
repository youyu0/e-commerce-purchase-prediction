# e-commerce puchase prediction

This repository contains an **e-commerce purchase prediction project** built with Python and machine learning models.  
It focuses on predicting whether a product added to a shopping cart will eventually be purchased, based on user behavior and session-level features.

## Overview

This project uses e-commerce event data to build a binary classification task.  
The workflow includes:

- feature engineering from user behavior logs
- building a training dataset from cart and purchase events
- splitting data by `user_session`
- handling class imbalance
- training and tuning machine learning models
- comparing model performance using evaluation metrics

The project currently includes:

- XGBoost training and threshold tuning
- LightGBM / Random Forest / XGBoost model comparison
- session-based feature engineering
- automatic model evaluation and result export

## Features

- Build training data from raw e-commerce event logs
- Predict whether carted items will be purchased
- Create behavior-based features from user sessions
- Use `GroupShuffleSplit` and `GroupKFold` based on `user_session`
- Tune XGBoost with `RandomizedSearchCV`
- Compare XGBoost, LightGBM, and Random Forest
- Evaluate models with PR-AUC, ROC-AUC, F1, Precision, and Recall
- Export predictions and model comparison results to CSV files

## Tech Stack

- Python
- pandas
- numpy
- scikit-learn
- XGBoost
- LightGBM

## Project Structure

```bash
e-comeerce/
├── README.md
├── predict_v2_0324.py          # Main training / preprocessing / evaluation pipeline
└── compare_models_0325.py      # Compare XGBoost, LightGBM, and Random Forest
```

## How It Works

1. Read the raw dataset from `2019-Oct.csv`
2. Clean and preprocess event data
3. Create behavior-based features from user sessions
4. Build a training dataset from cart and purchase events
5. Split the dataset into train / validation / test sets by `user_session`
6. Train and tune the XGBoost model
7. Find the best decision threshold
8. Evaluate the model on the test set
9. Compare multiple models and save the results

## Main Input Data

The default input file is:

```bash
2019-Oct.csv
```

The scripts expect event-level e-commerce data containing fields such as:

- `event_time`
- `event_type`
- `product_id`
- `price`
- `user_id`
- `user_session`
- `brand`
- `category_code`

## Engineered Features

The project creates several behavior-based features, including:

- prior session view count
- prior session cart count
- prior session remove count
- prior session event count
- prior same product view count
- prior same product cart count
- prior same product remove count
- seconds from session start
- event weekday
- category level features

## Installation

Clone the repository:

```bash
git clone https://github.com/youyu0/e-comeerce.git
cd e-comeerce
```

Install dependencies:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm
```

## Usage

Run the main training pipeline:

```bash
python predict_v2_0324.py
```

Run model comparison:

```bash
python compare_models_0325.py
```

## Output

The scripts generate output files in the `output/` directory, such as:

- `training_data.csv`
- `test_predictions.csv`
- `model_comparison_results.csv`

## Evaluation Metrics

The project evaluates models using:

- PR-AUC
- ROC-AUC
- Accuracy
- Precision
- Recall
- F1 Score

## Notes

- The dataset is split by `user_session` to reduce leakage between training and testing
- XGBoost tuning is based on **PR-AUC**
- Threshold selection is currently based on **F1 score**
- The project is designed for experimentation and model comparison on e-commerce behavior data

## Future Improvements

- Add a requirements file
- Add sample dataset format documentation
- Save trained models for reuse
- Add feature importance visualization
- Add notebook-based analysis and charts
- Improve README with dataset source and example results

## Author

youyu0

## License

This repository currently does not specify a license.
