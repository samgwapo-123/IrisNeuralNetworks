# Data Description

**Dataset:** Iris flower dataset (R. A. Fisher, 1936; UCI Machine Learning Repository, open data).
**File:** `Data/iris.csv` — 150 rows, 4 numeric features, 1 label column, no missing values.

| Column | Type | Meaning |
|---|---|---|
| sepal_length | feature (cm) | Length of the sepal |
| sepal_width | feature (cm) | Width of the sepal |
| petal_length | feature (cm) | Length of the petal |
| petal_width | feature (cm) | Width of the petal |
| species | label | `setosa`, `versicolor`, or `virginica` |

- **Classes:** 3, perfectly balanced (50 samples each).
- **Task:** multi-class classification — predict the species from the 4 measurements.
- **Why this dataset:** small and clean, so training is fast and every part of the pipeline
  (scaling, splitting, tuning, evaluation) can be inspected; one class is linearly separable and two overlap slightly,
  which makes the confusion matrix and loss curve meaningful to discuss.
- **Preprocessing:** Min-Max scaling to [0,1] (fitted on training data only), one-hot encoded labels,
  stratified 70/15/15 train/validation/test split.
