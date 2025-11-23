
# Multi-Class-Classification – Election Winner Education Level Prediction

## Task

You are provided with a training dataset and a test dataset. The goal is to train a machine-learning model (SVM, KNN, Decision Tree, Random Forest, Naive Bayes, or similar) on this dataset and perform **multi-class classification** to predict the education level of election winners. Only standard Python ML libraries are used (pandas, NumPy, scikit-learn, etc.). No deep learning libraries such as PyTorch, TensorFlow, Keras, or ANN-based methods are used.

---

## Repository Structure

```text
Data/
    raw/
        train.csv
        test.csv
    Processed/
        my_df.csv
        new_final_df.csv

notebooks/
    EDA.ipynb

Output/
    final_submission_1.csv
    final_submission_2.csv
    final_submission_3.csv

src/
    main.py
    models/
        dt.py
        knn.py
        nbc.py
        rf.py
        svc.py
    preprocessing/
        pre_processing.py
        process_augment.py
        smote.py
        undersample.py
    utils/
        utils.py

requirements.txt
README.md
```

---

## Contents

### Code

* [`src/main.py`](src/main.py)
  Driver script. Orchestrates preprocessing, model training, prediction, and writing submissions.

* [`src/preprocessing/pre_processing.py`](src/preprocessing/pre_processing.py)
  Functions for basic preprocessing of the training and test datasets (cleaning, encoding, handling missing values, etc.).

* [`src/preprocessing/process_augment.py`](src/preprocessing/process_augment.py)
  Script for generating additional synthetic training data (using CTGAN) for data augmentation.

* [`src/preprocessing/smote.py`](src/preprocessing/smote.py)
  Functions for applying SMOTE-based oversampling to address class imbalance.

* [`src/preprocessing/undersample.py`](src/preprocessing/undersample.py)
  Functions for undersampling majority classes to achieve better balance across target labels.

* Model implementations (each using grid search / tuning for hyperparameters):

  * [`src/models/dt.py`](src/models/dt.py) – Decision Tree
  * [`src/models/knn.py`](src/models/knn.py) – K-Nearest Neighbours
  * [`src/models/nbc.py`](src/models/nbc.py) – Naive Bayes classifier
  * [`src/models/rf.py`](src/models/rf.py) – Random Forest
  * [`src/models/svc.py`](src/models/svc.py) – Support Vector Classifier

* [`src/utils/utils.py`](src/utils/utils.py)
  Utility helpers (common functions used by multiple scripts).

* [`notebooks/EDA.ipynb`](notebooks/EDA.ipynb)
  Jupyter notebook containing exploratory data analysis and visualisations required by the assignment (distribution plots, party-wise analysis, etc.).

### Data

* [`Data/raw/train.csv`](Data/raw/train.csv)
  Original training dataset provided in the assignment.

* [`Data/raw/test.csv`](Data/raw/test.csv)
  Original test dataset on which predictions are generated.

* [`Data/Processed/my_df.csv`](Data/Processed/my_df.csv)
  Preprocessed version of the training data after cleaning and feature processing.

* [`Data/Processed/new_final_df.csv`](Data/Processed/new_final_df.csv)
  Augmented dataset combining original training data with CTGAN-generated synthetic samples.

### Submissions

* [`Output/final_submission_1.csv`](Output/final_submission_1.csv)
  Best submission generated using the SVC model.
  Public score: **0.22828**, private score: **0.25139**.

* [`Output/final_submission_2.csv`](Output/final_submission_2.csv)
  Second-best submission generated using the SVC model.
  Public score: **0.23874**, private score: **0.24618**.

* [`Output/final_submission_2.csv`](Output/final_submission_3.csv)
  Third-best submission generated using the Decision Tree model.

---

## Instructions

1. Install all required libraries:

   ```bash
   pip install -r requirements.txt
   ```

2. (Optional) Regenerate processed/augmented data:

   ```bash
   python src/preprocessing/process_augment.py
   ```

3. Ensure the test file you want to evaluate on is available (for example `Data/raw/test.csv`).

4. Run the main script:

   ```bash
   python src/main.py <model_name> <path_to_test_file>
   ```

   Examples:

   ```bash
   python src/main.py svc Data/raw/test.csv
   python src/main.py rf Data/raw/test.csv
   ```

   * If `<path_to_test_file>` is omitted, the script defaults to `Data/raw/test.csv` (subject to how `main.py` is implemented).
   * If `<model_name>` is omitted, the SVC model is used by default.
   * `<model_name>` must be one of: `dt`, `knn`, `nbc`, `rf`, `svc` (without the `.py` extension).
