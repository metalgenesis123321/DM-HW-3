# Data Mining HW3 — K-Means & Recommender Systems

## Installation
Install required dependencies:

pip install numpy pandas scikit-learn matplotlib
# Task 1 — K-Means (from scratch)
Run:
python kmeans_from_scratch.py --data data.csv --label label.csv --out results_kmeans --max_iter 200

Outputs will be saved in the results_kmeans/ folder.

# Task 2 — Recommender Systems
Run 5-fold cross-validation:
python recommender_experiments.py --ratings ratings_small.csv --sample 30000 --out results_reco_sample --folds 5

Outputs will be saved in the results_reco_sample/ folder.
