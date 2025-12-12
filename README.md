# Credit Card Fraud Detection (Unsupervised Learning)
# Project Overview
This project explores unsupervised machine learning techniques to detect fraudulent credit card transactions.
Since fraud cases are rare (~8.7%), unsupervised methods like Isolation Forest and Agglomerative Clustering are applied to identify anomalies.

# The project demonstrates:
Data preprocessing and feature scaling
Dimensionality reduction using PCA
Clustering and anomaly detection
Evaluation of unsupervised models against known fraud labels

# Dataset:
Original dataset: 1,000,000 transactions with 7 features + fraud label total 8 columns.
Sampled dataset included in this repository: 100,000 rows (card_transdata_sample.csv)
Reason for sampling: The full dataset exceeds GitHub’s 25MB limit.

# Columns:
distance_from_home – distance from the user’s home
distance_from_last_transaction – distance from the previous transaction
ratio_to_median_purchase_price – transaction price ratio
repeat_retailer – whether retailer is repeated
used_chip – whether chip was used
used_pin_number – whether PIN was used
online_order – whether it was an online order
fraud – 0 = non-fraud, 1 = fraud (used for evaluation only)

# Note: The fraud column is ignored during training in this unsupervised project.
Getting Started
Prerequisites
Python 3.9+

# Required libraries:
pip install pandas numpy matplotlib seaborn scikit-learn
Usage

# Clone the repository:
git clone <your-repo-url>
cd <repo-folder>

# Open the notebook:
code CreditCard_Fraud_Detection.ipynb

# Run all cells to:
Load and preprocess the dataset
Apply PCA for dimensionality reduction
Fit Isolation Forest and Agglomerative Clustering models
Visualize and evaluate results

# Project Structure:
├── card_transdata_sample.csv  # Sampled dataset for GitHub
├── CreditCard_Fraud_Detection.ipynb  # Main notebook
├── README.md                  # Project description and instructions

# Results:
K-Means Clustering for a grouping a data
Isolation Forest detected most anomalous/fraudulent transactions.
DBSCAN Clustering shows clusters of normal and suspicious transactions.
Visualization using PCA helps identify outliers.

# Notes:
To use the full dataset, replace card_transdata_sample.csv with your full CSV locally.
Outputs may vary slightly due to random sampling and model initialization.
