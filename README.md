# Machine Learning Project — Team 21

A university coursework project for the **Pattern Recognition & Machine Learning (2025–2026)** course at the Aristotle University of Thessaloniki (AUTH). The project spans fundamental ML algorithms built from scratch through advanced ensemble classification techniques.

## Project Structure

| File | Description |
|------|-------------|
| `Team21-AC.ipynb` | Parts A, B & C — foundational ML algorithms |
| `Team21-D.ipynb` | Part D — advanced multi-class classification pipeline |
| `PR_Assignment_2025_2026.pdf` | Assignment specification |

## Parts A–C: Foundational Algorithms

Implementations built from scratch (NumPy only, no scikit-learn):

- **Part A — Maximum Likelihood Estimation (MLE):** Gaussian parameter estimation (mean, covariance) for three classes and 3-D probability density visualization.
- **Part B — Parzen Window Density Estimation:** Gaussian and hypercube kernel density estimators with bandwidth optimization via MSE.
- **Part C — k-Nearest Neighbours (KNN):** Custom KNN classifier with Euclidean distance, optimal *k* selection, and decision-boundary visualization.

## Part D: Advanced Classification Pipeline

A full ML pipeline for a 5-class classification problem (8 743 training samples, 224 features):

1. **Preprocessing** — RobustScaler standardization, PCA (99 % variance retention), feature selection (SelectKBest with mutual information).
2. **Models evaluated** — Random Forest, SVM, XGBoost, LightGBM, KNN, Logistic Regression.
3. **Ensemble** — Stacking classifier combining XGBoost, KNN, Random Forest and SVM with a Logistic Regression meta-learner.
4. **Tuning** — GridSearchCV with Stratified K-Fold cross-validation.

## Requirements

- Python 3
- NumPy, pandas, matplotlib, seaborn
- scikit-learn
- XGBoost, LightGBM

## Usage

Open the notebooks in Jupyter and run the cells sequentially. The datasets (`dataset1.csv`, `dataset2.csv`, `dataset3.csv`, `testset.csv`, `datasetTV.csv`, `datasetTest.csv`) should be placed in the working directory.
