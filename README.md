# ✈️ Logistic Regression – Airline Customer Satisfaction Prediction

## Project Overview

This project builds a **Binomial Logistic Regression** model to predict whether an airline passenger is `satisfied` or `dissatisfied` using the **Invistico Airline Passenger Survey** dataset (129,880 records, 22 columns).

---

## 📂 Repository Structure

```
├── logistic_regression_analysis.ipynb   # Main Jupyter Notebook (all cells executed)
├── README.md                            # This file
├── confusion_matrix.png                 # Confusion matrix heatmap
├── roc_curve.png                        # ROC-AUC curve
├── feature_coefficients.png             # Top 15 coefficient bar chart
├── eda_categorical.png                  # EDA: satisfaction by segment
├── eda_service_ratings.png              # EDA: average service ratings
└── target_distribution.png             # Target class balance plot
```

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **File** | `Invistico_Airline.csv` |
| **Rows** | 129,880 |
| **Columns** | 22 |
| **Target** | `satisfaction` → `satisfied` (1) or `dissatisfied` (0) |
| **Class Balance** | 54.7% satisfied / 45.3% dissatisfied |
| **Missing Values** | 393 in `Arrival Delay in Minutes` → filled with median |

**Feature Columns Used (22 → model uses 22 after encoding):**
- Demographics: `Age`
- Flight info: `Flight Distance`, `Departure Delay in Minutes`, `Arrival Delay in Minutes`
- Passenger segment: `Customer Type`, `Type of Travel`, `Class`
- 14 service ratings (0–5 scale): Seat comfort, Inflight entertainment, Online boarding, etc.

---

## ⚙️ Modeling Approach

### 1. Preprocessing
| Step | Detail |
|---|---|
| Missing values | `Arrival Delay in Minutes` (393 NaNs) → filled with column median |
| Target encoding | `satisfied` → 1, `dissatisfied` → 0 |
| Binary encoding | `Customer Type`, `Type of Travel` → `LabelEncoder` |
| One-hot encoding | `Class` (3 categories) → 2 dummy columns (Business = reference) |
| Feature scaling | `StandardScaler` — fit on train only (prevents data leakage) |

### 2. Train/Test Split
- **80% train (103,904 rows) / 20% test (25,976 rows)**
- `stratify=y` ensures equal class ratios in both splits

### 3. Model
```python
LogisticRegression(max_iter=1000, random_state=42)
```
Converges in **17 iterations**.

---

## 📈 Final Performance Metrics (on Test Set)

| Metric | Score | Interpretation |
|---|---|---|
| **Accuracy** | 82.89% | Overall correct predictions |
| **Precision** | 84.41% | Of predicted satisfied, 84.4% truly were |
| **Recall** | 84.30% | Of all satisfied passengers, 84.3% correctly identified |
| **F1-Score** | 84.36% | Balanced precision-recall score |
| **ROC-AUC** | 0.9035 | Strong discrimination ability |

### Confusion Matrix (Test Set: 25,976 rows)

|  | Predicted Dissatisfied | Predicted Satisfied |
|---|---|---|
| **Actually Dissatisfied** | 9,546 (TN) ✅ | 2,213 (FP) ❌ |
| **Actually Satisfied** | 2,232 (FN) ❌ | 11,985 (TP) ✅ |

**Precision** answers: *"Of passengers we flagged as satisfied, 84.4% were correct — low false alarm rate."*  
**Recall** answers: *"We correctly identified 84.3% of all truly satisfied passengers."*

---

## 🔑 Key Satisfaction Drivers (Top 5 by Coefficient)

| Feature | Coefficient | Direction | Business Meaning |
|---|---|---|---|
| Inflight entertainment | +0.97 | ⬆️ Strong positive | #1 driver — 2.6x satisfaction multiplier |
| Customer Type (disloyal) | −0.73 | ⬇️ Strong negative | Disloyal customers far less satisfied |
| Seat comfort | +0.39 | ⬆️ Positive | Comfort matters after entertainment |
| On-board service | +0.39 | ⬆️ Positive | Crew attentiveness is highly valued |
| Checkin service | +0.36 | ⬆️ Positive | First impression sets tone for trip |

---

## 💼 Business Recommendations

1. **Upgrade Inflight Entertainment** — strongest predictor; invest in screens and content library
2. **Launch a Loyalty Recovery Programme** — disloyal passengers have the lowest satisfaction; target with first-3-flight incentives
3. **Improve Seat Comfort & Crew Service** — both carry strong positive coefficients
4. **Streamline Check-In & Online Booking** — frictionless digital experience lifts satisfaction
5. **Reduce Arrival Delays** — negative predictor; proactive communications with compensation offers

---

## ⚠️ Limitations & Next Steps

- Logistic Regression assumes **linear relationships** — consider **Random Forest** or **XGBoost** for potentially higher accuracy
- Survey data may reflect **response bias** (only engaged passengers complete surveys)
- Retrain model **quarterly** as service standards change
- **Deployment idea:** Score passengers at check-in in real-time; flag predicted-dissatisfied passengers for proactive service recovery

---

## 🛠️ How to Run (Google Colab)

1. Upload `logistic_regression_analysis.ipynb` to Google Colab
2. Upload `Invistico_Airline.csv` via the Files panel (left sidebar)
3. Click **Runtime → Run All**

**Dependencies** (all pre-installed in Colab):
```
pandas  numpy  matplotlib  seaborn  scikit-learn
```
