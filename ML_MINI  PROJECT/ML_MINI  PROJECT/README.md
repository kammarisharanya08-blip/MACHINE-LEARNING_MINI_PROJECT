# Smart Agriculture: Irrigation Prediction Using Machine Learning

**Machine Learning Laboratory – Mini Project**
MNR University, School of Engineering & Technology and AI
B.Tech CSE (Artificial Intelligence and Data Science)
**Guide:** Dr M Praveen (Associate Professor)

| Roll No. | Name |
|---|---|
| 25RBAID0022 | O Sheshanth |
| 25RBAID0003 | K Sharanya |
| 25RBAID0027 | Y Akhil |
| 25RBAID0020 | N Vijayalakshmi |

---

## 1. Overview
Traditional irrigation relies on manual observation or fixed schedules, which can waste water (over-irrigation) or stress crops (under-irrigation). This project builds a binary classifier that predicts **whether irrigation is required** from five agricultural parameters, and compares three classification algorithms.

> **Note:** The dataset is *synthetically generated* for academic demonstration. The results do not claim real-world agricultural performance.

## 2. Dataset
**File:** `smart_agriculture_irrigation_dataset.csv` – 1,000 rows × 6 columns, no missing values.

| Column | Description | Unit | Range generated |
|---|---|---|---|
| `soil_moisture` | Moisture present in the soil | % | 5 – 80 |
| `temperature` | Environmental temperature | °C | 15 – 45 |
| `humidity` | Relative humidity | % | 20 – 95 |
| `rainfall` | Recent rainfall | mm | 0 – 30 |
| `soil_nutrients` | Illustrative soil nutrient index | index | 10 – 100 |
| `irrigation_required` | **Target:** 0 = not required, 1 = required | – | 0 / 1 |

**Class distribution:** 747 × `0` (No irrigation) and 253 × `1` (Irrigation required), so the data is imbalanced (about 75 / 25).

### How the data was generated
Features are drawn from uniform distributions using `np.random.default_rng(42)`. The label comes from a scoring rule plus Gaussian noise:

```python
score = (0.075 * (45 - soil_moisture)
       + 0.045 * (temperature - 25)
       - 0.025 * (humidity - 55)
       - 0.11  * rainfall
       + 0.008 * (50 - soil_nutrients))

irrigation_required = ((score + Normal(0, 0.65)) > 0.15).astype(int)
```
Low moisture and high temperature push the score up; rainfall and humidity push it down. Because the seed is fixed (42), re-running the code reproduces the identical CSV.

## 3. Methodology
1. Generate the dataset (NumPy + Pandas)
2. Inspect it (`head`, `info`, `describe`, missing values)
3. Exploratory data analysis: class distribution, feature histograms, correlation matrix
4. Separate features `X` (5 columns) and target `y`
5. Split 80 % train / 20 % test (`random_state=42`, `stratify=y`) → 800 / 200 samples
6. Train three models
7. Evaluate with Accuracy, Precision, Recall, F1-Score and a confusion matrix
8. Select the best model by **F1-Score**
9. Predict irrigation for a new sensor reading

### Models
| Model | Configuration |
|---|---|
| Decision Tree | `max_depth=6` |
| Random Forest | `n_estimators=150`, `max_depth=8`, `class_weight="balanced"` |
| Logistic Regression | `StandardScaler` + `LogisticRegression(max_iter=1000, class_weight="balanced")` |

## 4. Results (from the executed notebook)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Logistic Regression** | 0.890 | 0.7164 | 0.9412 | **0.8136** |
| Decision Tree | 0.890 | 0.7636 | 0.8235 | 0.7925 |
| Random Forest | 0.895 | 0.8000 | 0.7843 | 0.7921 |

**Best model:** Logistic Regression (highest F1-Score).

**Confusion matrix (Logistic Regression, 200 test samples):**

| | Predicted No | Predicted Yes |
|---|---|---|
| **Actual No** | 130 (TN) | 19 (FP) |
| **Actual Yes** | 3 (FN) | 48 (TP) |

It misses only 3 of 51 fields that need water, at the cost of some unnecessary irrigation alerts (19 false positives).

**Random Forest feature importance:** soil_moisture 0.546 > rainfall 0.203 > humidity 0.111 > temperature 0.089 > soil_nutrients 0.051.

**Sample prediction:** soil moisture 20, temperature 35 °C, humidity 40 %, rainfall 0 mm, nutrients 60 → **Irrigation required** (99.87 % probability).

> Exact metrics can vary slightly across scikit-learn versions, particularly for Random Forest. The best-model choice (Logistic Regression) was the outcome of the executed notebook.

## 5. Technologies
Python · NumPy · Pandas · Matplotlib · scikit-learn · Jupyter Notebook

**Minimum requirements:** Intel i3 or above, 4 GB RAM, 500 MB free disk space.

## 6. How to Run
```bash
pip install numpy pandas matplotlib scikit-learn jupyter
jupyter notebook
```
Open the project notebook and choose **Run All**. It will:
- generate the dataset and save `smart_agriculture_irrigation_dataset.csv`
- train and compare the three models, saving `model_performance_comparison.csv`
- display the EDA plots, confusion matrix, comparison chart, feature importance and the new-reading prediction

To use the provided CSV directly:
```python
import pandas as pd
data = pd.read_csv("smart_agriculture_irrigation_dataset.csv")
X = data.drop(columns="irrigation_required")
y = data["irrigation_required"]
```

## 7. Project Files
| File | Description |
|---|---|
| `smart_agriculture_irrigation_dataset.csv` | Synthetic dataset (1,000 records) |
| `README.md` | This file |
| `smart_agriculture.docx` | Full project report |
| `smart_agriculture_output_code.pdf` | Notebook code and outputs |
| `Smart_Agriculture_Irrigation_Prediction_ML.pptx` | Presentation |

## 8. Limitations and Future Work
**Limitations:** synthetic data; no validation on real farms; crop type, soil type and weather forecasts are not modelled; no automatic control of irrigation equipment.

**Future work:** real sensor / IoT data, automated pumps and valves, weather-forecast integration, a web or mobile app, and training on real multi-season datasets.

## 9. References
- Aurélien Géron, *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, O'Reilly
- Christopher M. Bishop, *Pattern Recognition and Machine Learning*, Springer
- scikit-learn, pandas, NumPy and Matplotlib documentation
