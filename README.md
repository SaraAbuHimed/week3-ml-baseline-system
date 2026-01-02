# 🚚 Shipping Crisis – ML Baseline System

## **Project Overview**

This project builds a machine learning baseline system to predict whether a shipment will be delayed and quantify the business value (SAR impact) of using the model.

The notebook follows a structured, phased approach:

1. Exploratory Data Analysis (EDA)
2. Feature Engineering
3. Model Training & Evaluation
4. Business Value Calculation

The goal is to support proactive customer retention and operational decision-making.

## **Project Structure**
```bash
week3-ml-baseline-system/
│
├── data/
│
├── notebooks/
│   ├── Shipping_Crisis_Template.ipynb   ← MAIN NOTEBOOK
│
├── models/
├── reports/
├── src/
├── tests/
│
├── .gitignore
├── pyproject.toml
├── uv.lock
├── README.md
```
you only need to run Shipping_Crisis_Template.ipynb.

## **Environment Setup (Required)**

### 1. Install `uv` (if not already installed)
```bash
pip install uv
```
### 2. Create a virtual environment
```bash
uv venv --python 3.11
```

### 3. Activate the virtual environment
```bash
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate    # Windows
```

### 4. Sync Virtual Environment
From the project root:
```bash
uv sync
```
This installs all required dependencies exactly as used in development.

## **Running the Notebook**

Open the project folder in VS Code and open `notebooks/Shipping_Crisis_Template.ipynb`.  
When prompted, select the `.venv` Python environment created using `uv`.

## **Notebook Execution Order (IMPORTANT)**

⚠️ Run cells sequentially from top to bottom. Do not skip steps.

**Phase 1: Exploratory Data Analysis**
- Dataset inspection
- Delay patterns by:
    - Shipment mode
    - Warehouse zone
    - Product weight
    - Discount strategy
- Interactive Plotly visualizations

**Phase 2: Feature Engineering**
- Drop non-informative columns (e.g., Tracking_ID)
- Define categorical and numerical features
- Encode categorical variables
- Scale numerical features
- Train / test split (stratified)

**Phase 3: Training, Evaluation & Strategy**

- Baseline model comparison using PyCaret
- Metric selection focused on Recall
- Model tuning
- Final model evaluation
- Confusion Matrix
- Classification Report

Why Recall?

Missing a delayed shipment (False Negative) has a higher business cost than sending an unnecessary coupon.

**Phase 4: Business Value Calculator**

The model’s predictions are converted into SAR impact using:

| Outcome        | Business Impact |
| -------------- | --------------- |
| True Positive  | +75 SAR         |
| False Positive | −18.75 SAR      |
| False Negative | −187.50 SAR     |

This quantifies real financial value, not just accuracy.

## **Expected Outputs**

- Interactive EDA charts (Plotly)
- Model comparison table
- Tuned classifier
- Confusion Matrix (interactive)
- Classification Report
- Total Business Value in SAR