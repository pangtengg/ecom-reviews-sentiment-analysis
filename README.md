# E-Commerce Reviews Sentiment Analysis

## Team Members

1. Kueh Pang Teng
2. Anis Syifaa' binti Mohd Zaffarin
3. Nur Insyirah Iman binti Mohd Azman
4. Sofia Batrisyia binti Mohamad Faris

## Overview

This project is a sentiment analysis dashboard for Malaysian e-commerce product reviews from platforms such as Shopee and Lazada. The system classifies customer reviews into **Positive**, **Neutral**, and **Negative** sentiments, detects common review aspects, and presents the results through an interactive Streamlit dashboard.

The dashboard is designed to help sellers quickly understand customer feedback, inspect negative reviews, view sentiment trends, and download analyzed review results.

## Objectives

1. To develop a sentiment analysis system that classifies Malaysian e-commerce product reviews from Shopee and Lazada into positive, neutral, and negative sentiments using machine learning and transformer-based approaches.
2.	To preprocess and represent multilingual Malaysian e-commerce review text, in-cluding English, Malay and Manglish-style reviews, and evaluate the performance of models using accuracy, precision, recall, macro F1-score, and weighted F1-score.
3.	To develop a Streamlit dashboard that helps Malaysian e-commerce sellers visu-alize customer sentiment, identify common review aspects, inspect negative feedback, and download analyzed review results for decision-making.

## Main Features

- Upload review data in CSV or Excel format.
- Automatically detect or allow selection of the review text column.
- Predict sentiment using a fine-tuned DistilBERT model.
- Display KPI cards for total reviews and sentiment percentages.
- Visualize sentiment distribution and sentiment trends.
- Detect review aspects such as product quality, delivery, packaging, seller service, and price/value.
- Show aspect analysis and negative aspect ranking.
- Generate word clouds for all reviews and negative reviews.
- View analyzed review results in a table.
- Optionally generate a LIME explanation for one selected prediction.
- Download analyzed reviews as a CSV file.

## System Requirements

Recommended environment:

- Python 3.11
- Windows, macOS, or Linux
- At least 4 GB RAM, 8 GB recommended
- Internet connection for first-time package installation

The app runs on CPU by default. Large datasets may take longer to analyze because DistilBERT is a transformer-based model.

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd ecom-reviews-sentiment-analysis
```

2. Create a virtual environment:

```bash
python -m venv .venv
```

3. Activate the virtual environment.

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

## Required Model Files

The Streamlit app loads the trained DistilBERT model from:

```text
model_training/results/distilbert
```

This folder should contain:

```text
config.json
label_mapping.json
model.safetensors
tokenizer.json
tokenizer_config.json
vocab.txt
```

If the trained model folder is missing, the app will fall back to a simple rule-based sentiment method, which is less accurate than the trained DistilBERT model.

## Running the Application

Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## How To Use

1. Open the Streamlit dashboard.
2. Upload a CSV or Excel file containing customer reviews.
3. If the app detects multiple possible text columns, select the column that contains the review content.
4. Wait for the sentiment analysis process to complete. A progress bar will be shown during analysis.
5. Review the dashboard outputs:
   - KPI cards
   - Sentiment distribution
   - Sentiment trend
   - Aspect analysis
   - Negative aspect ranking
   - Word clouds
   - Analyzed review table
6. Optionally open the explanation section and generate a LIME explanation for one selected review.
7. Download the analyzed review results as a CSV file.

## Input File Format

The uploaded file can be `.csv`, `.xlsx`, or `.xls`.

The file should contain at least one text column with customer reviews. The column does not need to be named exactly `review_text`. The app can detect common column names such as:

- `review_text`
- `review`
- `review_content`
- `comment`
- `feedback`
- `text`

Optional columns:

| Column | Purpose |
|---|---|
| `review_date` | Used to generate sentiment trend over time |
| `star_rating` | Displayed in the analyzed review table |
| `review_text_translated` | Preferred input for DistilBERT if available |

## Model Summary

The project compares traditional machine learning and transformer-based approaches:

- Logistic Regression
- Naive Bayes
- Support Vector Machine
- DistilBERT

Traditional machine learning models use TF-IDF features generated from normalized and tokenized review text. DistilBERT uses translated review text because transformer models are designed to learn contextual meaning from natural language sentences.

DistilBERT was selected for dashboard deployment because it achieved the best overall performance across the main evaluation metrics.

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Weighted F1-score
- Class-wise F1-score
- Confusion matrix

## Deployment Notes

For Streamlit Cloud deployment, this repository includes:

```text
requirements.txt
runtime.txt
.streamlit/config.toml
```


