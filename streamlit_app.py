import re
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

try:
    from lime.lime_text import LimeTextExplainer

    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False


ASPECT_KEYWORDS = {
    "Product Quality": [
        "quality",
        "defect",
        "defective",
        "broken",
        "poor",
        "damage",
        "damaged",
        "not working",
        "stopped working",
        "rosak",
        "scratch",
        "fake",
        "original",
    ],
    "Delivery": [
        "delivery",
        "deliver",
        "shipping",
        "ship",
        "courier",
        "late",
        "slow",
        "arrived late",
        "delay",
        "delayed",
        "received",
        "sampai",
    ],
    "Packaging": [
        "packaging",
        "package",
        "parcel",
        "box",
        "bubble wrap",
        "wrap",
        "dented",
        "kemek",
        "crushed",
        "leak",
        "leaking",
    ],
    "Seller Service": [
        "seller",
        "service",
        "reply",
        "response",
        "refund",
        "exchange",
        "chat",
        "support",
        "rude",
        "helpful",
    ],
    "Price / Value": [
        "price",
        "value",
        "worth",
        "cheap",
        "expensive",
        "voucher",
        "discount",
        "money",
        "rm",
        "berbaloi",
        "mahal",
    ],
}


NEGATIVE_WORDS = [
    "bad",
    "poor",
    "late",
    "delay",
    "delayed",
    "damaged",
    "damage",
    "broken",
    "defect",
    "defective",
    "dented",
    "slow",
    "not working",
    "stopped working",
    "wrong",
    "missing",
    "rude",
    "leak",
    "leaking",
    "mahal",
    "rosak",
    "kemek",
]


POSITIVE_WORDS = [
    "good",
    "great",
    "excellent",
    "fast",
    "worth",
    "nice",
    "love",
    "recommended",
    "recommend",
    "quality",
    "works well",
    "berbaloi",
    "cantik",
    "mantap",
]


MODEL_DIR = Path(__file__).parent / "model_training" / "results" / "distilbert"
SENTIMENT_LABELS = {
    "negative": "Negative",
    "neutral": "Neutral",
    "positive": "Positive",
}
LIME_CLASS_NAMES = ["Negative", "Neutral", "Positive"]


def load_demo_data():
    base_date = datetime.today().date()
    demo_reviews = [
        {
            "review_text": "Fast delivery and product quality is good.",
            "review_date": base_date - timedelta(days=13),
            "star_rating": 5,
        },
        {
            "review_text": "Packaging was damaged when received.",
            "review_date": base_date - timedelta(days=11),
            "star_rating": 2,
        },
        {
            "review_text": "Item arrived late and box was dented.",
            "review_date": base_date - timedelta(days=9),
            "star_rating": 2,
        },
        {
            "review_text": "Worth the price and works well.",
            "review_date": base_date - timedelta(days=6),
            "star_rating": 5,
        },
        {
            "review_text": "Product quality is poor and stopped working after one week.",
            "review_date": base_date - timedelta(days=5),
            "star_rating": 1,
        },
        {
            "review_text": "Seller replied quickly and solved my exchange request.",
            "review_date": base_date - timedelta(days=4),
            "star_rating": 4,
        },
        {
            "review_text": "Courier was slow but item quality is okay.",
            "review_date": base_date - timedelta(days=2),
            "star_rating": 3,
        },
    ]
    return pd.DataFrame(demo_reviews)


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource(show_spinner=False)
def load_distilbert_model():
    if not MODEL_DIR.exists():
        return None, None, None

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to("cpu")
    model.eval()

    label_path = MODEL_DIR / "label_mapping.json"
    if label_path.exists():
        with open(label_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        id2label = {int(k): v for k, v in mapping["id2label"].items()}
    else:
        id2label = model.config.id2label

    return tokenizer, model, id2label


def predict_sentiment(text):
    tokenizer, model, id2label = load_distilbert_model()
    if tokenizer is not None and model is not None:
        probabilities = predict_sentiment_probabilities([str(text)])[0]
        pred_id = int(np.argmax(probabilities))
        label = id2label[pred_id]
        return SENTIMENT_LABELS.get(str(label).lower(), str(label).title())

    cleaned = clean_text(text)
    if any(word in cleaned for word in NEGATIVE_WORDS):
        return "Negative"
    if any(word in cleaned for word in POSITIVE_WORDS):
        return "Positive"
    return "Neutral"


def predict_sentiment_probabilities(texts):
    tokenizer, model, _ = load_distilbert_model()
    if tokenizer is None or model is None:
        return np.array([rule_based_probabilities(text) for text in texts])

    inputs = tokenizer(
        [str(text) for text in texts],
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=160,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.softmax(logits, dim=1).cpu().numpy()


def rule_based_probabilities(text):
    prediction = predict_rule_based_sentiment(text)
    probabilities = {"Negative": 0.05, "Neutral": 0.05, "Positive": 0.05}
    probabilities[prediction] = 0.90
    return [probabilities[label] for label in LIME_CLASS_NAMES]


def predict_rule_based_sentiment(text):
    cleaned = clean_text(text)
    if any(word in cleaned for word in NEGATIVE_WORDS):
        return "Negative"
    if any(word in cleaned for word in POSITIVE_WORDS):
        return "Positive"
    return "Neutral"


@st.cache_resource(show_spinner=False)
def load_lime_explainer():
    if not LIME_AVAILABLE:
        return None
    return LimeTextExplainer(class_names=LIME_CLASS_NAMES)


def detect_aspect(text):
    cleaned = clean_text(text)
    for aspect, keywords in ASPECT_KEYWORDS.items():
        if any(keyword in cleaned for keyword in keywords):
            return aspect
    return "General"


def generate_recommendations(df):
    negative_df = df[df["predicted_sentiment"] == "Negative"]
    if negative_df.empty:
        return ["No high negative aspect detected yet. Add a trained model to unlock stronger complaint recommendations."]

    aspect_counts = negative_df["detected_aspect"].value_counts()
    total_negative = len(negative_df)
    recommendations = []

    checks = {
        "Packaging": "Improve packaging materials, add bubble wrap, use stronger boxes, and add fragile labels.",
        "Delivery": "Review courier performance and improve estimated delivery time.",
        "Product Quality": "Check supplier quality control and update product descriptions clearly.",
        "Seller Service": "Improve response time and clarify refund/exchange policy.",
        "Price / Value": "Review pricing strategy and offer bundles or vouchers.",
    }

    for aspect, message in checks.items():
        share = aspect_counts.get(aspect, 0) / total_negative
        if share >= 0.2:
            recommendations.append(message)

    if not recommendations:
        recommendations.append("Negative reviews are spread across aspects. Review the table below to inspect individual complaints.")

    return recommendations


def plot_sentiment_distribution(df):
    sentiment_order = ["Positive", "Neutral", "Negative"]
    counts = (
        df["predicted_sentiment"]
        .value_counts()
        .reindex(sentiment_order, fill_value=0)
        .reset_index()
    )
    counts.columns = ["Sentiment", "Reviews"]
    fig = px.bar(
        counts,
        x="Sentiment",
        y="Reviews",
        color="Sentiment",
        color_discrete_map={
            "Positive": "#16a34a",
            "Neutral": "#ca8a04",
            "Negative": "#dc2626",
        },
        text="Reviews",
    )
    fig.update_layout(showlegend=False, height=330, margin=dict(l=10, r=10, t=20, b=10))
    return fig


def plot_sentiment_trend(df):
    trend = df.copy()
    trend["review_date"] = pd.to_datetime(trend["review_date"], errors="coerce")
    trend = trend.dropna(subset=["review_date"])
    if trend.empty:
        return None

    trend["date"] = trend["review_date"].dt.date
    grouped = trend.groupby(["date", "predicted_sentiment"]).size().reset_index(name="Reviews")
    fig = px.line(
        grouped,
        x="date",
        y="Reviews",
        color="predicted_sentiment",
        markers=True,
        color_discrete_map={
            "Positive": "#16a34a",
            "Neutral": "#ca8a04",
            "Negative": "#dc2626",
        },
    )
    fig.update_layout(
        height=330,
        margin=dict(l=10, r=10, t=20, b=10),
        legend_title_text="Sentiment",
        xaxis_title="Review date",
    )
    return fig


def plot_aspect_analysis(df):
    aspect_counts = df["detected_aspect"].value_counts().reset_index()
    aspect_counts.columns = ["Aspect", "Reviews"]
    frequency_fig = px.bar(
        aspect_counts,
        x="Reviews",
        y="Aspect",
        orientation="h",
        color="Aspect",
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    frequency_fig.update_layout(showlegend=False, height=350, margin=dict(l=10, r=10, t=20, b=10))

    aspect_sentiment = pd.crosstab(df["detected_aspect"], df["predicted_sentiment"]).reset_index()
    for sentiment in ["Positive", "Neutral", "Negative"]:
        if sentiment not in aspect_sentiment.columns:
            aspect_sentiment[sentiment] = 0
    aspect_sentiment = aspect_sentiment[["detected_aspect", "Positive", "Neutral", "Negative"]]
    aspect_sentiment = aspect_sentiment.rename(columns={"detected_aspect": "Aspect"})

    negative_ranking = (
        df[df["predicted_sentiment"] == "Negative"]["detected_aspect"]
        .value_counts()
        .reset_index()
    )
    negative_ranking.columns = ["Aspect", "Negative Reviews"]

    return frequency_fig, aspect_sentiment, negative_ranking


def read_uploaded_file(uploaded_file):
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def add_predictions(df):
    analyzed = df.copy()
    analyzed["review_text"] = analyzed["review_text"].fillna("").astype(str)
    sentiment_text_col = "review_text_translated" if "review_text_translated" in analyzed.columns else "review_text"
    analyzed[sentiment_text_col] = analyzed[sentiment_text_col].fillna("").astype(str)
    analyzed["clean_review_text"] = analyzed["review_text"].apply(clean_text)
    analyzed["model_input_text"] = analyzed[sentiment_text_col]
    analyzed["predicted_sentiment"] = analyzed["model_input_text"].apply(predict_sentiment)
    analyzed["detected_aspect"] = analyzed["review_text"].apply(detect_aspect)
    return analyzed


def render_lime_explanation(df):
    st.subheader("Explain One Sentiment Prediction")
    if not LIME_AVAILABLE:
        st.info("Install LIME to enable word-level explanations: pip install lime")
        return

    if df.empty:
        st.info("No reviews available to explain.")
        return

    explainer = load_lime_explainer()
    if explainer is None:
        st.info("LIME is not available in this environment.")
        return

    display_options = [
        f"{idx}: {row['predicted_sentiment']} - {str(row['review_text'])[:90]}"
        for idx, row in df.head(100).iterrows()
    ]
    selected = st.selectbox("Choose a review to explain", display_options)
    selected_idx = int(selected.split(":", 1)[0])
    row = df.loc[selected_idx]
    text_to_explain = str(row.get("model_input_text", row["review_text"]))

    st.write("Model input")
    st.caption(text_to_explain)

    if st.button("Generate LIME Explanation"):
        probabilities = predict_sentiment_probabilities([text_to_explain])[0]
        predicted_id = int(np.argmax(probabilities))
        predicted_label = LIME_CLASS_NAMES[predicted_id]

        explanation = explainer.explain_instance(
            text_to_explain,
            predict_sentiment_probabilities,
            labels=[predicted_id],
            num_features=10,
            num_samples=500,
        )
        explanation_df = pd.DataFrame(
            explanation.as_list(label=predicted_id),
            columns=["word", "influence"],
        )
        explanation_df["direction"] = explanation_df["influence"].apply(
            lambda value: f"Pushes {predicted_label}" if value >= 0 else f"Pushes away from {predicted_label}"
        )

        st.write(f"Predicted sentiment: {predicted_label} ({probabilities[predicted_id]:.2%} confidence)")
        fig = px.bar(
            explanation_df,
            x="influence",
            y="word",
            color="direction",
            orientation="h",
            color_discrete_map={
                f"Pushes {predicted_label}": "#16a34a",
                f"Pushes away from {predicted_label}": "#dc2626",
            },
        )
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width="stretch")
        st.dataframe(explanation_df, width="stretch", hide_index=True)


def create_wordcloud(text_series, title):
    st.subheader(title)
    text = " ".join(text_series.dropna().astype(str).tolist()).strip()
    if not text:
        st.info("Not enough review text to generate this word cloud.")
        return

    if not WORDCLOUD_AVAILABLE:
        st.info("Install wordcloud to enable this visual: pip install wordcloud")
        return

    wordcloud = WordCloud(width=900, height=380, background_color="white", colormap="viridis").generate(text)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig, width="stretch")


def pct(part, whole):
    return 0 if whole == 0 else round((part / whole) * 100, 1)


def render_kpi_cards(df):
    total = len(df)
    positive = pct((df["predicted_sentiment"] == "Positive").sum(), total)
    neutral = pct((df["predicted_sentiment"] == "Neutral").sum(), total)
    negative = pct((df["predicted_sentiment"] == "Negative").sum(), total)

    negative_aspects = df[df["predicted_sentiment"] == "Negative"]["detected_aspect"]
    common_negative_aspect = "None detected"
    if not negative_aspects.empty:
        common_negative_aspect = negative_aspects.value_counts().idxmax()

    cards = st.columns(5)
    cards[0].metric("Total reviews", f"{total:,}")
    cards[1].metric("Positive %", f"{positive}%")
    cards[2].metric("Neutral %", f"{neutral}%")
    cards[3].metric("Negative %", f"{negative}%")
    cards[4].metric("Top negative aspect", common_negative_aspect)


def configure_page():
    st.set_page_config(
        page_title="Seller Sentiment Dashboard",
        page_icon="ST",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.5rem;}
        [data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
        }
        [data-testid="stMetric"] label,
        [data-testid="stMetric"] [data-testid="stMetricLabel"] {
            color: #334155;
            font-weight: 600;
        }
        [data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #0f172a;
            font-weight: 700;
        }
        .dashboard-band {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 16px;
            background: #ffffff;
            margin-bottom: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    configure_page()

    st.title("E-Commerce Sentiment Dashboard")
    st.caption("Upload your Shopee or Lazada reviews and inspect customer sentiment to improve your sales.")

    uploaded_file = st.file_uploader("Upload review CSV or Excel", type=["csv", "xlsx", "xls"])
    using_demo = uploaded_file is None

    if using_demo:
        raw_df = load_demo_data()
        st.info("Demo mode is active because no seller file was uploaded.")
    else:
        raw_df = read_uploaded_file(uploaded_file)
        st.dataframe(raw_df.head(20), width="stretch", hide_index=True)

    if "review_text" not in raw_df.columns:
        st.warning("Please upload a file with a review_text column.")
        st.stop()

    if "review_text_translated" in raw_df.columns:
        st.caption("Sentiment model input: review_text_translated")
    else:
        st.caption("Sentiment model input: review_text. For best DistilBERT results, upload a preprocessed file with review_text_translated.")

    analyzed_df = add_predictions(raw_df)
    dashboard_df = analyzed_df

    render_kpi_cards(dashboard_df)

    left, right = st.columns(2)
    with left:
        st.subheader("Sentiment Distribution")
        st.plotly_chart(plot_sentiment_distribution(dashboard_df), width="stretch")

    with right:
        st.subheader("Sentiment Trend")
        if "review_date" in dashboard_df.columns:
            trend_fig = plot_sentiment_trend(dashboard_df)
            if trend_fig is None:
                st.info("Time-series trend is unavailable because no valid review date values were provided.")
            else:
                st.plotly_chart(trend_fig, width="stretch")
        else:
            st.info("Time-series trend is unavailable because no review date column was provided.")

    st.subheader("Aspect Analysis")
    aspect_fig, aspect_sentiment_table, negative_ranking = plot_aspect_analysis(dashboard_df)
    aspect_left, aspect_right = st.columns([1.15, 1])
    with aspect_left:
        st.plotly_chart(aspect_fig, width="stretch")
    with aspect_right:
        st.write("Aspect vs Sentiment")
        st.dataframe(aspect_sentiment_table, width="stretch", hide_index=True)
        st.write("Negative Aspect Ranking")
        if negative_ranking.empty:
            st.info("No negative reviews detected by the current sentiment placeholder.")
        else:
            st.dataframe(negative_ranking, width="stretch", hide_index=True)

    st.subheader("Rule-Based Improvement Suggestions")
    for recommendation in generate_recommendations(dashboard_df):
        st.success(recommendation)

    wc_left, wc_right = st.columns(2)
    with wc_left:
        create_wordcloud(dashboard_df["review_text"], "Word Cloud: All Reviews")
    with wc_right:
        create_wordcloud(
            dashboard_df[dashboard_df["predicted_sentiment"] == "Negative"]["review_text"],
            "Word Cloud: Negative Reviews Only",
        )

    st.subheader("Analyzed Review Table")
    table_columns = ["review_text", "predicted_sentiment", "detected_aspect"]
    optional_columns = ["review_date", "star_rating"]
    table_columns.extend([column for column in optional_columns if column in dashboard_df.columns])
    st.dataframe(dashboard_df[table_columns], width="stretch", hide_index=True)

    render_lime_explanation(dashboard_df)

    download_df = raw_df.copy()
    download_df["predicted_sentiment"] = analyzed_df["predicted_sentiment"]
    download_df["detected_aspect"] = analyzed_df["detected_aspect"]
    st.download_button(
        "Download Analyzed Reviews CSV",
        data=download_df.to_csv(index=False).encode("utf-8"),
        file_name="analyzed_reviews.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
