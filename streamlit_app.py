import re
import json
import html
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


def predict_sentiments(texts, batch_size=32, progress_callback=None):
    tokenizer, model, id2label = load_distilbert_model()
    texts = [str(text) for text in texts]
    total = len(texts)
    predictions = []

    if tokenizer is None or model is None:
        for position, text in enumerate(texts, start=1):
            predictions.append(predict_rule_based_sentiment(text))
            if progress_callback is not None:
                progress_callback(position, total)
        return predictions

    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        probabilities = predict_sentiment_probabilities(batch)
        pred_ids = np.argmax(probabilities, axis=1)
        for pred_id in pred_ids:
            label = id2label[int(pred_id)]
            predictions.append(SENTIMENT_LABELS.get(str(label).lower(), str(label).title()))

        if progress_callback is not None:
            progress_callback(min(start + len(batch), total), total)

    return predictions


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


def guess_date_column(columns):
    preferred_names = [
        "review_date",
        "date",
        "review date",
        "created_at",
        "created date",
        "timestamp",
        "time_stamp",
        "time",
    ]
    normalized = {str(column).lower().strip(): column for column in columns}
    for name in preferred_names:
        if name in normalized:
            return normalized[name]

    for column in columns:
        lowered = str(column).lower()
        if "date" in lowered or "time" in lowered:
            return column

    return None


def plot_sentiment_trend(df, date_col):
    trend = df.copy()
    trend["_trend_date"] = pd.to_datetime(trend[date_col], errors="coerce")
    trend = trend.dropna(subset=["_trend_date"])
    if trend.empty:
        return None

    trend["date"] = trend["_trend_date"].dt.date
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
    frequency_fig.update_layout(showlegend=False, height=300, margin=dict(l=10, r=10, t=10, b=10))

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


def guess_text_column(columns):
    preferred_names = [
        "review_content",
        "review content",
        "review_body",
        "review body",
        "review_text",
        "review text",
        "review_description",
        "review description",
        "review",
        "reviews",
        "comment",
        "comments",
        "feedback",
        "text",
        "content",
    ]
    normalized = {str(column).lower().strip(): column for column in columns}
    for name in preferred_names:
        if name in normalized:
            return normalized[name]

    for column in columns:
        lowered = str(column).lower()
        if any(keyword in lowered for keyword in ["review", "comment", "feedback", "text"]):
            return column

    return columns[0] if len(columns) else None


def is_text_candidate_column(column):
    lowered = str(column).lower().strip()
    excluded_keywords = [
        "id",
        "rating",
        "score",
        "count",
        "date",
        "time",
        "price",
        "percentage",
        "product",
        "category",
    ]
    if any(keyword in lowered for keyword in excluded_keywords):
        return False
    return any(keyword in lowered for keyword in ["review", "comment", "feedback", "text", "content"])


def get_candidate_text_columns(columns):
    candidates = [column for column in columns if is_text_candidate_column(column)]
    priority = ["content", "body", "text", "comment", "feedback", "review", "title"]
    return sorted(
        candidates,
        key=lambda column: next(
            (index for index, keyword in enumerate(priority) if keyword in str(column).lower()),
            len(priority),
        ),
    )


def add_predictions(df, review_text_col, sentiment_text_col=None, progress_callback=None):
    analyzed = df.copy()
    analyzed["review_text"] = analyzed[review_text_col].fillna("").astype(str)
    if sentiment_text_col is None:
        sentiment_text_col = "review_text_translated" if "review_text_translated" in analyzed.columns else "review_text"
    analyzed[sentiment_text_col] = analyzed[sentiment_text_col].fillna("").astype(str)
    analyzed["clean_review_text"] = analyzed["review_text"].apply(clean_text)
    analyzed["model_input_text"] = analyzed[sentiment_text_col]

    analyzed["predicted_sentiment"] = predict_sentiments(
        analyzed["model_input_text"].tolist(),
        batch_size=32,
        progress_callback=progress_callback,
    )
    analyzed["detected_aspect"] = analyzed["review_text"].apply(detect_aspect)
    return analyzed


def render_lime_explanation(df, show_header=True):
    if show_header:
        st.subheader("Explain One Sentiment Prediction")
    else:
        st.markdown("**Explain one sentiment prediction**")

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

    card_data = [
        ("Total reviews", f"{total:,}", "kpi-card-blue"),
        ("Positive %", f"{positive}%", "kpi-card-green"),
        ("Neutral %", f"{neutral}%", "kpi-card-amber"),
        ("Negative %", f"{negative}%", "kpi-card-red"),
        ("Top negative aspect", common_negative_aspect, "kpi-card-violet"),
    ]

    cards = st.columns(5)
    for column, (label, value, class_name) in zip(cards, card_data):
        column.markdown(
            f"""
            <div class="kpi-card {class_name}">
                <div class="kpi-label">{html.escape(label)}</div>
                <div class="kpi-value">{html.escape(str(value))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


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
        .kpi-card {
            min-height: 104px;
            border-radius: 8px;
            padding: 14px 16px;
            border: 1px solid rgba(148, 163, 184, 0.35);
            box-shadow: 0 10px 22px rgba(15, 23, 42, 0.14);
        }
        .kpi-label {
            font-size: 0.82rem;
            font-weight: 700;
            color: rgba(15, 23, 42, 0.72);
            margin-bottom: 8px;
        }
        .kpi-value {
            font-size: clamp(1.45rem, 1.8vw, 2rem);
            line-height: 1.15;
            font-weight: 800;
            color: #0f172a;
            overflow-wrap: anywhere;
        }
        .kpi-card-blue { background: linear-gradient(135deg, #e0f2fe 0%, #f8fafc 72%); border-color: #7dd3fc; }
        .kpi-card-green { background: linear-gradient(135deg, #dcfce7 0%, #f8fafc 72%); border-color: #86efac; }
        .kpi-card-amber { background: linear-gradient(135deg, #fef3c7 0%, #f8fafc 72%); border-color: #fcd34d; }
        .kpi-card-red { background: linear-gradient(135deg, #fee2e2 0%, #f8fafc 72%); border-color: #fca5a5; }
        .kpi-card-violet { background: linear-gradient(135deg, #ede9fe 0%, #f8fafc 72%); border-color: #c4b5fd; }
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

    st.title("E-Commerce Review Sentiment Dashboard 🛒")
    st.caption("Analyse customer sentiment, review aspects, and common product or service issues.")

    uploaded_file = st.file_uploader(
        "Upload review CSV or Excel",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
    )
    if uploaded_file is None:
        st.stop()

    raw_df = read_uploaded_file(uploaded_file)
    st.dataframe(raw_df.head(20), width="stretch", hide_index=True)

    if raw_df.empty or len(raw_df.columns) == 0:
        st.warning("The uploaded file is empty or does not contain any columns.")
        st.stop()

    columns = list(raw_df.columns)
    text_candidates = get_candidate_text_columns(columns)
    default_review_col = guess_text_column(columns)
    if len(text_candidates) > 1:
        review_text_col = st.selectbox(
            "Select the column that contains customer reviews",
            text_candidates,
            index=text_candidates.index(default_review_col) if default_review_col in text_candidates else 0,
        )
    else:
        review_text_col = text_candidates[0] if text_candidates else default_review_col
        st.caption(f"Review column detected: {review_text_col}")

    sentiment_text_col = "review_text_translated" if "review_text_translated" in raw_df.columns else review_text_col

    progress_text = st.empty()
    progress_bar = st.progress(0)

    def update_prediction_progress(done, total):
        percent = 0 if total == 0 else done / total
        progress_bar.progress(percent)
        progress_text.caption(f"Analysing reviews... {done:,} of {total:,}")

    analyzed_df = add_predictions(raw_df, review_text_col, sentiment_text_col, update_prediction_progress)
    progress_bar.empty()
    progress_text.empty()

    dashboard_df = analyzed_df

    render_kpi_cards(dashboard_df)

    left, right = st.columns(2)
    with left:
        st.subheader("Sentiment Distribution")
        st.plotly_chart(plot_sentiment_distribution(dashboard_df), width="stretch")

    with right:
        st.subheader("Sentiment Trend")
        date_col = guess_date_column(dashboard_df.columns)
        if date_col is not None:
            trend_fig = plot_sentiment_trend(dashboard_df, date_col)
            if trend_fig is None:
                st.info("Time-series trend is unavailable because no valid review date values were provided.")
            else:
                st.plotly_chart(trend_fig, width="stretch")
        else:
            st.info("Time-series trend unavailable because no date information provided.")

    st.subheader("Aspect Analysis")
    aspect_fig, aspect_sentiment_table, negative_ranking = plot_aspect_analysis(dashboard_df)
    aspect_chart_col, aspect_table_col, negative_table_col = st.columns([1.05, 1, 0.85])
    with aspect_chart_col:
        st.plotly_chart(aspect_fig, width="stretch")
    with aspect_table_col:
        st.markdown("**Aspect vs Sentiment**")
        st.dataframe(aspect_sentiment_table, width="stretch", hide_index=True, height=300)
    with negative_table_col:
        st.markdown("**Negative Aspect Ranking**")
        if negative_ranking.empty:
            st.info("No negative reviews detected.")
        else:
            st.dataframe(negative_ranking, width="stretch", hide_index=True, height=300)

    wc_left, wc_right = st.columns(2)
    with wc_left:
        create_wordcloud(dashboard_df["review_text"], "Word Cloud: All Reviews")
    with wc_right:
        create_wordcloud(
            dashboard_df[dashboard_df["predicted_sentiment"] == "Negative"]["review_text"],
            "Word Cloud: Negative Reviews Only",
        )

    st.subheader("Analyzed Reviews")
    table_columns = ["review_text", "predicted_sentiment", "detected_aspect"]
    optional_columns = [guess_date_column(dashboard_df.columns), "star_rating"]
    table_columns.extend([column for column in optional_columns if column in dashboard_df.columns])
    st.dataframe(dashboard_df[table_columns], width="stretch", hide_index=True)

    with st.expander("Explain a selected review prediction"):
        render_lime_explanation(dashboard_df, show_header=False)

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
