# Import tools
import os
import ast
import pickle
import warnings

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DATA_PATH = "../data/master_clean.csv"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

PALETTE = {
    "positive": "#4CAF50",
    "neutral":  "#E7F5BC",
    "negative": "#F44336",
}

CAT_COLORS = ["#90CAF9", "#A5D6A7", "#FFCC80", "#F48FB1"] 

TFIDF_PARAMS = dict(
    max_features   = 5000,
    ngram_range    = (1, 2),
    min_df         = 2,
    max_df         = 0.95,
    sublinear_tf   = True,
)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

df = pd.read_csv(DATA_PATH)

# Parse token lists stored as string representation
def parse_tokens(raw):

    if isinstance(raw, str):
        try:
            tokens = ast.literal_eval(raw)
            return " ".join(tokens) if isinstance(tokens, list) else raw
        except(ValueError, SyntaxError):
            return raw
    return ""

df["token_text"] = df["review_tokens"].apply(parse_tokens)

train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
print(f"Total rows: {len(df)}")
print(f"Train rows: {len(train_df)}")

# ─────────────────────────────────────────────
# 2. TF-IDF VECTORIZATION
# ─────────────────────────────────────────────

vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
X_train = vectorizer.fit_transform(train_df["token_text"])

print(f"Vocublary size: {len(vectorizer.vocabulary_)}")
print(f"Feature Matrix: {X_train.shape}")

# Save tf-idf features
with open(f"{OUTPUT_DIR}/tfidf_features.pkl", "wb") as f:
    pickle.dump(X_train, f)

with open(f"{OUTPUT_DIR}/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# ─────────────────────────────────────────────
# 3. WORD CLOUDS
# ─────────────────────────────────────────────

def make_wordcloud(texts, color, title, save_path):
    combined = " ".join(texts)
    wc = WordCloud(
        width            = 900,
        height           = 500,
        background_color = "white",
        colormap         = None,
        color_func       = lambda *a, **kw: color,
        collocations     = False,
        max_words        = 120,
        prefer_horizontal= 0.85,
    ).generate(combined)
 
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path.split('/')[-1]}")
 
for sentiment, color in PALETTE.items():
    subset = train_df[train_df["sentiment"] == sentiment]["token_text"].tolist()
    make_wordcloud(
        texts     = subset,
        color     = color,
        title     = f"Most Frequent Terms — {sentiment.capitalize()} Reviews",
        save_path = f"{OUTPUT_DIR}/fig_wordcloud_{sentiment}.png",
    )

# ─────────────────────────────────────────────
# 4. SENTIMENT CLASS DISTRIBUTION CHART
# ─────────────────────────────────────────────

sentiment_counts = df["sentiment"].value_counts().reindex(["positive", "neutral", "negative"])
total = sentiment_counts.sum()
 
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(
    sentiment_counts.index,
    sentiment_counts.values,
    color  = [PALETTE[s] for s in sentiment_counts.index],
    width  = 0.5,
    edgecolor = "white",
    linewidth = 1.5,
)
 
for bar, val in zip(bars, sentiment_counts.values):
    pct = val / total * 100
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 4,
        f"{val}\n({pct:.1f}%)",
        ha="center", va="bottom", fontsize=11, fontweight="bold",
    )
 
ax.set_title("Class Distribution (Sentiment)", fontsize=15, fontweight="bold", pad=14)
ax.set_xlabel("Sentiment", fontsize=12)
ax.set_ylabel("Number of Reviews", fontsize=12)
ax.set_ylim(0, sentiment_counts.max() * 1.18)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="x", labelsize=12)
 
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig_class_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────
# 5. CATEGORY DISTRIBUTION CHART
# ─────────────────────────────────────────────

cat_counts = df["category"].value_counts()
colors_mapped = CAT_COLORS[:len(cat_counts)]
 
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("Category Distribution", fontsize=15, fontweight="bold", y=1.01)

# Bar Chart
ax_bar = axes[0]
bars = ax_bar.bar(
    cat_counts.index,
    cat_counts.values,
    color    = colors_mapped,
    width    = 0.55,
    edgecolor= "white",
    linewidth= 1.5,
)

for bar, val in zip(bars, cat_counts.values):
    ax_bar.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1.5,
        str(val),
        ha="center", va="bottom", fontsize=11, fontweight="bold",
    )

ax_bar.set_title("Count per Category", fontsize=12)
ax_bar.set_xlabel("Category", fontsize=11)
ax_bar.set_ylabel("Number of Reviews", fontsize=11)
ax_bar.set_ylim(0, cat_counts.max() * 1.15)
ax_bar.spines[["top", "right"]].set_visible(False)
ax_bar.tick_params(axis="x", rotation=20)
 
# Pie Chart
ax_pie = axes[1]
wedges, texts, autotexts = ax_pie.pie(
    cat_counts.values,
    labels       = None,
    colors       = colors_mapped,
    autopct      = "%1.1f%%",
    startangle   = 140,
    pctdistance  = 0.78,
    wedgeprops   = dict(linewidth=1.5, edgecolor="white"),
)
for autotext in autotexts:
    autotext.set_fontsize(11)
    autotext.set_fontweight("bold")
 
legend_patches = [
    mpatches.Patch(facecolor=c, label=lbl.capitalize())
    for c, lbl in zip(colors_mapped, cat_counts.index)
]
ax_pie.legend(
    handles   = legend_patches,
    loc       = "lower center",
    bbox_to_anchor = (0.5, -0.12),
    ncol      = 2,
    fontsize  = 10,
    frameon   = False,
)
ax_pie.set_title("Proportional Share", fontsize=12)
 
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig_category_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────
# 6. SUMMARY
# ─────────────────────────────────────────────

print(f"tfidf_features.pkl shape: {X_train.shape}")
print(f"Vectorizer vocab size: {len(vectorizer.vocabulary_)}")
print(f"Figures saved to: {OUTPUT_DIR}")