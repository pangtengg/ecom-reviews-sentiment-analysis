# =============================================================================
# preprocessing.py
# Usage:
#   from preprocessing import preprocess_dataframe, split_dataset
#
#   df = pd.read_csv('your_file.csv')
#   processed_df = preprocess_dataframe(df)
#   train_df, val_df, test_df, final_df = split_dataset(processed_df)
# =============================================================================

import os
import re
import time
import nltk
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.model_selection import train_test_split
from deep_translator import GoogleTranslator

# Download required NLTK resources
nltk.download('stopwords',                      quiet=True)
nltk.download('punkt',                          quiet=True)
nltk.download('punkt_tab',                      quiet=True)
nltk.download('wordnet',                        quiet=True)
nltk.download('averaged_perceptron_tagger',     quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# ==============================================================================
# SLANG DICTIONARY
# ==============================================================================

SLANG_DICT = {
    # BASIC SHORT FORMS
    'x': 'tiada',        'tak': 'tiada',        'takde': 'tiada',
    'xde': 'tiada',      'xtau': 'tidak tahu',  'xpe': 'tidak apa',
    'xboleh': 'tidak boleh',                     'xbole': 'tidak boleh',
    'dah': 'sudah',      'dh': 'sudah',
    'sy': 'saya',        'aq': 'saya',           'ak': 'saya',
    'u': 'you',          'ur': 'your',

    # CONNECTORS / FILLERS
    'dgn': 'dengan',     'dengn': 'dengan',      'ngn': 'dengan',
    '&': 'dan',
    'tp': 'tetapi',      'tpi': 'tetapi',        'tapi': 'tetapi',
    'je': 'sahaja',      'jer': 'sahaja',
    'ni': 'ini',         'tu': 'itu',            'kt': 'dekat',
    'yg': 'yang',        'utk': 'untuk',
    'sbb': 'sebab',      'sb': 'sebab',
    'pastu': 'selepas itu',                       'pastuh': 'selepas itu',

    # QUANTITY / TIME
    'bape': 'berapa',    'brp': 'berapa',
    'lame': 'lama',      'lamer': 'lama',
    'jgk': 'juga',

    # PRODUCT TERMS
    'brg': 'barang',     'barag': 'barang',      'barang2': 'barang-barang',
    'buyi': 'bunyi',     'bnyi': 'bunyi',        'bunyii': 'bunyi',
    'gune': 'guna',      'pakai': 'guna',
    'qty': 'quantity',

    # QUALITY / CONDITION
    'ok': 'okay',        'okayy': 'okay',        'okey': 'okay',
    'btol': 'betul',     'betui': 'betul',
    'rengan': 'ringan',  'berat2': 'berat',
    'asik': 'asyik',     'asyik2': 'asyik',
    'sgt': 'sangat',     'sangat2': 'sangat',

    # MALAY WORDS / EXPRESSIONS
    'mmg': 'memang',     'mcm': 'macam',         'mcm2': 'macam-macam',
    'mkin': 'makin',     'kdg': 'kadang',        'kdg2': 'kadang-kadang',
    'lncr': 'lancar',    'tgk': 'tengok',        'tgok': 'tengok',
    'blh': 'boleh',      'bole': 'boleh',
    'dlm': 'dalam',      'drpd': 'daripada',     'drp': 'daripada',
    'kat': 'dekat',      'nk': 'mahu',           'nak': 'mahu',
    'dpt': 'dapat',      'dpt2': 'dapat-dapat',  'xdpt': 'tidak dapat',
    'harap2': 'harap-harap',
    'mantapp': 'mantap', 'mantappp': 'mantap',
    'bestla': 'best',    'lawa': 'cantik',
    'hampeh': 'hampa',
    'powerfull': 'powerful',                      'pawer': 'power',
    'wayer': 'wayar',    'cas': 'charge',         'caj': 'charge',
    'bot': 'bought',     'cust': 'customer',      'gt': 'got',

    # FILLER / EXPRESSIONS (removed)
    'sumpah': '', 'la': '',  'lah': '', 'leh': '',
    'woo': '',    'nih': '', 'wei': '', 'weh': '',
    'haha': '',   'hahaha': '', 'hahahha': '',
    'huhu': '',   'hehe': '',
    'hmm': '',    'hhm': '', 'meh': '', 'huh': '',
    'wow': '',    'woah': '',

    # ENGLISH MIX / CHAT ABBREVIATIONS
    'pls': 'please',         'thx': 'thank you',
    'tq': 'thank you',       'tqvm': 'thank you very much',
    'alr': 'already',        'tbh': 'to be honest',
    'imo': 'in my opinion',  'omg': 'oh my god',
    'info': 'information',   'pic': 'picture',       'pics': 'pictures',
    'msg': 'message',        'ori': 'original',      'orig': 'original',
    'rec': 'recommend',      'recom': 'recommend',
    'diff': 'different',     'prob': 'problem',      'probs': 'problems',
    'def': 'definitely',     'defo': 'definitely',   'esp': 'especially',
    'approx': 'approximately',                        'avg': 'average',
    'btw': 'by the way',     'fyi': 'for your information',
    'superb': 'superb',
    'gg': '', 'lol': '', 'xd': '',
}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def normalize_slang(text):
    """Replace slang/abbreviations with full forms using SLANG_DICT."""
    if not isinstance(text, str):
        return text
    words = text.split()
    normalized = [SLANG_DICT.get(word.lower(), word) for word in words]
    return ' '.join([w for w in normalized if w.strip() != ''])


def safe_translate(text, max_length=4500, retries=3, delay=2):
    """Translate text to English with retry logic and chunking for long texts."""
    if not isinstance(text, str) or text.strip() == '':
        return text
    try:
        if len(text) > max_length:
            chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            translated_chunks = []
            for chunk in chunks:
                for attempt in range(retries):
                    try:
                        translated_chunks.append(
                            GoogleTranslator(source='auto', target='en').translate(chunk)
                        )
                        time.sleep(delay)
                        break
                    except Exception:
                        if attempt < retries - 1:
                            time.sleep(delay * 2)
                        else:
                            translated_chunks.append(chunk)
            return ' '.join(translated_chunks)

        for attempt in range(retries):
            try:
                result = GoogleTranslator(source='auto', target='en').translate(text)
                time.sleep(0.5)
                return result
            except Exception:
                if attempt < retries - 1:
                    time.sleep(delay * 2)
                else:
                    return text
    except Exception as e:
        print(f"Translation failed: {e}")
        return text


EMOJI_PATTERN = re.compile("["
    u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F6FF"
    u"\U0001F1E0-\U0001F1FF"
    u"\U00002700-\U000027BF"
    u"\U0001F900-\U0001F9FF"
    u"\U00002500-\U00002BEF"
"]+", flags=re.UNICODE)

STOP_WORDS = set(stopwords.words('english'))


def clean_text(text):
    """Full text cleaning pipeline."""
    if not isinstance(text, str):
        return text
    text = text.lower()                                                   # 1. lowercase
    text = re.sub(r'http\S+|www\S+', '', text)                           # 2. remove URLs
    text = EMOJI_PATTERN.sub('', text)                                    # 3. remove emojis
    text = re.sub(r'\*{1,2}|_{1,2}', '', text)                          # 4. remove bold/italic markdown
    text = re.sub(r'rm\s*\d+|\d+\s*rm', '', text)                       # 5. remove currency (rm60, 60rm)
    text = re.sub(r'\d+', '', text)                                       # 6. remove remaining numbers
    text = re.sub(r':', ' ', text)                                        # 7. replace colons with space
    text = re.sub(r'[^\w\s]', '', text)                                   # 8. remove punctuation
    text = re.sub(r'(.)\1{2,}', r'\1', text)                            # 9. remove repeated chars
    text = ' '.join([w for w in text.split() if w not in STOP_WORDS])    # 10. remove stopwords
    text = re.sub(r'\s+', ' ', text).strip()                             # 11. remove extra whitespace
    return text


def get_wordnet_pos(tag):
    """Map POS tag to WordNet POS for accurate lemmatization."""
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


LEMMATIZER = WordNetLemmatizer()


def lemmatize_text_pos(text):
    """Lemmatize with POS tagging for accurate base-form reduction."""
    if not isinstance(text, str):
        return text
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    return ' '.join([
        LEMMATIZER.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in pos_tags
    ])


def tokenize_text(text):
    """Split text into word tokens."""
    if not isinstance(text, str):
        return []
    return word_tokenize(text)


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def preprocess_dataframe(df):
    """
    Full preprocessing pipeline for a single user-uploaded CSV.

    Expected column: 'review_text'
    Returns: processed DataFrame with added columns:
        - review_text_translated
        - review_text_normalized
        - review_tokens
    """
    df = df.copy()

    # Step 1: Slang normalization
    print("Step 1/5: Normalizing slang...")
    df['review_text'] = df['review_text'].apply(normalize_slang)

    # Step 2: Translation
    print("Step 2/5: Translating to English...")
    df['review_text_translated'] = df['review_text'].apply(safe_translate)
    print("         Translation complete.")

    # Step 3: Text cleaning
    print("Step 3/5: Cleaning text...")
    df['review_text_translated'] = df['review_text_translated'].apply(clean_text)

    # Step 4: Normalization (POS-aware lemmatization)
    print("Step 4/5: Lemmatizing...")
    df['review_text_normalized'] = df['review_text_translated'].apply(lemmatize_text_pos)

    # Step 5: Tokenization
    print("Step 5/5: Tokenizing...")
    df['review_tokens'] = df['review_text_normalized'].apply(tokenize_text)

    print("Preprocessing complete.")
    return df


def split_dataset(df):
    """
    Split dataframe into train (70%), validation (15%), test (15%).
    Requires a 'sentiment' column for stratification.

    Returns: train_df, val_df, test_df, final_df (all combined with 'split' column)
    """
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df['sentiment']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df['sentiment']
    )

    train_df = train_df.copy()
    val_df   = val_df.copy()
    test_df  = test_df.copy()

    train_df['split'] = 'train'
    val_df['split']   = 'val'
    test_df['split']  = 'test'

    final_df = pd.concat([train_df, val_df, test_df]).sort_index()

    print(f"\nData split:")
    print(f"  Train : {len(train_df)} rows ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val   : {len(val_df)} rows  ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test  : {len(test_df)} rows  ({len(test_df)/len(df)*100:.1f}%)")

    return train_df, val_df, test_df, final_df


# ==============================================================================
# COMMAND LINE USAGE
# python preprocessing.py --input your_file.csv --output cleaned.csv
# ==============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess a review CSV file.')
    parser.add_argument('--input',  required=True,  help='Path to input CSV file')
    parser.add_argument('--output', default=None,    help='Path to save cleaned CSV (optional)')
    parser.add_argument('--split',  action='store_true', help='Also split into train/val/test')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded: {df.shape[0]} rows")

    processed_df = preprocess_dataframe(df)

    if args.split:
        _, _, _, processed_df = split_dataset(processed_df)

    output_path = args.output or args.input.replace('.csv', '_clean.csv')
    processed_df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

# =============================================================================
# TO USE FOR WEBSITE
# from preprocessing import preprocess_dataframe, split_dataset
#
# @app.route('/upload', methods=['POST'])
# def upload():
#     file = request.files['file']
#     df = pd.read_csv(file)
#     processed_df = preprocess_dataframe(df)
#     return processed_df.to_json()
# =============================================================================