# =============================================================================
# preprocessing.py
# =============================================================================

import re
import time
import nltk
import pandas as pd
from deep_translator import GoogleTranslator
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

slang_dict = {
    # =========================
    # BASIC SHORT FORMS
    # =========================
    'x': 'tidak',
    'tak': 'tidak',
    'takde': 'tidak ada',
    'xde': 'tidak ada',
    'xtau': 'tidak tahu',
    'xpe': 'tidak apa',
    'xboleh': 'tidak boleh',
    'xbole': 'tidak boleh',
    'dah': 'sudah',
    'dh': 'sudah',

    'sy': 'saya',
    'aq': 'saya',
    'ak': 'saya',

    'u': 'you',
    'ur': 'your',

    # =========================
    # CONNECTORS / FILLERS
    # =========================
    'dgn': 'dengan',
    'dengn': 'dengan',
    'ngn': 'dengan',
    '&': 'dan',
    'n': 'dan',

    'tp': 'tetapi',
    'tpi': 'tetapi',
    'tapi': 'tetapi',

    'je': 'sahaja',
    'jer': 'sahaja',

    'ni': 'ini',
    'tu': 'itu',
    'kt': 'dekat',

    'yg': 'yang',
    'utk': 'untuk',

    'sbb': 'sebab',
    'sb': 'sebab',

    'pastu': 'selepas itu',
    'pastuh': 'selepas itu',

    # =========================
    # QUANTITY / TIME
    # =========================
    'bape': 'berapa',
    'brp': 'berapa',

    'lame': 'lama',
    'lamer': 'lama',

    'jgk': 'juga',
    'juga': 'juga',

    # =========================
    # PRODUCT / SHOPEE TERMS
    # =========================
    'brg': 'barang',
    'barag': 'barang',
    'barang2': 'barang-barang',

    'buyi': 'bunyi',
    'bnyi': 'bunyi',
    'bunyii': 'bunyi',

    'gune': 'guna',
    'pakai': 'guna',

    'rm': 'ringgit',
    'qty': 'quantity',

    # =========================
    # QUALITY / CONDITION WORDS
    # =========================
    'ok': 'okay',
    'okayy': 'okay',
    'okey': 'okay',

    'btol': 'betul',
    'betui': 'betul',

    'rengan': 'ringan',
    'berat2': 'berat',

    'asik': 'asyik',
    'asyik2': 'asyik',

    'sgt': 'sangat',
    'sangat2': 'sangat',

    'lame': 'lama',

    # =========================
    # EMOTIONAL / EXPRESSIONS
    # =========================
    'haha': '',
    'huhu': '',
    'hehe': '',
    'hmm': '',
    'la': '',

    # =========================
    # ENGLISH MIX / COMMON CHAT
    # =========================
    'pls': 'please',
    'thx': 'thank you',
    'tq': 'thank you',
    'tqvm': 'thank you very much',

    'alr': 'already'
}

def init_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)

def safe_translate(text, max_length=4500, retries=3, delay=2):
    if not isinstance(text, str) or text.strip() == '':
        return text

    try:
        if len(text) > max_length:
            chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            result = []

            for chunk in chunks:
                for attempt in range(retries):
                    try:
                        result.append(
                            GoogleTranslator(source='auto', target='en').translate(chunk)
                        )
                        time.sleep(delay)
                        break
                    except Exception:
                        if attempt < retries - 1:
                            time.sleep(delay * 2)
                        else:
                            result.append(chunk)

            return ' '.join(result)

        for attempt in range(retries):
            try:
                return GoogleTranslator(source='auto', target='en').translate(text)
            except Exception:
                if attempt < retries - 1:
                    time.sleep(delay * 2)
                else:
                    return text

    except Exception:
        return text
    
def preprocess_dataframe(df):
    """
    Full preprocessing pipeline for uploaded CSV (single input file).
    Returns processed dataframe.
    """

    df = df.copy()

    # ======================
    # STEP 1: SLANG NORMALIZATION
    # ======================
    def normalize_slang(text):
        if not isinstance(text, str):
            return text

        words = text.split()

        normalized_words = [
            slang_dict.get(word.lower(), word)
            for word in words
        ]

        return ' '.join(normalized_words)

    df['review_text_fixed'] = df['review_text'].apply(normalize_slang)

    # ======================
    # STEP 1: TRANSLATION
    # ======================
    df['review_text_translated'] = df['review_text_fixed'].apply(safe_translate)

    # ======================
    # STEP 2: CLEANING
    # ======================

    text_cols = ['review_text_translated']

    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002700-\U000027BF"
        u"\U0001F900-\U0001F9FF"
        u"\U00002500-\U00002BEF"
    "]+", flags=re.UNICODE)

    for col in text_cols:
        df[col] = df[col].str.lower()
        df[col] = df[col].str.replace(r'http\S+|www\S+', '', regex=True)
        df[col] = df[col].apply(lambda x: emoji_pattern.sub('', x) if isinstance(x, str) else x)
        df[col] = df[col].str.replace(r'[^\w\s]', '', regex=True)
        df[col] = df[col].str.replace(r'\d+', '', regex=True)
        df[col] = df[col].apply(lambda x: re.sub(r'(.)\1{2,}', r'\1', x) if isinstance(x, str) else x)

    stop_words = set(stopwords.words('english'))

    for col in text_cols:
        df[col] = df[col].apply(
            lambda x: ' '.join([w for w in x.split() if w not in stop_words])
            if isinstance(x, str) else x
        )

        df[col] = df[col].str.strip().str.replace(r'\s+', ' ', regex=True)

    # ======================
    # STEP 3: LEMMATIZATION
    # ======================

    lemmatizer = WordNetLemmatizer()

    def lemmatize_text_pos(text):
        if not isinstance(text, str):
            return text

        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)

        lemmatized = [
            lemmatizer.lemmatize(word, get_wordnet_pos(tag))
            for word, tag in pos_tags
        ]

        return ' '.join(lemmatized)

    df['review_text_normalized'] = df['review_text_translated'].apply(lemmatize_text_pos)

    # ======================
    # STEP 4: TOKENIZATION
    # ======================

    df['review_tokens'] = df['review_text_normalized'].apply(
        lambda x: word_tokenize(x) if isinstance(x, str) else []
    )

    return df

def split_dataset(df):
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df['sentiment']
    )

    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df['sentiment']
    )

    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'

    final_df = pd.concat([train_df, val_df, test_df])

    return train_df, val_df, test_df, final_df

# =============================================================================
# TO USE FOR WEBSITE
# from preprocessing import preprocess_dataframe
#
# @app.route('/upload', methods=['POST'])
# def upload():
#    file = request.files['file']
#    df = pd.read_csv(file)

#    processed_df = preprocess_dataframe(df)

#    return processed_df.to_json()
# =============================================================================