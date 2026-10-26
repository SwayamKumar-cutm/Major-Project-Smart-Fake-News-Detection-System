import pandas as pd

def clean_dataset(path):
    """Load and clean a labeled fake-news CSV with text and label columns."""
    df = pd.read_csv(path)
    required = {'text', 'label'}
    if not required.issubset(df.columns):
        raise ValueError('Dataset must contain text and label columns.')
    df = df.dropna(subset=['text', 'label']).drop_duplicates().copy()
    df['text'] = df['text'].astype(str).str.strip()
    df['label'] = df['label'].astype(str).str.upper().str.strip()
    return df[(df['text'] != '') & df['label'].isin(['REAL', 'FAKE'])]
