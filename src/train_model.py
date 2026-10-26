from pathlib import Path
import json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'fake_news_dataset.csv'
MODELS = ROOT / 'models'; OUT = ROOT / 'outputs' / 'evaluation_results'
MODELS.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(DATA).dropna(subset=['text', 'label']).drop_duplicates()
df['text'] = df['text'].astype(str).str.strip(); df['label'] = df['label'].astype(str).str.upper().str.strip()
df = df[(df.text != '') & df.label.isin(['REAL', 'FAKE'])]
Xtr, Xte, ytr, yte = train_test_split(df.text, df.label, test_size=.25, random_state=42, stratify=df.label)
vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2)); A = vec.fit_transform(Xtr); B = vec.transform(Xte)
models = {'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42), 'Multinomial Naive Bayes': MultinomialNB()}
results = {}
for name, model in models.items():
    model.fit(A, ytr); p = model.predict(B)
    results[name] = {'accuracy': float(accuracy_score(yte,p)), 'precision_macro': float(precision_score(yte,p,average='macro',zero_division=0)), 'recall_macro': float(recall_score(yte,p,average='macro',zero_division=0)), 'f1_macro': float(f1_score(yte,p,average='macro',zero_division=0))}
best = max(results, key=lambda n: results[n]['f1_macro']); joblib.dump(models[best], MODELS/'model.joblib'); joblib.dump(vec, MODELS/'tfidf_vectorizer.joblib')
(OUT/'model_metrics.json').write_text(json.dumps({'selected_model':best,'metrics':results,'test_rows':int(len(yte)),'note':'Scores reflect only the supplied dataset.'},indent=2),encoding='utf-8')
pd.DataFrame([{'model':k,**v} for k,v in results.items()]).to_csv(OUT/'model_comparison.csv',index=False)
print('Selected model:', best); print(json.dumps(results,indent=2))