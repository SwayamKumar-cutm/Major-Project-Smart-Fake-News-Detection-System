from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT/'models'/'model.joblib'; VECTOR = ROOT/'models'/'tfidf_vectorizer.joblib'; METRICS = ROOT/'outputs'/'evaluation_results'/'model_metrics.json'
st.set_page_config(page_title='Smart Fake News Detection', page_icon='📰')
st.title('Smart Fake News Detection System')
st.warning('Educational text classifier only. It does not verify facts or sources, and predictions may be wrong.')
if not MODEL.exists() or not VECTOR.exists(): st.error('Run python src/train_model.py from the project root first.'); st.stop()
model, vec = joblib.load(MODEL), joblib.load(VECTOR)
text = st.text_area('Paste a news headline or article', height=180)
if st.button('Analyze news', type='primary'):
    if not text.strip(): st.error('Enter some text first.')
    else:
        X = vec.transform([text]); prediction = model.predict(X)[0]
        st.write('Model prediction:', 'LIKELY REAL' if prediction == 'REAL' else 'LIKELY FAKE')
        if hasattr(model,'predict_proba'): st.caption('Scores are model outputs, not verified truth probabilities.'); st.dataframe(pd.DataFrame({'Class':model.classes_,'Score':model.predict_proba(X)[0]}))
if METRICS.exists():
    st.subheader('Model evaluation'); data=json.loads(METRICS.read_text(encoding='utf-8'))
    st.caption('Selected model: '+str(data.get('selected_model')))
    st.dataframe(pd.DataFrame([{'Model':k,**v} for k,v in data.get('metrics',{}).items()]).round(3),use_container_width=True)