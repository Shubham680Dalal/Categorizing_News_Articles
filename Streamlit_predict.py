import streamlit as st
import pandas as pd
import datetime
import preprocessing as pf
import pickle
import joblib


st.set_page_config(layout="wide")
param_dict=pickle.load(open('best_estimator.pkl','rb'))

checkpoint_index=param_dict['checkpoint_index']  ##one already covered
best_param_index=param_dict['best_param_index']
best_estimator=joblib.load("only_model.joblib")


best_val_accuracy=param_dict['best_val_accuracy']
best_val_ndcg=param_dict['best_val_ndcg']


vectorizer=param_dict['vectorizer']


numeric_cols=param_dict['numeric_cols']
scalar=param_dict['scalar']
target_encode=param_dict['target_encode']
rev_target_encode=param_dict['rev_target_encode']

with st.form(key='categorizing news articles'):
    st.write("## Identifying Article Category")
    st.write("### Among Buisness, Entertainment, Politics, Sports and Technology")

    article=st.text_area("Article", "worldcom boss left books alone former worldcom boss bernie ebbers who is accused of overseeing an $11bn (£5.8bn) fraud never made accounting decisions a witness has told jurors. david myers made the comments under questioning by defence lawyers who have been arguing that mr ebbers was not responsible for worldcom s problems. the phone company collapsed in 2002 and prosecutors claim that losses were hidden to protect the firm s shares. mr myers has already pleaded guilty to fraud and is assisting prosecutors. on monday defence lawyer reid weingarten tried to distance his client from the allegations. during cross examination he asked mr myers if he ever knew mr ebbers make an accounting decision . not that i am aware of mr myers replied. did you ever know mr ebbers to make an accounting entry into worldcom books mr weingarten pressed. no replied the witness. mr myers has admitted that he ordered false accounting entries at the request of former worldcom chief financial officer scott sullivan. defence lawyers have been trying to paint mr sullivan who has admitted fraud and will testify later in the trial as the mastermind behind worldcom s accounting house of cards. mr ebbers team meanwhile are looking to portray him as an affable boss who by his own admission is more pe graduate than economist. whatever his abilities mr ebbers transformed worldcom from a relative unknown into a $160bn telecoms giant and investor darling of the late 1990s. worldcom s problems mounted however as competition increased and the telecoms boom petered out. when the firm finally collapsed shareholders lost about $180bn and 20 000 workers lost their jobs. mr ebbers trial is expected to last two months and if found guilty the former ceo faces a substantial jail sentence. he has firmly declared his innocence.")
    
    submit_button = st.form_submit_button(label="Submit")

if submit_button:
    df = pd.DataFrame([[article]],columns='Article')
    preprocessor = pf.TextPreprocessor()
    df = preprocessor.preprocess_dataframe(df)

    df=pf.transform_test(df, col='tokens', vectorizer=vectorizer)
    df=pf.transform_standardize_data(df,numeric_cols,scalar)
    st.write(df)

    y_pred_probs=pd.DataFrame(best_estimator.predict_proba(df),columns=best_estimator.classes_)
    st.write(y_pred_probs)

