import streamlit as st
import pandas as pd
import datetime
from lime.lime_tabular import LimeTabularExplainer
import preprocessing as pf
import pickle
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
param_dict=pickle.load(open('best_estimator.pkl','rb'))

checkpoint_index=param_dict['checkpoint_index']  ##one already covered
best_param_index=param_dict['best_param_index']
best_estimator=joblib.load("only_model.joblib")


best_val_accuracy=param_dict['best_val_accuracy']
best_val_ndcg=param_dict['best_val_ndcg']


vectorizer=param_dict['vectorizer']
xtrain_np=param_dict['xtrain_np']
feature_set=param_dict['feature_set']

numeric_cols=param_dict['numeric_cols']
scalar=param_dict['scalar']
target_encode=param_dict['target_encode']
rev_target_encode=param_dict['rev_target_encode']

with st.form(key='categorizing news articles'):
    st.write("## Identifying Article Category")
    st.write("### Between Buisness, Entertainment, Politics, Sports and Technology")

    article=st.text_area("Article", "worldcom boss left books alone former worldcom boss bernie ebbers who is accused of overseeing an $11bn (£5.8bn) fraud never made accounting decisions a witness has told jurors. david myers made the comments under questioning by defence lawyers who have been arguing that mr ebbers was not responsible for worldcom s problems. the phone company collapsed in 2002 and prosecutors claim that losses were hidden to protect the firm s shares. mr myers has already pleaded guilty to fraud and is assisting prosecutors. on monday defence lawyer reid weingarten tried to distance his client from the allegations. during cross examination he asked mr myers if he ever knew mr ebbers make an accounting decision . not that i am aware of mr myers replied. did you ever know mr ebbers to make an accounting entry into worldcom books mr weingarten pressed. no replied the witness. mr myers has admitted that he ordered false accounting entries at the request of former worldcom chief financial officer scott sullivan. defence lawyers have been trying to paint mr sullivan who has admitted fraud and will testify later in the trial as the mastermind behind worldcom s accounting house of cards. mr ebbers team meanwhile are looking to portray him as an affable boss who by his own admission is more pe graduate than economist. whatever his abilities mr ebbers transformed worldcom from a relative unknown into a $160bn telecoms giant and investor darling of the late 1990s. worldcom s problems mounted however as competition increased and the telecoms boom petered out. when the firm finally collapsed shareholders lost about $180bn and 20 000 workers lost their jobs. mr ebbers trial is expected to last two months and if found guilty the former ceo faces a substantial jail sentence. he has firmly declared his innocence.")

    submit_button = st.form_submit_button(label="Submit")

if submit_button:
    df = pd.DataFrame([[article]],columns=['Article'])
    preprocessor = pf.TextPreprocessor()
    df = preprocessor.preprocess_dataframe(df)
    
    df=pf.transform_test(df, col='tokens', vectorizer=vectorizer)
    df=pf.transform_standardize_data(df,numeric_cols,scalar)
    #st.write(df)

    y_pred_probs=pd.DataFrame(best_estimator.predict_proba(df),columns=best_estimator.classes_)
    y_pred_probs.columns=[rev_target_encode[col] for col in y_pred_probs.columns]
    display_text = r'<span style="font-size:20px;">Probability that the Article is:<br>'

    y_pred_probs=y_pred_probs.iloc[0].sort_values(ascending=False)
    for ind in y_pred_probs.index:
        # Format the column name in red and uppercase, and percentage in bold red
        display_text += r'<span style="color:red;font-size:15px;">{} : </span> <span style="color:red; font-size:15px; font-weight:bold;">{:.1f}%</span><br>'.format(ind.title(), float(y_pred_probs[ind]) * 100)

    st.markdown(display_text+r'</span>', unsafe_allow_html=True)

    

    # Convert xtest to a numpy array (if not already)
    xtest_np = np.array(df[feature_set])

    # Initialize the LIME explainer
    lime_explainer = LimeTabularExplainer(
        training_data=xtrain_np,  # Training data
        mode='classification',      
        feature_names=feature_set,  # Feature names
        verbose=True,            # More detailed explanations
        random_state=42
    )

    # Explain the first 5 rows in xtest
    
    
    lime_exp = lime_explainer.explain_instance(
        data_row=xtest_np[0],  # Row to explain
        predict_fn=best_estimator.predict_proba  # Prediction function
    )

    df_explanation=pf.show_lime_with_original_values(lime_exp, scalar, numeric_cols)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("#### LIME Explanation (Text)")
        st.write(df_explanation)
        #lime_exp.show_in_notebook(show_table=True)
    with col2:

        # fig = lime_exp.as_pyplot_figure()
        # st.pyplot(fig)
        st.write("#### Considering following encoding for categories")
        st.write(rev_target_encode)
        html_exp = lime_exp.as_html()
        styled_html = f"""
        <div style="background-color:white; padding:10px; border-radius:5px;">
            {html_exp}
        </div>
        """

        st.components.v1.html(styled_html, height=500, scrolling=True)

    

    # Initialize SHAP explainer for the XGBoost model
    shap_explainer = shap.Explainer(best_estimator, pd.DataFrame(xtrain_np,columns=feature_set))
    shap_values = shap_explainer(df[feature_set].iloc[:1])  

    xtest_reverse=pf.revtransform_standardize_data(df[feature_set].iloc[:1],numeric_cols,scalar)
    predicted_classes = np.argmax(best_estimator.predict_proba(df[feature_set].iloc[:1]), axis=1)

    predicted_class = predicted_classes[0] 
    shap_exp=shap.Explanation(
        values=shap_values[0, :, predicted_class],  # Use the predicted class
        base_values=shap_values.base_values[0, predicted_class],  # Base value for predicted class
        feature_names=feature_set,
        data=xtest_reverse.iloc[0]  # Reverse-transformed values
    )

    col1, col2 = st.columns([1, 2])  # Adjust width if needed

    with col1:
        st.write("#### SHAP Interpretation")
        # for feature, value in zip(feature_set, shap_exp.values):
        #     st.write(f"**{feature}:** {value:.4f}")

    with col2:
        st.write("#### SHAP Waterfall Plot")

        fig, ax = plt.subplots(figsize=(8, 6))
        shap.waterfall_plot(shap_exp, max_display=10)  # Customize display
        st.pyplot(fig)
