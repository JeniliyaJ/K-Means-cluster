import streamlit as st
import pandas as pd
import joblib
# load model
model=joblib.load('customer_segmentation_model.pkl')
scaler=joblib.load('scaler.pkl')
cluster_labels={0:'mediumincome,mediumspending',
                1:'highincome,highspending',
                2:'lowincome,highspending',
                3:'highincome,lowspending',
                4:'lowincome,lowspending'}
# stremlit UI
st.title('Customer Segmentation using k-means clustering')
st.markdown('This app uses k-means clustering to segment customers based on their income and spending score.')
# input from user
income=st.number_input('Annual Income (k$)',min_value=10,max_value=150,value=50)
spending_score=st.number_input('Spending Score (1-100)',min_value=1,max_value=100,value=50)
#predicted value
if st.button('predicte cluster'):
    new_data=pd.DataFrame({'Annual Income (k$)':[income],'Spending Score (1-100)':[spending_score]})
    new_data_scaled=scaler.transform(new_data)
    cluster=model.predict(new_data_scaled)[0]
    st.success(f'The customer belongs to cluster:{cluster}-{cluster_labels.get(cluster,"Unknown")}')
               


