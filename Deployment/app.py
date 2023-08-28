import streamlit as st
import eda # python file
import prediction # python file

navigation = st.sidebar.selectbox('Page Navigation: ',('EDA','Predict Patients Conditions'))

if navigation == 'EDA':
    eda.run()
else:
    prediction.run()

