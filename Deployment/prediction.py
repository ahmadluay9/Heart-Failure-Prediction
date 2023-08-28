import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
# Load All Files

with open('rf_gridcv_best.pkl', 'rb') as file_1:
  rf_gridcv_best = pickle.load(file_1)

with open('Drop_Columns.txt', 'r') as file_2:
  Drop_Columns = json.load(file_2)

def run():
  with st.form(key='form_fifa_2022'):
      age = st.number_input('age',min_value=0,max_value=99,value=67,step=1,help='Age of Patients')
      anemia = st.number_input('Have Anemia ? ',min_value=0, max_value=1,value=0,help='0 for No, 1 for Yes')
      creatinine_phosphokinase = st.number_input('Level of the CPK enzyme in the blood',min_value=0,max_value=9999,value=213)
      diabetes = st.number_input('Have Diabetes?',min_value=0,max_value=1,value=0,help='0 for No, 1 for Yes')
      ejection_fraction = st.number_input('Percentage of blood leaving the heart at each contraction (%)',min_value=0,max_value=100,value=38)
      high_blood_pressure = st.number_input('Have Hypertension?',min_value=0,max_value=1,value=0,help='0 for No, 1 for Yes')
      platelets = st.number_input('Platelets in the blood (kiloplatelets/mL)',min_value=0,max_value=999999,value=215000,help='in kiloplatelets/mL')
      serum_creatinine = st.number_input('Level of serum creatinine in the blood ',step=0.01,format="%.2f",min_value=0.00,max_value=10.00,value=1.20,help='in mg/dL')
      serum_sodium = st.number_input('Level of serum sodium in the blood',min_value=0,max_value=150,value=133,help='in mEq/dL')
      sex = st.number_input('Gender',min_value=0,max_value=1,value=0,help='(Female = 0, Male = 1)')
      smoking = st.number_input('Smoker or Not Smoker ?',min_value=0,max_value=1,value=0,help='(No= 0, Yes = 1)')
      time = st.number_input('Follow Up Period',min_value=0,max_value=285,value=245,help='in Days')

      
     
      submitted = st.form_submit_button('Is the patient still alive?')

  df_inf = {
      'age': age,
      'anaemia': anemia,
      'creatinine_phosphokinase': creatinine_phosphokinase,
      'diabetes': diabetes,
      'ejection_fraction': ejection_fraction,
      'high_blood_pressure': high_blood_pressure,
      'platelets': platelets,
      'serum_creatinine': serum_creatinine,
      'serum_sodium': serum_sodium,
      'sex': sex,
      'smoking': smoking,
      'time':time
  }
  df_inf = pd.DataFrame([df_inf])
  # Data Inference
  df_inf_copy = df_inf.copy()
  df_inf_copy

  # Removing unnecessary features
  df_inf_final = df_inf_copy.drop(Drop_Columns,axis=1).sort_index()
  df_inf_final
  
  st.dataframe(df_inf_final)

  if submitted:
      # Predict using RandomForest
      y_pred_inf = rf_gridcv_best.predict(df_inf_final)
      st.write('# Is the patient still alive ?')
      if y_pred_inf == 0:
         st.subheader('Still Alive (^o^)/ ')
      else:
         st.subheader('Died from heart failure (T_T)')

if __name__ == '__main__':
    run()