import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import sklearn
from sklearn.preprocessing import LabelEncoder

from PIL import Image

st.set_page_config(
    page_title='Heart Failure Prediction based on Ensamble Classifier: Random Forest Classifier and Adaptive Boost Classifier',
    layout = 'wide',
    initial_sidebar_state='expanded'
)

def run():
    # membuat title
    st.title('Heart Failure Prediction')

    # membuat sub header
    st.subheader ('Exploratory Data Analysis of the dataset.')

    # Menambahkan Gambar
    image = Image.open('heart_failure.jpg')
    st.image(image,caption = 'Heart Failure ilustration')

    # Menambahkan Deskripsi
    st.write('**What is heart failure ?**')
    st.write("Heart failure means that the heart is unable to pump blood around the body properly. It usually happens because the heart has become too weak or stiff. It's sometimes called congestive heart failure, although this name is not widely used now. Heart failure does not mean your heart has stopped working.")
    st.write('# Dataset')


    # show dataframe
    df = pd.read_csv('phase1_ftds_018_rmt.csv')
    st.dataframe(df)

    # DEATH_EVENT
    st.write('# Exploratory Data Analysis ')
    st.write('## Number of Death Event ')
    # death_event  value count
    death_event = df.DEATH_EVENT.value_counts().to_frame()
    death_event = death_event.reset_index()
    death_event['index'] = death_event['index'].replace({0:'No',1:'Yes'})
    death_event

    # Plot PieChart with Plotly
    fig = px.pie(death_event,values='DEATH_EVENT', names='index')
    # fig.update_traces(text = death_event['DEATH_EVENT'].value_counts(), textinfo = 'label+percent+value')
    fig.update_layout(title_text = "Death Event", title_x = 0.5)
    st.plotly_chart(fig)
    st.write('Number of deaths after the following days are different, where **Non-Death are 36% greater than Death**. This will be keep in mind if there is any imbalance data or not. But first, the death_event--as the target--will be compared with other variables so we can get the conclusion for the skewness and handling imbalance data.')

    # Gender Distribution
    st.write('### Gender Distribution ')
    # gender value count
    sex = df.groupby(by=['sex','DEATH_EVENT']).aggregate(Number_of_DEATH_EVENT=('DEATH_EVENT','count'))
    sex = sex.reset_index()
    sex['sex'] = sex['sex'].replace({0:'Female',1:'Male'})
    sex

    # plotting bar plot
    fig = px.bar(sex, x="sex", y="Number_of_DEATH_EVENT", color="DEATH_EVENT",
             orientation="v",hover_name="sex"        
                
             )
    fig.update_layout(title_text = "Gender Distribution", title_x = 0.5)
    st.plotly_chart(fig)
    st.write('From the visualization above, Male patients who have smoking habits have a higher chance of dying during follow up periods than any other conditions.')

    
    # Male Patients Condition
    st.write('## Male Patients Condition ')
    # male patients deceased during the follow up period 
    df_male = df.loc[(df['sex']==1)&
                    (df['DEATH_EVENT']==1)]
    df_male.head()

    # Create Bar Charts
    sns.set(font_scale=2)
    fig, ax = plt.subplots(1,4, sharex=True, figsize=(40,25))
    sns.countplot(ax=ax[0],x=df_male['anaemia'], palette='winter')
    ax[0].set_title('Male patients with anemia')
    sns.countplot(ax=ax[1],x=df_male['diabetes'], palette='winter')
    ax[1].set_title('Male patients with diabetes')
    sns.countplot(ax=ax[2],x=df_male['high_blood_pressure'], palette='winter')
    ax[2].set_title('Male patients with high blood pressure')
    sns.countplot(ax=ax[3],x=df_male['smoking'], palette='winter')
    ax[3].set_title('Male patients with habit of smoking')
    st.pyplot(fig)
    st.write('From the table and visualization above, it can be seen that the number of male patients with heart failure is more than female patients. **Where about 32% die during the follow-up period**. Further data exploration is necessary to find out the condition of male patients.')

    # Comparison between Death Event with other variables
    st.write('## Comparison between Death Event with other variables ')
    # Creating new dataframe for the histogram
    sns.set(font_scale=1)
    output = 'DEATH_EVENT'
    cols = [f for f in df.columns if df.dtypes[f] != "object"]
    f = pd.melt(df, id_vars=output, value_vars=cols)

    # Creating histogram
    g = sns.FacetGrid(f, hue=output, col="variable", col_wrap=4, sharex=False, sharey=False )
    g = g.map(sns.histplot, "value", kde=True).add_legend()
    st.pyplot(g)
    st.write('Based on the histogram above, we can see that the distribution of **Not Death** is still dominating that Death. However, we should check wherer variable time looks different than the others, where Death is high with time between 0-100 days. From here we should check the skewness of time as well.')

    # Using LabelEncoder to convert categorical into numerical data
    st.write('## Correlation Matrix Analysis')
    df_copy =df.copy()
    categorical = ['anaemia','diabetes','high_blood_pressure','sex','smoking','DEATH_EVENT']
    m_LabelEncoder = LabelEncoder()

    for col in df_copy[categorical]:
        df_copy[col]=m_LabelEncoder.fit_transform(df_copy[col])
    
    # Plotting Correlation Matrix of Features and DEATH_EVENT
    sns.set(font_scale=1)
    fig = plt.figure(figsize=(20,20))
    sns.heatmap(df_copy.corr(),annot=True,cmap='coolwarm', fmt='.2f')
    st.pyplot(fig)
    st.write('Based on visualization above, the `education_level`, `sex`, `marital_status` has a low correlation to the target (`DEATH_EVENT`).')

if __name__ == '__main__':
    run()