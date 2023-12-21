import streamlit as st
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import pandas as pd
with open('/Users/jordankanius/LHL_projects/Face2Face_Real_vs_Fake/models/Resnet_history.pkl', 'rb') as f:
    ResNet_history = pickle.load(f)
st.title('DeepFakeGuard: Real or Fake')



st.divider()

tab1, tab2, tab3, tab4, tab5= st.tabs(['Baseline','Xception', 'VGG16', 'ResNet50', 'Ensemble'])

with tab1:
    st.header("Baseline")
    col1,col2,col3 = st.columns(3)
    col1.metric(label="Accuracy", value="90")
    col2.metric(label='Precision',value='90')
    col3.metric(label='Recall', value='90')

with tab2:    
    st.header("Xception")
    col1,col2,col3 = st.columns(3)
    col1.metric(label="Accuracy", value="91")
    col2.metric(label='Precision',value='88')
    col3.metric(label='Recall', value='88')
    
with tab3:
    st.header("VGG16")
    col1,col2,col3 = st.columns(3)
    col1.metric(label="Accuracy", value="90")
    col2.metric(label='Precision',value='90')
    col3.metric(label='Recall', value='90')

with tab4:
    st.header('ResNet50')
    col1,col2,col3 = st.columns(3)
    col1.metric(label="Accuracy", value="66")
    col2.metric(label='Precision',value='66')
    col3.metric(label='Recall', value='68')
    
with tab5:
    st.header("Ensemble")
    col1,col2,col3 = st.columns(3)
    col1.metric(label="Accuracy", value="90")
    col2.metric(label='Precision',value='90')
    col3.metric(label='Recall', value='90')