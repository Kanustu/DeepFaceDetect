import streamlit as st
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import pandas as pd
st.title('DeepFakeGuard: Real or Fake')



st.divider()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['Baseline','Xception', 'VGG16', 'ResNet50', 'Custom','Ensemble'])

with tab1:
    st.header("Baseline")
    st.write("Baseline metrics have been established using the dataset's 50/50 distribution of class labels.")
    col1,col2,col3,col4 = st.columns(4)
    col1.metric(label="Accuracy", value="50")
    col2.metric(label="Precision",value="50")
    col3.metric(label="Recall", value="50")
    col4.metric(label="F-1 Score", value="50")


with tab2:    
    st.header("Xception")
    st.write('Xception is a deep learning convolutional neural network (CNN), and is known for its efficiency in terms of parameters and computational cost.')
    col1,col2,col3,col4 = st.columns(4)
    col1.metric(label="Accuracy", value=91, delta=91-50)
    col2.metric(label="Precision",value=88, delta=88-50)
    col3.metric(label="Recall", value=88, delta=88-50)
    col4.metric(label="F-1 Score", value=88, delta=88-50)
    
with tab3:
    st.header("VGG16")
    col1,col2,col3,col4 = st.columns(4)
    col1.metric(label="Accuracy", value=84,delta=f'{84-50}(*)')
    col2.metric(label="Precision",value=84, delta=f'{84-50}(*)')
    col3.metric(label="Recall", value=84, delta=f'{84-50}(*)')
    col4.metric(label="F-1 Score", value=84, delta=f'{84-50}(*)')
    st.caption('*=compared to _Baseline_')

with tab4:
    st.header('ResNet50')
    col1,col2,col3,col4 = st.columns(4)
    col1.metric(label="Accuracy", value="66")
    col2.metric(label="Precision",value="66")
    col3.metric(label="Recall", value="68")
    col4.metric(label="F-1 Score", value="68")

with tab5:
    st.header("Custom")
    col1,col2,col3,col4 = st.columns(4)
    col1.metric(label="Accuracy", value=95, delta =f'{95-50}(*)')
    col2.metric(label="Precision", value=96, delta=f'{86-50}(*)')
    col3.metric(label="Recall", value=94, delta=f'{94-50}(*)')
    col4.metric(label="F-1 Score", value=95, delta=f'{95-50}(*)')
    st.caption('*=compared to _Baseline_')
    st.image('/Users/jordankanius/LHL_projects/Face2Face_Real_vs_Fake/confusion_custom.png')

    
with tab6:
    st.header("Ensemble")
    col1,col2,col3,col4 = st.columns(4)
    col1.metric(label="Accuracy", value="90")
    col2.metric(label="Precision",value="90")
    col3.metric(label="Recall", value="90")
    col4.metric(label="F-1 Score", value="90")