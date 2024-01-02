import streamlit as st
import tensorflow as tf
import keras
from keras.models import load_model
import pickle
import pandas as pd
st.title('DeepFakeGuard: Real or Fake')
with open("Face2Face_Real_vs_Fake/models/Xception.pkl", 'rb') as f:
            Xception = pickle.load(f)

st.sidebar.file_uploader("Choose a file to upload")


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['Baseline','Xception', 'VGG16', 'ResNet50', 'Custom','Ensemble'])


with tab1:
    st.header("Baseline")
    st.write("Baseline metrics have been established using the dataset's 50/50 distribution of \
    class labels.", "")
    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    col1,col2,col3,col4 = st.columns(4)
    col1.metric(label="Accuracy", value=50, delta=f"{50-50}(*)")
    col2.metric(label="Precision",value="50", delta=f"{50-50}(*)")
    col3.metric(label="Recall", value="50", delta=f"{50-50}(*)")
    col4.metric(label="F-1 Score", value="50", delta=f"{50-50}(*)")
    st.caption('*=compared to _Baseline_')
    st.divider()


with tab2:
    st.header("Xception")
    st.write('Xception is a deep learning convolutional neural network (CNN), and is known for its efficiency in terms of parameters and computational cost.')
    st.divider()
    col1,col2,col3,col4 = st.columns(4)
    col1.metric(label="Accuracy", value=91, delta=f'{91-50}*')
    col2.metric(label="Precision",value=88, delta=f'{88-50}*')
    col3.metric(label="Recall", value=88, delta=f'{88-50}*')
    col4.metric(label="F-1 Score", value=88, delta=f'{88-50}*')
    st.caption('*=compared to _Baseline_')
    st.divider()
    st.image("/confusion_tables/confusion_Xception.png")
    st.divider()


with tab3:
    st.header("VGG16")
    st.write("VGG16 is a deep learning convolutional neural network (CNN), and is known for it's simplicity and is often used as a pre-trained model for the purpose of transfer learning")
    st.divider()
    col1,col2,col3,col4 = st.columns(4)
    col1.metric(label="Accuracy", value=83,delta=f"{83-50}(*)")
    col2.metric(label="Precision",value=84, delta=f'{84-50}(*)')
    col3.metric(label="Recall", value=84, delta=f'{84-50}(*)')
    col4.metric(label="F-1 Score", value=84, delta=f'{84-50}(*)')
    st.caption('*=compared to _Baseline_')
    st.divider()
    st.image('/Users/jordankanius/LHL_projects/Face2Face_Real_vs_Fake/confusion_VGG16.png')
    st.divider()


with tab4:
    st.header('ResNet50')
    st.write("ResNet50 is a deep convolutional neural network architecture consisting of 50 layers,\
     known for introducing residual connections that mitigate the vanishing gradient problem")
    st.divider()
    col1,col2,col3,col4 = st.columns(4)
    col1.metric(label="Accuracy", value=70, delta=f"{70-50}(*)")
    col2.metric(label="Precision",value=64, delta=f"{64-50}(*)")
    col3.metric(label="Recall", value=67, delta=f"{67-50}(*)")
    col4.metric(label="F-1 Score", value=66, delta=f"{66-50}(*)")
    st.caption("*=compared to _Baseline_")
    st.divider()
    st.image('/Users/jordankanius/LHL_projects/Face2Face_Real_vs_Fake/confusion_ResNet50.png')
    st.divider()

with tab5:
    st.header("Custom")
    st.write("This model is designed for binary classification with three convolutional \
    layers followed by max pooling layers, a flattening layer, and two fully connected (dense) \
    layers.")
    st.divider()
    col1,col2,col3,col4 = st.columns(4)
    col1.metric(label="Accuracy", value=95, delta =f'{95-50}(*)')
    col2.metric(label="Precision", value=96, delta=f'{96-50}(*)')
    col3.metric(label="Recall", value=94, delta=f'{94-50}(*)')
    col4.metric(label="F-1 Score", value=95, delta=f'{95-50}(*)')
    st.caption('*=compared to _Baseline_')
    st.divider()
    st.image('/Users/jordankanius/LHL_projects/Face2Face_Real_vs_Fake/confusion_Custom.png')
    st.divider()
    
with tab6:
    st.header("Ensemble")
    st.write("The ensemble method incorporates predictions from four distinct models: Xception, VGG16, ResNet50, and a Custom model architecture.")
    st.divider()
    col1,col2,col3,col4 = st.columns(4)
    col1.metric(label="Accuracy", value=95, delta=f'{95-50}(*)')
    col2.metric(label="Precision",value=96, delta=f'{95-50}(*)')
    col3.metric(label="Recall", value=96, delta=f'{96-50}(*)')
    col4.metric(label="F-1 Score", value=95, delta=f'{95-50}(*)')
    st.caption('*=compared to _Baseline_')
    st.divider()
    st.image('/Users/jordankanius/LHL_projects/Face2Face_Real_vs_Fake/confusion_Ensemble.png') 
    st.divider()