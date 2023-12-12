import streamlit as st

st.title('DeepFakeGuard: Real or Fake')


st.sidebar.title('Image Verification')
st.sidebar.file_uploader('Upload your images')

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
    col1.metric(label="Accuracy", value="90")
    col2.metric(label='Precision',value='90')
    col3.metric(label='Recall', value='90')
    
with tab3:
    st.header("VGG16")
    col1,col2,col3 = st.columns(3)
    col1.metric(label="Accuracy", value="90")
    col2.metric(label='Precision',value='90')
    col3.metric(label='Recall', value='90')

with tab4:
    st.header("ResNet50")
    col1,col2,col3 = st.columns(3)
    col1.metric(label="Accuracy", value="90")
    col2.metric(label='Precision',value='90')
    col3.metric(label='Recall', value='90')

with tab5:
    st.header("Ensemble")
    col1,col2,col3 = st.columns(3)
    col1.metric(label="Accuracy", value="90")
    col2.metric(label='Precision',value='90')
    col3.metric(label='Recall', value='90')