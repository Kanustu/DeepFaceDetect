import streamlit as st

st.title('DeepFakeGuard: Real or Fake')
st.divider()

st.sidebar.title('Something')


st.divider()

tab1, tab2, tab3 = st.tabs(['Xception', 'VGG16', 'ResNet50'])

with tab1:
    st.header("Xception")
    st.metric(label="Temperature", value="70 °F", delta="1.2 °F")
    st.file_uploader('Upload your images')
with tab2:
    st.header("VGG16")
    st.metric(label="Temperature", value="70 °F", delta="1.2 °F")
    st.file_uploader('Upload your images')
with tab3:
    st.header("ResNet50")
    st.metric(label="Temperature", value="70 °F", delta="1.2 °F")
    st.file_uploader('Upload your images')