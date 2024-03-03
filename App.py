import streamlit as st
from ultralytics import YOLO
from PIL import Image

@st.cache_resource
def get_model():
    model = YOLO("yolov8n")
    return model

model = get_model()

st.title("YOLO model Basic")

uploaded_file = st.file_uploader("Choose a file", accept_multiple_files=False, type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    res = model.predict(img)
    res_plotted = res[0].plot()[:, :, ::-1]

    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption="Original Image")
    
    with col2:
        st.image(res_plotted, "Result Image")