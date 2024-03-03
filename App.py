import streamlit as st
from ultralytics import YOLO

@st.cache_resource
def get_model():
    model = YOLO("yolov8n")
    return model

model = get_model()

st.title("YOLO model Basic")

uploaded_file = st.file_uploader("Choose a file", accept_multiple_files=False, type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    st.image(uploaded_file)
    res = model.predict(image, conf=conf)
    res_plotted = res[0].plot()
    st.image(res_plotted)