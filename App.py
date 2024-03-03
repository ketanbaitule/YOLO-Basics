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
    st.image(uploaded_file, caption="Original Image")
    img = Image.open(uploaded_file)
    res = model.predict(img)
    res_plotted = res[0].plot()[:, :, ::-1]
    st.image(res_plotted, "Result Image")