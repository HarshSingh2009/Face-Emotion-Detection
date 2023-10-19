from PIL import Image
import tensorflow as tf
import streamlit as st
from pipeline import PredictionPipeline

st.title('Face Emotion Detection')
st.write('This Project is built using CNN (Convolutional Neural Networks) and help in classifying face images into Happy and Sad, they work with an accuracy of 100%')

st.write('')
st.write('')


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Process the uploaded image here
    with st.container():
        col1, col2 = st.columns([3, 2])
        col1.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        if col1.button('Predict Face Emotion'):
            pipeline = PredictionPipeline()
            y_pred, y_prob = pipeline.predict_emotion(uploaded_file)
            if y_pred[0][0] == 0:
                col2.balloons()
                col2.text('')
                col2.text('')
                col2.subheader('Happy Face 😇')
                acc = float("{:.2f}".format(100*(1 - y_prob[0][0])))
                col2.write(f'Accuracy: {acc}%')
            else:
                col2.snow()
                col2.text('')
                col2.text('')
                col2.subheader('Sad Face 😞') 
                acc = float("{:.2f}".format(100*(y_prob[0][0])))
                col2.write(f'Accuracy: {acc}%')
else:
    st.write("Please upload an image file (jpg, png, jpeg, or gif).")
