import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model 
model = load_model("C:/Users/Connect/Desktop/animal_classifier_app/animal_classifier_kaggle.h5")
class_names = [ 'Buffalo', 'Elephant', 'Rihno','Zebra'] 
#titel
st.title('Animal Classification')
st.write('Please upload your animal')
#upload an image
uploaded_image=st.file_uploader("upload your image",type=['jpg','png','jpeg'])
if uploaded_image is not None:
    img = Image.open(uploaded_image)
    st.image(img,caption='Uploaded Image..',use_column_width=True)
    # convert image to array
    img_array = np.array(img)
    #convert image to gray scale
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)
    elif img_array.shape[2] == 1:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    #resize image to 128*128
    resized_img = cv2.resize(img_array, (128, 128))
    #NORMALIZE IMAGE
    normalized_img = resized_img.astype('float32') / 255.0
    # reshape image to match the model input shape
    input_img = np.expand_dims(normalized_img, axis=0) 
    #make prediction
    prediction = model.predict(input_img)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"The predicted animal is: *{predicted_class}*")

