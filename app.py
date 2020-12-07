import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os



face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_face(img):
    
	img = np.array(img)
	face_img = img.copy()
	face_rects = face_cascade.detectMultiScale(face_img) 
    
	for (x,y,w,h) in face_rects: 
		cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 10) 
        
	return face_img
	


st.title("Face detection")

html_temp = """
<body style="background-color:red;">
<div style="background-color:teal ;padding:10px">
<h2 style="color:white;text-align:center;">Face Recognition WebApp</h2>
</div>
</body>
"""
st.markdown(html_temp, unsafe_allow_html=True)

image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
if image_file is not None:
	img = Image.open(image_file)
	st.text("Original Image")
	st.image(img, use_column_width=True)

if st.button("Compute"):
	result_img= detect_face(img)
	st.image(result_img, use_column_width=True)
	