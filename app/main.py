"""
main.py

This script sets up a Gradio interface for recognizing handwritten digits using a pre-trained TensorFlow model.

Author: NhanPhamThanh-IT
Date: 28/07/2025
"""

# Importing necessary libraries
import gradio as gr
import tensorflow as tf
import cv2

# Importing system variables from the configs module
from configs import SystemVariables

# Load the trained model
model = tf.keras.models.load_model(SystemVariables.MODELS)

# Function to recognize digit from the image
def predict(img):
    imArray = (img['composite'])
    img = cv2.resize(imArray, (SystemVariables.IMG_SIZE, SystemVariables.IMG_SIZE))
    img = img.reshape(1, SystemVariables.IMG_SIZE, SystemVariables.IMG_SIZE, 1)

    preds = model.predict(img)[0]

    return {label: float(pred) for label, pred in zip(SystemVariables.LABELS, preds)}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Sketchpad(crop_size=(28,28), type='numpy', image_mode='L', brush=gr.Brush()), 
    outputs=gr.Label(num_top_classes=3), 
    title=SystemVariables.TITLE, 
    description=SystemVariables.HEAD
)

if __name__ == "__main__":
    # Launch the Gradio app
    demo.launch()

