#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/24 22:51
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   home.py
# @Desc     :   

from streamlit import title, expander, caption, empty

empty_messages = empty()
empty_messages.info("Please check the details at the different pages of core functions.")

title("Convolutional Neural Network (CNN) for Cat & Dog Classification")
with expander("**INTRODUCTION**", expanded=True):
    caption("This Streamlit app provides a complete workflow for training, testing, and visualizing a Convolutional Neural Network (CNN) for binary image classification (Cats vs Dogs).")
    caption("+ **Data Preparation**: Preprocess images with rotation, shift, shear, zoom, and flip augmentations.")
    caption("+ **Customizable Batch Loading**: Load images in batches and inspect individual samples.")
    caption("+ **CNN Model Training**: Train a simple but effective CNN with adjustable epochs.")
    caption("+ **Real-time Training Metrics**: Streamlit callbacks show live metrics such as loss and accuracy during training.")
    caption("+ **Model Testing and Evaluation**: Evaluate the trained model with accuracy, precision, recall, AUC, and F1 score.")
    caption("+ **Single Image Prediction**: Select a test image and predict its label instantly.")
    caption("+ **Save & Load Models**: Save trained models and reload them for inference.")

