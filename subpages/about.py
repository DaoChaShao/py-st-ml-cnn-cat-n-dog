#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/24 22:51
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   about.py
# @Desc     :   

from streamlit import title, expander, caption

title("**Application Information**")
with expander("About this application", expanded=True):
    caption("This application demonstrates end-to-end deep learning workflow using TensorFlow and Streamlit:")
    caption("- **Data Augmentation**: Uses Keras `ImageDataGenerator` for preprocessing with adjustable transformations.")
    caption("- **Batch Navigation**: Explore individual images and their labels before training or testing.")
    caption("- **CNN Architecture**: Sequential model with multiple Conv2D, MaxPooling2D, Flatten, and Dense layers.")
    caption("- **Training Logger**: Custom Keras callback updates Streamlit metrics placeholders after each epoch.")
    caption("- **Prediction Interface**: Interactive selection of test images for on-demand predictions.")
    caption("- **Evaluation Metrics**: Comprehensive evaluation with binary classification metrics.")
    caption("- **Streamlit Integration**: Fully interactive app with sidebar settings, sliders, and buttons for smooth workflow.")
