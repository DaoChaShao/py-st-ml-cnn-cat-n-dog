#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/24 22:51
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   home.py
# @Desc     :   

from streamlit import title, expander, caption, empty

empty_message = empty()
empty_message.info("Please check the details at the different pages of core functions.")

title("Convolutional Neural Network (CNN) for Cat & Dog Classification")
with expander("**INTRODUCTION**", expanded=True):
    caption("+")
