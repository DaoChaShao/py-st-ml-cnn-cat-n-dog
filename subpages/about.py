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
    caption("-")
