#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/24 22:51
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   preparation.py
# @Desc     :   

from numpy import ceil
from streamlit import (empty, sidebar, subheader, session_state, button,
                       rerun, caption, slider, selectbox, spinner,
                       number_input, columns, markdown, image)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils.helper import Timer

empty_messages: empty = empty()
col_train, col_test = columns(2, gap="small")

pre_sessions: list[str] = ["pTimer", "train_datagen", "test_datagen", ]
for session in pre_sessions:
    session_state.setdefault(session, None)
load_sessions: list[str] = ["train_arr", "test_arr", "lTimer"]
for session in load_sessions:
    session_state.setdefault(session, None)

TRAIN_PATH: str = "data/train"
TEST_PATH: str = "data/test"

with sidebar:
    subheader("Data Preparation Settings")

    if session_state["train_datagen"] is None:
        empty_messages.error("Please click the button below to preprocess the data.")

        rotation_range: int = slider(
            "Rotation Range", min_value=0, max_value=360, value=20, step=1,
            help="Degree range for image rotations."
        )
        width_shift_range: float = slider(
            "Width Shift Range", min_value=0.0, max_value=1.0, value=0.02, step=0.01,
            help="Fraction of total width for horizontal shifts."
        )
        height_shift_range: float = slider(
            "Height Shift Range", min_value=0.0, max_value=1.0, value=0.02, step=0.01,
            help="Fraction of total height for vertical shifts."
        )
        shear_range: float = slider(
            "Shear Range", min_value=0.0, max_value=1.0, value=0.2, step=0.01,
            help="Shear intensity (shear angle in radians)."
        )
        zoom_range: float = slider(
            "Zoom Range", min_value=0.0, max_value=1.0, value=0.2, step=0.01,
            help="Range for random zoom."
        )
        horizontal_flip: bool = selectbox(
            "Horizontal Flip", options=[True, False], index=0,
            help="Randomly flip inputs horizontally."
        )
        fill_mode: str = selectbox(
            "Fill Mode", options=["nearest", "constant", "reflect", "wrap"], index=0,
            help="Points outside the boundaries are filled according to the given mode."
        )

        rescale = 1.0 / 255
        if button("Data Preprocessing", type="primary", width="stretch"):
            with spinner("Preprocessing data..."):
                with Timer("Data Preprocessing") as session_state["pTimer"]:
                    session_state["train_datagen"] = ImageDataGenerator(
                        rescale=rescale,
                        rotation_range=rotation_range,
                        width_shift_range=width_shift_range,
                        height_shift_range=height_shift_range,
                        shear_range=shear_range,
                        zoom_range=zoom_range,
                        horizontal_flip=horizontal_flip,
                        fill_mode=fill_mode
                    )
                    session_state["test_datagen"] = ImageDataGenerator(rescale=rescale)
            rerun()
    else:
        print(type(session_state["train_datagen"]), type(session_state["test_datagen"]))

        if session_state["train_arr"] is None:
            target_size: int = number_input(
                "Target Size", min_value=30, max_value=500, value=150, step=1,
                help="Dimensions to which all images found will be resized."
            )
            batch_size_load: int = number_input(
                "Batch Size for Load images", min_value=32, max_value=512, value=32, step=32,
                help="Number of samples per gradient update for training."
            )
            class_mode: str = selectbox(
                "Class Mode", options=["binary", "categorical", "sparse", "input"], index=0, disabled=True,
                help="Determines the type of label arrays that are returned."
            )
            caption(f"Cat and Dog dataset will be categorised into two classes with **{class_mode}**.")

            empty_messages.info(f"{session_state["pTimer"]} You can load the processed data now.")

            if button("Load Processed Data", type="primary", width="stretch"):
                with spinner("Loading processed data..."):
                    with Timer("Load Processed Data") as session_state["lTimer"]:
                        session_state["train_arr"] = session_state["train_datagen"].flow_from_directory(
                            TRAIN_PATH,
                            target_size=(target_size, target_size),
                            batch_size=batch_size_load,
                            class_mode=class_mode
                        )
                        session_state["test_arr"] = session_state["test_datagen"].flow_from_directory(
                            TEST_PATH,
                            target_size=(target_size, target_size),
                            batch_size=batch_size_load,
                            class_mode=class_mode
                        )
                rerun()
        else:
            empty_messages.info(f"{session_state["pTimer"]}. Data loaded successfully.")

            print(type(session_state["train_arr"]), type(session_state["test_arr"]))
            print(len(session_state["train_arr"]), len(session_state["test_arr"]))

            amount_batch_train: int = len(session_state["train_arr"]) - 1
            index_batch_train: int = number_input(
                "Select train batch index",
                min_value=0, max_value=amount_batch_train, value=0, step=1,
                help="Select which train batch to display."
            )
            caption(f"Note: the train values are in [0, {amount_batch_train}].")
            amount_batch_test: int = len(session_state["test_arr"]) - 1
            index_batch_test: int = number_input(
                "Select test batch index",
                min_value=0, max_value=amount_batch_test, value=0, step=1,
                help="Select which test batch to display."
            )
            caption(f"Note: the test values are in [0, {amount_batch_test}].")

            train_images, train_labels = session_state["train_arr"][index_batch_train]
            test_images, test_labels = session_state["test_arr"][index_batch_test]
            print(type(train_images), type(test_images))
            print(train_images.shape, test_images.shape)

            amount_image_train: int = len(train_images) - 1
            index_image_train: int = slider(
                "Select train sample index in batch",
                min_value=0, max_value=amount_image_train, value=0, step=1,
                help="Select an index to display a specific image in the train dataset."
            )
            caption(f"Note: the index is in [0, {amount_image_train}].")
            amount_image_test: int = len(test_images) - 1
            index_image_test: int = slider(
                "Select test sample index in batch",
                min_value=0, max_value=amount_image_test, value=0, step=1,
                help="Select an index to display a specific image in the test dataset."
            )
            caption(f"Note: the index is in [0, {amount_image_test}].")

            with col_train:
                markdown(f"#### Train Sample at Index {index_image_train} in Batch {index_batch_train}")
                image(
                    train_images[index_image_train],
                    caption=f"Label: **{'cat' if train_labels[index_image_train] == 0 else 'dog'}**",
                    width="stretch"
                )
            with col_test:
                markdown(f"#### Test Sample at Index {index_image_test} in Batch {index_batch_test}")
                image(
                    test_images[index_image_test],
                    caption=f"Label: **{'cat' if test_labels[index_image_test] == 0 else 'dog'}**",
                    width="stretch"
                )

            if button("Reset Data Generators", type="secondary", width="stretch"):
                for session in load_sessions:
                    session_state[session] = None
                rerun()

        if button("Reprocess Data", type="secondary", width="stretch"):
            for session in pre_sessions:
                session_state[session] = None
            rerun()
