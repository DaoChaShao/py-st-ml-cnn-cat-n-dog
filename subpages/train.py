#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/24 22:52
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   train.py
# @Desc     :   

from os import path, remove
from streamlit import (empty, sidebar, subheader, session_state, button,
                       rerun, caption, spinner, columns, number_input)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

from utils.helper import Timer, StTFKLoggerForBinaryLabels

empty_messages: empty = empty()
empty_result_title: empty = empty()
col_los, col_acc = columns(2, gap="small")
col_los_valid, col_acc_valid = columns(2, gap="small")
placeholder_los = col_los.empty()
placeholder_acc = col_acc.empty()
placeholder_los_val = col_los_valid.empty()
placeholder_acc_val = col_acc_valid.empty()

pre_sessions: list[str] = ["train_datagen", "test_datagen"]
for session in pre_sessions:
    session_state.setdefault(session, None)
load_sessions: list[str] = ["train_arr", "test_arr", "target_size"]
for session in load_sessions:
    session_state.setdefault(session, None)
model_sessions: list[str] = ["model", "histories", "mTimer"]
for session in model_sessions:
    session_state.setdefault(session, None)

MODEL_PATH: str = "cnn_model.h5"

with sidebar:
    if session_state["train_datagen"] is None and session_state["test_datagen"] is None:
        empty_messages.error("Please preprocess the data in the 'Data Preparation' page first.")
    else:
        if session_state["train_arr"] is None and session_state["test_arr"] is None:
            empty_messages.error("Please load the preprocessed data in the 'Data Preparation' page first.")
        else:
            empty_messages.warning("You can now train the model with the loaded data.")
            subheader("Model Training Settings")

            # Initialize the metrics placeholders
            placeholders: dict = {
                "loss": placeholder_los,
                "accuracy": placeholder_acc,
                "val_loss": placeholder_los_val,
                "val_accuracy": placeholder_acc_val,
            }
            # Initialise the callback for visualisation
            callback = StTFKLoggerForBinaryLabels(placeholders)

            if session_state["model"] is None:
                empty_messages.info(f"Data is ready. You can start training the model.")

                epochs = number_input(
                    "Epochs",
                    min_value=1,
                    max_value=100,
                    value=25,
                    step=1,
                    help="Number of epochs to train the model.",
                )
                caption("Note: **25** batch size is recommended for **QUICK** training.")

                if button("Train the CNN Model", type="primary", width="stretch"):
                    with spinner("Training the CNN Model...", show_time=True, width="stretch"):
                        with Timer("CNN Model Training") as session_state["mTimer"]:
                            session_state["model"] = Sequential([
                                Input(shape=(session_state["target_size"], session_state["target_size"], 3,)),
                                # First Conv Layer: 150 * 150 * 32
                                Conv2D(32, (3, 3), strides=(1, 1), padding="same", activation="relu"),
                                # First Pooling Layer: 75 * 75 * 32
                                MaxPooling2D((2, 2), strides=(2, 2), padding="same"),
                                # Second Conv Layer: 75 * 75 * 64
                                Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation="relu"),
                                # Second Pooling Layer: 38 * 38 * 64
                                MaxPooling2D((2, 2), strides=(2, 2), padding="same"),
                                # Third Conv Layer: 38 * 38 * 128
                                Conv2D(128, (3, 3), strides=(1, 1), padding="same", activation="relu"),
                                # Third Pooling Layer: 19 * 19 * 128
                                MaxPooling2D((2, 2), strides=(2, 2), padding="same"),
                                Flatten(),
                                Dense(128, activation="relu"),
                                Dense(1, activation="sigmoid")
                            ])

                            session_state["model"].compile(
                                optimizer="adam",
                                loss="binary_crossentropy",
                                metrics=["accuracy"],
                            )
                            print(session_state["model"].summary())

                            session_state["model"].fit(
                                session_state["train_arr"],
                                epochs=epochs,
                                validation_data=session_state["test_arr"],
                                callbacks=[callback]
                            )

                            # Get the training history for storage
                            session_state["histories"] = callback.get_history()
                    rerun()
            else:
                hist = session_state["histories"]
                if hist:
                    last_epoch = len(hist["loss"])
                    for key, placeholder in placeholders.items():
                        if key in hist and placeholder is not None:
                            value = hist[key][-1]
                            label = f"Epoch {last_epoch}: {key.replace("val_", "Valid ").capitalize()}"
                            placeholder.metric(label=label, value=f"{value:.4f}")

                if not path.exists(MODEL_PATH):
                    empty_messages.info(
                        f"{session_state["mTimer"]}. Model trained successfully. You can now save the trained model."
                    )

                    if button("Save the Trained Model", type="primary", width="stretch"):
                        with spinner("Saving the trained model...", show_time=True, width="stretch"):
                            with Timer("Save the Trained Model...") as timer:
                                session_state["model"].save(MODEL_PATH)
                        empty_messages.success(f"{timer}. Model saved successfully as '{MODEL_PATH}'.")
                        rerun()
                else:
                    empty_messages.success(
                        f"{session_state['mTimer']}. Model is trained and saved as '{MODEL_PATH}'."
                    )
                    if button("Delete the Trained Model", type="secondary", width="stretch"):
                        with spinner("Deleting the trained model...", show_time=True, width="stretch"):
                            with Timer("Delete the Trained Model...") as timer:
                                remove(MODEL_PATH)
                                for session in model_sessions:
                                    session_state[session] = None
                        empty_messages.warning(f"{timer}. Model deleted successfully.")
                        rerun()
