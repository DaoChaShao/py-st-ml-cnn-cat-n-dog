#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/24 22:52
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   test.py
# @Desc     :   

from os import path
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from streamlit import (empty, sidebar, subheader, session_state, button,
                       spinner, rerun, columns, metric, slider,
                       caption, image, markdown, number_input)
from tensorflow.keras.models import load_model

from subpages.train import MODEL_PATH
from utils.helper import Timer

empty_messages: empty = empty()
empty_samp_title: empty = empty()
col_img, col_num = columns(2, gap="small")
empty_result_title: empty = empty()
col_acc, col_pre, col_rec, col_auc, col_f1 = columns(5, gap="small")

pre_sessions: list[str] = ["train_datagen", "test_datagen"]
for session in pre_sessions:
    session_state.setdefault(session, None)
load_sessions: list[str] = ["train_arr", "test_arr"]
for session in load_sessions:
    session_state.setdefault(session, None)
test_sessions: list[str] = ["y_pred", "tTimer"]
for session in test_sessions:
    session_state.setdefault(session, None)

with sidebar:
    if session_state["train_datagen"] is None and session_state["test_datagen"] is None:
        empty_messages.error("Please load the data on the Home page first.")
    else:
        if session_state["train_arr"] is None and session_state["test_arr"] is None:
            empty_messages.error("Please preprocess the data on the Data Preparation page first.")
        else:
            if not path.exists(MODEL_PATH):
                empty_messages.error("Please train the model on the Model Training page and save it first.")
            else:
                subheader("Model Testing Settings")

                # Load the trained model
                model = load_model(MODEL_PATH)

                if session_state["y_pred"] is None:
                    empty_messages.info("You can now test the trained model with the test data.")

                    if button("Test the Model", type="primary", width="stretch"):
                        with spinner("Testing the model...", show_time=True, width="stretch"):
                            with Timer("Model Testing") as session_state["tTimer"]:
                                # Predict the labels for the test data
                                y_pred_prob = model.predict(session_state["test_arr"])
                                session_state["y_pred"] = (y_pred_prob > 0.5).astype("int32").flatten()
                        rerun()
                else:
                    empty_messages.info(
                        f"{session_state["tTimer"]} Model testing is complete."
                    )

                    y_true = session_state["test_arr"].labels
                    # y_true: list[int] = []
                    # for i in range(len(session_state["test_arr"])):
                    #     _, labels = session_state["test_arr"][i]
                    #     y_true.extend(labels)

                    empty_result_title.markdown("#### Test Results")
                    accuracy = accuracy_score(y_true, session_state["y_pred"])
                    precision = precision_score(y_true, session_state["y_pred"])
                    recall = recall_score(y_true, session_state["y_pred"])
                    auc = roc_auc_score(y_true, session_state["y_pred"])
                    f1 = f1_score(y_true, session_state["y_pred"])
                    with col_acc:
                        metric("Accuracy", f"{accuracy:.3%}", delta=None, delta_color="normal")
                    with col_pre:
                        metric("Precision", f"{precision:.3%}", delta=None, delta_color="normal")
                    with col_rec:
                        metric("Recall", f"{recall:.3%}", delta=None, delta_color="normal")
                    with col_auc:
                        metric("AUC", f"{auc:.4f}", delta=None, delta_color="normal")
                    with col_f1:
                        metric("F1 Score", f"{f1:.4f}", delta=None, delta_color="normal")

                    amount_batch: int = len(session_state["test_arr"]) - 1
                    index_batch: int = number_input(
                        "Select Test Batch Index",
                        min_value=0,
                        max_value=amount_batch,
                        value=0,
                        step=1,
                        help="Select an index to view the test image and its prediction.",
                    )
                    caption(f"Note: the test values are in [0, {amount_batch}].")

                    test_images, test_labels = session_state["test_arr"][index_batch]

                    amount_test: int = len(test_images) - 1
                    index_test = slider(
                        "Select Test Sample Index in Batch",
                        min_value=0,
                        max_value=amount_test,
                        value=0,
                        step=1,
                        help="Select an index to display a specific image in the test dataset.",
                    )
                    caption(f"Note: the index is in [0, {amount_test}].")
                    caption(f"Note: the image at test index **5** of batch **15** will surprise you.")

                    if button("Predict the Selected Sample", type="primary", width="stretch"):
                        with spinner("Predicting the selected sample", show_time=True, width="stretch"):
                            with Timer("Sample Prediction") as timer:
                                with col_img:
                                    empty_samp_title.markdown(
                                        f"### Test Sample at Index {index_test} of epoch {index_batch}"
                                    )
                                    image(
                                        test_images[index_test],
                                        caption=(
                                            f"True Label: **{'cat' if test_labels[index_test] == 0 else 'dog'}**"
                                        ),
                                        width="stretch"
                                    )
                                with col_num:
                                    print(type(test_images[index_test]), test_images[index_test].shape)
                                    single_image_for_test = test_images[index_test].reshape(
                                        1,
                                        *test_images[index_test].shape
                                    )
                                    pred_prob = model.predict(single_image_for_test)
                                    pred_label = (pred_prob > 0.5).astype("int32").flatten()[0]
                                    print(pred_label)

                                    markdown(
                                        f"<h1 style='font-size:300px; font-weight:bold; text-align:center;'>{pred_label}</h1>",
                                        unsafe_allow_html=True, width="stretch"
                                    )
