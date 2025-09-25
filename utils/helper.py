#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/24 22:50
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   helper.py
# @Desc     :   

from tensorflow.keras.callbacks import Callback
from time import perf_counter
from typing import override


class Timer(object):
    """ timing code blocks using a context manager """

    def __init__(self, description: str = None, precision: int = 5):
        """ Initialise the Timer class
        :param description: the description of a timer
        :param precision: the number of decimal places to round the elapsed time
        """
        self._description: str = description
        self._precision: int = precision
        self._start: float = 0.0
        self._end: float = 0.0
        self._elapsed: float = 0.0

    def __enter__(self):
        """ Start the timer """
        self._start = perf_counter()
        print("-" * 50)
        print(f"{self._description} has started.")
        print("-" * 50)
        return self

    def __exit__(self, *args):
        """ Stop the timer and calculate the elapsed time """
        self._end = perf_counter()
        self._elapsed = self._end - self._start

    def __repr__(self):
        """ Return a string representation of the timer """
        if self._elapsed != 0.0:
            # print("-" * 50)
            return f"{self._description} took {self._elapsed:.{self._precision}f} seconds."
        return f"{self._description} has NOT started."


class TFKerasLogger(Callback):
    """ Custom Keras Callback to log training metrics and update Streamlit placeholders.
    :param num_placeholders: a dictionary of Streamlit placeholders for metrics
    :return: None
    """

    def __init__(self, num_placeholders: dict = None):
        super().__init__()
        # The key name must match the callback logs
        self._history = {k: [] for k in [
            "loss", "accuracy", "precision", "recall", "auc",
            "val_loss", "val_accuracy", "val_precision", "val_recall", "val_auc"
        ]}
        self._placeholders = num_placeholders

    @override
    def on_epoch_end(self, epoch, logs=None):
        """ At the end of each epoch, log the metrics and update the placeholders.
        :param epoch: the current epoch number
        :param logs: the logs dictionary containing the metrics
        :return: None
        """
        logs = logs or {}
        # Save the training history per epoch
        for key in self._history.keys():
            self._history[key].append(logs.get(key, None))
        # Update the placeholders with the latest metrics
        if self._placeholders:
            for key, placeholder in self._placeholders.items():
                if key in logs and placeholder is not None:
                    placeholder.metric(
                        label=f"Epoch {epoch + 1}: {key.replace("val_", "valid ").capitalize()}",
                        value=f"{logs[key]:.4f}"
                    )

    def get_history(self):
        """ Get the training history."""
        return self._history
