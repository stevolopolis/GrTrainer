"""
This file contains the Path class that creates, deletes, or
modifies paths. Concerned paths may related for saving
traine models or saving images.

This file is Copyright (c) 2022 Steven Tin Sui Luo.
"""

import os

from parameters import Params

params = Params()

class Path:
    """This class prepares the directories for saving AM results
    in the <vis> folder.
    """
    def __init__(self):
        self.model_path = params.MODEL_PATH
        self.log_path = params.LOG_PATH

    def create_model_path(self):
        """This method creates a subdirectory for trained models."""
        if self.model_path not in os.listdir('.'):
            os.makedirs(self.model_path)

    def create_log_path(self):
        """This method creates a subdirectory for saving training logs."""
        if self.log_path not in os.listdir('.'):
            os.makedirs(self.log_path)
    