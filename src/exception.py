import sys
from src.logger import logging

def error_message_details(error, error_details: sys):
    """
    This function takes an error and its details, and returns a formatted string
    with the error message and the line number where the error occurred.
    """
    _, _, exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in script: [{file_name}] at line number: [{line_number}] error message: [{str(error)}]"
    return error_message

class CustomException(Exception):
    """
    Custom exception class that inherits from the built-in Exception class.
    It takes an error and its details, and formats them into a string.
    """
    def __init__(self, error, error_details: sys):
        super().__init__(error)
        self.error_message = error_message_details(error, error_details)
    
    def __str__(self):
        return self.error_message