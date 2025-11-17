import sys
from src.logger import logging

def error_message_detail(error, error_detail: sys):
    _,_,exc_tb = error_detail.exc_info() # Get the exception traceback information. The exc_info() function returns a tuple of three values: type, value, and traceback.And we want only Thrid value.
    file_name = exc_tb.tb_frame.f_code.co_filename # Get the filename from the traceback object.
    line_number = exc_tb.tb_lineno # Get the line number from the traceback object.
    error_message = f"Error occured in python script name [{file_name}] at line number [{line_number}] error message [{str(error)}]"
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        
        self.error_message = error_message_detail(error_message,error_detail = error_detail)
        super().__init__(self.error_message)

    def __str__(self):
        return self.error_message
    