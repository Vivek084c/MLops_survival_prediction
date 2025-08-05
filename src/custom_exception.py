import traceback
import sys

class CustomException(Exception):
    
    def __init__(self, error_message, error_details: sys):
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(error_message, error_details)

    @staticmethod
    def get_detailed_error_message(error_message, error_details: sys):
        """
        Fucntion to show the detailed report of the error message
        """
        _, _, exc_traceback = traceback.sys.exc_info()
        filename = exc_traceback.tb_frame.f_code.co_filename
        linenumber = exc_traceback.tb_lineno

        return f"Error in : {filename} , Line No: {linenumber}"
    
    def __str__(self):
        return self.error_message
