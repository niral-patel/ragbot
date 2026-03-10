# src/exception.py : Define custom exceptions for the application that tells EXACTLY where code broke
# USAGE   : from src.exception import CustomException
#           raise CustomException(e, sys)

import sys
from src.logger import logger

def get_error_details(error, error_detail) -> str:
    '''
    Python's default error messages can be vague and unhelpful, especially when you're trying to debug a complex application.
    This function tells you WHERE it broke — file name + line number.
    This is super useful for debugging, especially in larger projects where you might have multiple files and functions.
    Get detailed information about the error, including the file name and line number where the error occurred.
    
    sys.exc_info() returns 3 things:
      [0] = exception TYPE  (e.g. FileNotFoundError)
      [1] = exception VALUE (e.g. "file.csv not found")
      [2] = traceback object ← THIS is what want

    The traceback object walks the call stack and gives us:
      tb_frame.f_code.co_filename = the exact .py file where error happened
      tb_lineno                   = the exact line number where error happened
      
    '''
    _, _, exc_tb = error_detail.exc_info()
    
    #-- Guard: what if there's no traceback info? (shouldn't happen, but just in case) 
    #This happens if get_error_details is called outside an except block ---
    if exc_tb is None:
        return f"Error occurred but no traceback info available. Error message: {str(error)}"
    
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    
    
    error_message = (
        f"\n  File  : {file_name}"
        f"\n  Line  : {line_number}"
        f"\n  Type  : {type(error).__name__}"
        f"\n  Error : {str(error)}"
    )
                     
    return error_message

class CustomException(Exception):
    '''
    Custom exception class that extends the built-in Exception class.

    HOW TO USE in ANY file across ALL projects:

        import sys
        from src.exception import CustomException
        from src.logger import logger

        try:
            result = risky_function()
        except Exception as e:
            raise CustomException(e, sys)

    WHAT YOU GET instead of a cryptic Python error:

        CustomException:
          File : E:/Projects/ragbot/src/components/data_ingestion.py
          Line : 47
          Type : FileNotFoundError
          Error : data/train.csv not found
    When you raise CustomException, it captures the error details and logs them using the logger.
    '''
    
    def __init__(self, error_message, error_detail):
        #-- call the parent constructor to initialize the Exception class with with original message
        super().__init__(error_message)
        
        # -- build the custom error message with file name and line number detailed message
        self.error_message = get_error_details(
            error_message,
            error_detail
        )
        # ── Log immediately when exception is raised ───────────────────────
        # Log the error message using the logger every error is AUTOMATICALLY logged when CustomException is raised, so you don't have to log it separately in the except block.
        logger.error(f"Exception raised: {self.error_message}")
        
    def __str__(self):
        # When you print the exception or convert it to a string ex. print(exception), 
        # it will return the detailed error message with file name and line number.
        return self.error_message