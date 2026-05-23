import sys


class CustomException(Exception):
    """
    Custom Exception class for better error tracking.
    """

    def __init__(self, error_message, error_detail: sys):
        """
        Constructor for CustomException.
        """
        self.error_message = self.error_message_detail(
            error_message,
            error_detail
        )

        super().__init__(self.error_message)

    @staticmethod
    def error_message_detail(error_message, error_detail: sys):
        """
        Static method to extract detailed error information.
        """
        _, _, exc_tb = error_detail.exc_info()

        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

        return (
            f"\nError occurred in script: [{file_name}] "
            f"at line number: [{line_number}] "
            f"error message: [{str(error_message)}]"
        )

    def __str__(self):
        """
        User-friendly string representation.
        """
        return self.error_message

    def __repr__(self):
        """
        Developer-friendly representation.
        """
        return (
            f"CustomException(error_message='{self.error_message}')"
        )