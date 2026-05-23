from Book_recommender.logging.log import logging
from Book_recommender.exception.exception_handler import CustomException
import sys

logging.info("Starting the application")

try:
    a = 1/0

except Exception as e:
    raise CustomException(e, sys)