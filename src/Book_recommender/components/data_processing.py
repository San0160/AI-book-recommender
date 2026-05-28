import re
import pandas as pd
import os


class DataProcessing:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def clean_genre(genre_str):
        """ Get the top 1 genre for the book or
        Blank for the null values
        """
        if pd.isna(genre_str) or genre_str in ("Unknown", "Error", ""):
            return ""  # empty string, not "Unknown"
        
        first = str(genre_str).split(",")[0].strip().title()
        return first
    
    @staticmethod
    def clean_text(text):
        '''
        Removes Punctuation
        '''
        text = re.sub(r'[^a-zA-Z0-9\s]', '', str(text))

        return text
           
    def process(self, data):
        # Clean genre
        data["genre_clean"] = data["genre"].apply(DataProcessing.clean_genre)

        # Drop unnecessary column
        data.drop(["bookID",
                   "isbn",
                   "text_reviews_count",
                   "publication_date",
                   "publisher",
                   "Unnamed: 12",
                   "genre"],
                   axis = 1,
                   inplace = True)
        
        # create combined features  
        data["combined_features"] = (
            data["title"].fillna('') + " " +
            data["authors"].fillna('') + " " +
            data["genre_clean"].fillna('') + " " +
            data["description"].fillna(''))
        
        data["average_rating"] = pd.to_numeric(
            data["average_rating"],
            errors="coerce")
        
        # Lowercaseing       
        data["combined_features"] = data["combined_features"].str.lower()

        # Remove punctuation
        data["combined_features"] = data["combined_features"].apply(DataProcessing.clean_text)

        # Save to disk
        data.to_csv(self.config.local_data_file, index=False)

        return data