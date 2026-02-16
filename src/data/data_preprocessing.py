import numpy as np 
import pandas as pd
import os
import re
import nltk  
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging


# logging configuration
logger = logging.getLogger('data_preprocessing')
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('preprocessing_error.log')
file_handler.setLevel('ERROR')

formatter  = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Download requirements NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_comment(comment):
    """Apply preprocessing transformations to a single comment"""
    try:
        # 🔥 Handle NaN or non-string safely
        if not isinstance(comment, str):
            return ""

        # Convert to lowercase
        comment = comment.lower()

        # Remove leading/trailing whitespace
        comment = comment.strip()

        # Replace newline characters with space
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters except punctuation
        comment = re.sub( r'[^a-zA-Z0-9\s' + re.escape(string.punctuation) + ']','',comment)

        # Remove stopwords (retain sentiment-important words)
        comment = ' '.join(word for word in comment.split()if word not in stop_words)

        # Lemmatize words
        comment = ' '.join(
            lemmatizer.lemmatize(word)
            for word in comment.split()
        )

        return comment

    except Exception as e:
        logger.error('Error preprocessing comment: %s', str(e))
        return ""

def normalize_text(df):
    ##apply the preprocessing steps to the text data in the dataframe
    try:
        df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)
        logger.debug('Text normalization completed successfully')
        return df
    except Exception as e:
        logger.error('Error during text normalization: %s', str(e))
        raise
    
def save_data(train_data:pd.DataFrame, test_data:pd.DataFrame, data_path:str) -> None:              
    """Save the preprocessed training and testing data to the specified output directory.   """
    try:
        interim_data_path  = os.path.join(data_path, 'interim')             
        logger.debug(f"create directory {interim_data_path}")                    
        os.makedirs(interim_data_path, exist_ok=True)   # ensure the directory is created
        
        logger.debug(f"directory {interim_data_path} created or already exists")
        
        train_data.to_csv(os.path.join(interim_data_path, 'train_processed.csv'), index=False)
        test_data.to_csv(os.path.join(interim_data_path, 'test_processed.csv'), index=False)

        logger.debug(f"Processed data saved to {interim_data_path}")
    except Exception as e:  
        logger.error('Error saving preprocessed data: %s', str(e)) 
        raise
               

def main():
    try:
        logger.debug('Starting data preprocessing...' )
        
        
        # Fectch the data from the data/raw 
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded successfully from data/raw directory')        
        
        # Preprocess the data
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)
        
        # Save the preprocessed data to the data/processed directory
        save_data(train_processed_data,test_processed_data, './data')
    except Exception as e:  
        logger.error('Error during data preprocessing: %s', str(e))
        print(f"Error during data preprocessing: {str(e)}" )
        
if __name__=="__main__":
    main()