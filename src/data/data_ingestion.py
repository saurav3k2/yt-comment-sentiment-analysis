import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging  


# Logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('error.log')
file_handler.setLevel(logging.ERROR)

formatter  = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file at `params_path`."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)

        logger.debug('Parameters loaded successfully from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('Error parsing YAML file: %s', str(e))
        raise
def load_data(data_url:str)  -> pd.DataFrame:
    """Load data from the specified URL.   """
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded successfully from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Error parsing CSV from %s: %s', data_url, str(e))
        raise
    
def save_data(train_data:pd.DataFrame, test_data:pd.DataFrame, data_path:str) -> None:
    """Save the training and testing data to the specified output directory.   """
    try:
        raw_data_path = os.path.join(data_path, 'raw')                  
        
        # create the data/raw diectory if it doest not exist 
        os.makedirs(raw_data_path, exist_ok=True)       
         
        # save the train and test data to the data/raw directory
        train_data.to_csv(os.path.join(raw_data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path, 'test.csv'), index=False)
        
        logger.debug('train and test data saved to %s', raw_data_path)
        
    except Exception as e:
        logger.error('Unexpected error occured while saving the data: %s',  str(e))
        raise

class preprocess_data:
    def __init__(self, df):
        self.df = df

    def some_method(self, *args, **kwargs):
        raise NotImplementedError
    
        
def main():
    try:
        # load parameters from the params.yaml file in the root directory
        params = load_params(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'params.yaml'))
        test_size = params['data_ingestion']['test_size']
        
        # load the data from the specified URL
        df = load_data(data_url="https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/main/data/reddit.csv")
        
        # preprocess the data (currently no-op)
        final_df = df
        
        # split the data into training and testing sets
        train_data , test_data = train_test_split(final_df, test_size=test_size, random_state=42 )
        
        # save the split dataset and create the raw folder if it does not exist
        save_data(train_data , test_data, data_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')   )
    except Exception as e:  
        logger.error("Failed to complete data ingestion: %s", str(e) )
        print(f"Error: {str(e)}") 


if __name__ == "__main__":
    main()         
        
        
        