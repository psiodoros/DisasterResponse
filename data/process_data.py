import sys
import pandas as pd
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Takes inputs two CSV files
    Imports them as pandas dataframes
    Merges them into a single dataframe
    
    Arguments:
        messages_filepath -> Path to the CSV file containing messages
        categories_filepath -> Path to the CSV file containing categories
    Output:
        df -> Combined data containing messages and categories
    """
    #load data from csv
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    #merge dataframes
    df = pd.merge(messages,categories,on='id')
    return df 


def clean_data(df):
    '''
    input:
        df: The oiginal dataframe.
    output:
        df: dataframe after cleaning.
    '''
    categories = df.categories.str.split(';', expand = True)
    
    # select the first row of the categories dataframe
    row = categories.loc[0]
    # rename the columns of `categories`
    category_colnames = [x[:-2] for x in row]
    categories.columns = category_colnames

    for column in categories:
    # set each value to be the last character of the string
        categories[column] = [x[-1:] for x in categories[column].astype(str)]
    # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # drop the original categories column from `df`
    df.drop(['categories'],axis=1,inplace=True)
    df = pd.concat([df,categories], axis =1 )
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df

def save_data(df, database_filename):
    #saving to database
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('MessagesDisaster', engine, if_exists='replace',index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python data\process_data.py '\
              'data\disaster_messages.csv data\disaster_categories.csv '\
              'data\DisasterResponse.db')


if __name__ == '__main__':
    main()