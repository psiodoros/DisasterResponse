import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re

import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,  GridSearchCV 
from sklearn.metrics import classification_report
import pickle
def load_data(database_filepath):
    """Load the filepath and return the data"""
    
    # create conn string
    name = 'sqlite:///{}'.format(database_filepath)
    
    # initialize sql engine
    engine = create_engine(name)
    
    # load table to dataframe
    df = pd.read_sql_table('MessagesDisaster', con=engine) 
    #print(df.head())
    
    # Given value 2 in the related field are neglible so it could be error. Replacing 2 with 1 to consider it a valid response.
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)
    
    #Remove child alone as it has all zeros only
    df = df.drop(['child_alone'],axis=1)

    # define features and label arrays
    X = df['message']
    Y =  df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """tokenize and transform input text. Return cleaned text"""
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
   
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # Remove stop words
    stop = stopwords.words("english")
    words = [t for t in tokens if t not in stop]
    
    # lemmatize 
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in words:
        clean_tok = re.sub(r"^\s+|\s+$", "",lemmatizer.lemmatize(tok).lower())
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """Return Grid Search model with pipeline and Classifier"""
    
    # text processing and model pipeline
    # Pipeline: Random Forest Classifier
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',  MultiOutputClassifier(RandomForestClassifier()))
    ])
    # create gridsearch object and return as final model pipeline
    parameters = {'clf__estimator__max_depth': [1, 5, None],
              'clf__estimator__min_samples_leaf':[1, 2, 3]}

    # return model_pipeline
    cv = GridSearchCV(pipeline, parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """Print model results
    INPUT
    model -- required, estimator-object
    X_test -- required
    y_test -- required
    category_names = required, list of category strings
    OUTPUT
    None
    """
    # Get results and add them to a dataframe.
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))
    results = pd.DataFrame(columns=['Category', 'f_score', 'precision', 'recall'])


def save_model(model, model_filepath):
    """Save model as pickle file"""
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train.values, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()