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

import warnings
warnings.filterwarnings('ignore')

def load_data(database_filepath):
    """
    Loads data from SQL Database
    Args:
    database_filepath: SQL database file
    Returns:
    X pandas_dataframe: Features dataframe
    Y pandas_dataframe: Target dataframe
    category_names list: Target labels 
    """
    
    # create conn string
    name = 'sqlite:///{}'.format(database_filepath)
    
    # initialize sql engine
    engine = create_engine(name)
    
    # load table to dataframe
    df = pd.read_sql_table('MessagesDisaster', con=engine) 
    #print(df.head())
    
    # Given value 2 in the related field are neglible so it could be error. Replacing 2 with 1 to consider it a valid response.
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)
    
    #Remove child alone as it has zeros only
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
    
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    words = word_tokenize(text)
    
    # remove stop words
    stopwords_ = stopwords.words("english")
    words = [word for word in words if word not in stopwords_]
    
    # extract root form of words
    words = [WordNetLemmatizer().lemmatize(word, pos='v') for word in words]

    return words


def build_model():
    """Return Grid Search model with pipeline and Classifier"""
    
    # text processing and model pipeline
    # Pipeline: Random Forest Classifier
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(
                            OneVsRestClassifier(LinearSVC())))])
    # create gridsearch object and return as final model pipeline
    parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
                  'vect__max_df': (0.75, 1.0)
                  }
    # return model_pipeline
    model = GridSearchCV(estimator=pipeline,
            param_grid=parameters,
            verbose=3,
            cv=3)
    return model


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
    # Get predictions
    y_pred = model.predict(X_test)
    # print classification report
    print(classification_report(Y_test.values, y_pred, target_names=category_names))

    # print accuracy score
    print('Accuracy: {}'.format(np.mean(Y_test.values == y_pred)))



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
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'models\\train_classifier.py '\
              'C:\\Users\\p.siontoros\\Documents\\PythonScripts\\Udacity\\DisasterResponse\\data\\DisasterResponse.db '\
              'models\\model.pkl')


if __name__ == '__main__':
    main()