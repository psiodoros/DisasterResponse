from ast import operator
from collections import Counter
import json
import plotly
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from pprint import pprint

from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """
    Tokenizes text data
    Args:
    text str: Messages as text data
    Returns:
    # clean_tokens list: Processed text after normalizing, tokenizing and lemmatizing
    words list: Processed text after normalizing, tokenizing and lemmatizing
    """
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    words = word_tokenize(text)
    
    # remove stop words
    stop_words = stopwords.words("english")
    words = [word for word in words if word not in stop_words]
    
    # extract root form of words
    words = [WordNetLemmatizer().lemmatize(word, pos='v') for word in words]

    return words

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('MessagesDisaster', engine)

# load model
model = joblib.load("models/model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    label_counts = df.iloc[:, 4:].sum()
    label_names = list(label_counts.index)

    categories = df.iloc[:,4:]
    avg_categories = categories.mean().sort_values(ascending=False)[1:11]
    cat_names = list(avg_categories.index)
    # create visuals
    # TODO: Below is an example - modify to create your own visuals

    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=label_names,
                    y=label_counts,
                )
            ],

            'layout': {

                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'categoryorder': 'array',
                    'categoryarray': [x for _, x in sorted(zip(label_counts, label_names))]
                },
            }
        },
        {
                'data': [
                    Bar(
                        x= cat_names,
                        y= avg_categories
                    )
                ],

                'layout': {
                    'title': 'Top Categories',
                    'yaxis': {
                        'title': 'Percent'
                    },
                    'xaxis': {
                        'title': 'Category'
                    }
                }
        },
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
    main()
