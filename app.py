from flask import Flask, jsonify, request
import experta
import csv
import numpy as np
import pandas as pd
import ast
from termcolor import colored
import nltk
import simple_colors
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)


@app.route('/', methods=['POST'])
def hello_world():
    print("asdkhsdkf")
    # put application's code here

    # nltk.download()
    filename = "C:\\Users\\lenovo\\Downloads\\Telegram Desktop\\ingredients - final (2).csv"

    arr = request.form.get('title')
    print(arr)
    # arr='Asparagus, Lox  Stilton Frittata#Tuna Tonnato with Eggplant Salad'
    arr = arr.split("#")
    print(len(arr))

    # reading csv file
    data = pd.read_csv(filename)
    data.dropna()
    d = data.drop(data.columns[[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22]], axis=1,
                  )

    def edit_text(sen):
        sen = sen.replace('[^\w\s]', '\n')
        sen = sen.replace('\d+', '')
        sen = sen.lower()
        return sen

    # data.sort_values(by=['col1'])
    df = data
    # print(type(data))
    df.loc[:, 'title'] = data.loc[:, 'title'].str.replace('[^\w\s]', '\n', regex=True)
    df.loc[:, 'dishTypes'] = data.loc[:, 'dishTypes'].str.replace('[^\w\s]', '\n', regex=True)
    df.loc[:, 'ingredients'] = data.loc[:, 'ingredients'].str.replace('[^\w\s]', '\n', regex=True)
    df.loc[:, 'title'] = data.loc[:, 'title'].str.replace('\d+', '', regex=True)
    df.loc[:, 'dishTypes'] = data.loc[:, 'dishTypes'].str.replace('\d+', '', regex=True)
    df.loc[:, 'ingredients'] = data.loc[:, 'ingredients'].str.replace('\d+', '', regex=True)
    df.loc[:, 'title'] = data.loc[:, 'title'].str.lower()
    df.loc[:, 'dishTypes'] = data.loc[:, 'dishTypes'].str.lower()
    df.loc[:, 'ingredients'] = data.loc[:, 'ingredients'].str.lower()
    porter = PorterStemmer()

    def stemSentence(sentence):
        token_words = word_tokenize(sentence)
        token_words
        stem_sentence = []
        for word in token_words:
            stem_sentence.append(porter.stem(word))
            stem_sentence.append(" ")
        return "".join(stem_sentence)

    for i in range(len(arr)):
        arr[i] = edit_text(arr[i])
        arr[i] = stemSentence(arr[i])

    # print(arr[0] + ' ' + arr[1])
    # data['dishTypes'] = data['dishTypes'].astype(str)
    for i in range(len(df)):
        df['title'].iloc[i] = stemSentence(data['title'].iloc[i])  # Stem every word.
        df['dishTypes'].iloc[i] = stemSentence(data['dishTypes'].iloc[i])
        df['ingredients'].iloc[i] = stemSentence(data['ingredients'].iloc[i])

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['ingredients'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    dead = []
    res = []
    res = []
    def get_recommendations(name, cosine_sim=cosine_sim):
        if name != []:
            for j in range(len(name)):
                print(name[j] + 'asdasxzxcwedewqwdqwdwwswsawd')
                for q in range(len(df)):
                    if df['title'][q].find(name[j]) != -1:
                        print('find the food and continue')
                        # Get the index of the food that matches the title
                        idx = df.index[df['title'].str.contains(name[j])].tolist()

                        # Get the pairwsie similarity scores of all dishes with that food
                        sim_scores = list(enumerate(cosine_sim[idx[0]]))
                        # print(cosine_sim)

                        # Sort the dishes based on the similarity scores
                        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

                        # Get the scores of the 10 most similar dishes
                        sim_scores = sim_scores[1:10]

                        # print(len(sim_scores))
                        # Get the food indices

                        # [dead.append(food_indices) for food_indices in dead if food_indices not in res]
                        food_indices = [i[0] for i in sim_scores]
                        dead.append(food_indices)
                        # dead = list(set(dead))
                        # print(dead)
                        res.append(d['title'].iloc[food_indices])
                        # food_indices = list(set(food_indices))
                        #                     return res

        else:
            return d['title']
        return res
    a = get_recommendations(arr)
    # print(a)
    deu = pd.DataFrame(a)
    deu.isnull()
    k = deu.columns
    k.tolist()
    len(k)
    l = d['title'].iloc[k]
    l = l.tolist()
    print(len(l))
    # l = l.to_json()
    json_file = {'title': l}
    print(json_file)
    return json_file


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8000')
