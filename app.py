################ IMPORTS ######################
from flask import Flask, render_template, request, jsonify
from flask.helpers import url_for
import joblib
import nltk
import re
from celery import Celery
from flask_cors import CORS

# For sending GET requests from the API
import requests
# For saving access tokens and for file management when creating and adding to the dataset
import os
# For dealing with json responses we receive from the API
import json
# For displaying the data after
import pandas as pd
# For saving the response data in CSV format
import csv
# For parsing the dates received from twitter in readable formats
import datetime
import dateutil.parser
import unicodedata
#To add wait time between requests
import time
import math

def make_celery(app):
    os.environ['TOKEN'] = 'AAAAAAAAAAAAAAAAAAAAAFUPXwEAAAAAYyR3Kgg5btBKAgkBAAkyUBHDkQQ%3DexHE1M3ol9j9RhLBnlMGV2a5eksteqJ4EgFjmUbSYimdTFWHbt'
    broker = os.environ['REDIS_URL']
    backend = os.environ['REDIS_URL']
    name = os.environ.get('CELERY_NAME', 'default_name')

    celery = Celery(name, broker=broker,
                backend=backend)
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery

def auth():
    return os.getenv('TOKEN')

def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers    

def create_url(keyword, max_results = 10):
    
    search_url = "https://api.twitter.com/2/tweets/search/recent" #Change to the endpoint you want to collect data from

    #change params based on the endpoint you are using
    query_params = {'query': keyword,
                    'max_results': max_results,
                    'tweet.fields': 'id,text',
                    'place.fields': 'country',
                    'next_token': {}}
    return (search_url, query_params)

def connect_to_endpoint(url, headers, params, next_token = None):
    params['next_token'] = next_token   #params object received from create_url function
    response = requests.request("GET", url, headers = headers, params = params)
    print("Endpoint Response Code: " + str(response.status_code))
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()

nltk.download('stopwords')
nltk.download('rslp')
#Initialize the flask App
app = Flask(__name__)
CORS(app)
app.config['DEBUG'] = True
model = joblib.load('brimo_model.pkl')

celery = make_celery(app)

@app.route('/status/<task_id>')
def taskstatus(task_id):
    task = get_tweets.AsyncResult(task_id)
    if task.state == 'PENDING':
        # job did not start yet
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', '')
        }
        if 'result' in task.info:
            response['result'] = task.info['result']
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # this is the exception raised
        }
    return jsonify(response)
################# FLASK API ####################
#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

#To use the predict button in our web-app
@app.route('/classify',methods=['POST'])
def classify():
    request_data = request.get_json()
    task = get_tweets.apply_async([request_data['keyword'], request_data['language'], request_data['limit'], request_data['since'], request_data['until']])
    return jsonify({"task_id":task.id}), 202, {'Location': url_for('taskstatus',
                                                  task_id=task.id)}

############## GET TWEETS ################
@celery.task()
def get_tweets(keyword, langValue, limitValue, sinceValue, untilValue):
    bearer_token = auth()
    headers = create_headers(bearer_token)
    keyword = keyword + ' lang:pt -is:retweet -has:links -has:media'
    max_results = limitValue if (limitValue <= 100) else 100

    next_token = None
    loopLength = int(math.ceil(limitValue / 100))

    tweets = []
    tweets_for_classify = []

    for x in range(loopLength):
      url = create_url(keyword, max_results)
      json_response = connect_to_endpoint(url[0], headers, url[1], next_token)
      for tweetObj in json_response["data"]:
        tweets.append(tweetObj["text"])
      if 'next_token' in json_response['meta']:
        next_token = json_response['meta']['next_token']
        time.sleep(5)
      time.sleep(5)
    
    
    for tweet in tweets:
        tweet = re.sub(u'[^a-zA-Z0-9áéíóúÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇ: ]', '', tweet)
        tweet = ' '.join(word for word in tweet.split(' ') if not word.startswith('@'))
        tweets_for_classify.append(tweet)

    index = 0
    distribuicao_tristeza = 0
    distribuicao_alegria = 0
    distribuicao_medo = 0
    distribuicao_raiva = 0
    distribuicao_surpresa = 0
    distribuicao_nojo = 0
    for tweet in tweets_for_classify:
        index = index + 1
        tweet_without_special_chars = re.sub(u'[^a-zA-Z0-9áéíóúÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇ: ]', '', tweet)
        tweetStemming = []
        stemmer = nltk.stem.RSLPStemmer()
        for(palavras_treinamento) in tweet_without_special_chars.split():
            comStem = [p for p in palavras_treinamento.split()]
            tweetStemming.append(str(stemmer.stem(comStem[0])))
            
        novo = extrator_palavras(tweetStemming)

        distribuicao = model.prob_classify(novo)
        output = ""
        for classe in distribuicao.samples():
            if classe == "tristeza":
                distribuicao_tristeza = distribuicao_tristeza + distribuicao.prob(classe)
            elif classe == "alegria":
                distribuicao_alegria = distribuicao_alegria + distribuicao.prob(classe)
            elif classe == "medo":
                distribuicao_medo = distribuicao_medo + distribuicao.prob(classe)
            elif classe == "raiva":
                distribuicao_raiva = distribuicao_raiva + distribuicao.prob(classe)
            elif classe == "surpresa":
                distribuicao_surpresa = distribuicao_surpresa + distribuicao.prob(classe)
            else:
                distribuicao_nojo = distribuicao_nojo + distribuicao.prob(classe)
    
    if(index > 0):
      distribuicao_nojo = distribuicao_nojo / index
      distribuicao_raiva = distribuicao_raiva / index
      distribuicao_alegria = distribuicao_alegria / index
      distribuicao_tristeza = distribuicao_tristeza / index
      distribuicao_surpresa = distribuicao_surpresa / index
      distribuicao_medo = distribuicao_medo / index

    output = {"tristeza": distribuicao_tristeza,
      "nojo": distribuicao_nojo,
      "alegria": distribuicao_alegria,
      "surpresa": distribuicao_surpresa, 
      "medo": distribuicao_medo, 
      "raiva": distribuicao_raiva,
      "tweets": tweets_for_classify,
      "words": novo
    }
    return {'status': 'Tweets prontos para análise!',
            'result': output}
         
    
############## BRIMO #################
def extrator_palavras(documento):
    doc = set(documento)
    caracteristicas = {}
    for palavras in palavras_unicas_treinamento:
        caracteristicas['%s' % palavras] = (palavras in doc)
    return caracteristicas

def busca_palavras_unicas(frequencia):
    freq = frequencia.keys()
    return freq

def busca_frequencia(palavras):
    palavras = nltk.FreqDist(palavras)
    return palavras

def busca_Palavras(frases):
    todas_Palavras = []
    for(palavras, sentimento) in frases:
        todas_Palavras.extend(palavras)
    return todas_Palavras

def sentiment_Set(texto, words_emotions):
    sent_counter = {'raiva': 0, 'tristeza': 0, 'nojo': 0, 'surpresa': 0, 'alegria': 0, 'medo': 0}
    correct_sentiment_tweets = []
    new_texto = []
    for(palavras, sentimento) in texto:
        for palavra in palavras:
            for word in words_emotions:
                if palavra == word['word']:
                    sent_counter[word['emotion']]+= 1
        
        sentimento_new = max(sent_counter, key=sent_counter.get)
        if sent_counter[sentimento_new] > 0:
            if sent_counter[sentimento_new] > sent_counter[sentimento]:
                  sentimento = sentimento_new
        sent_counter = {'raiva': 0, 'tristeza': 0, 'nojo': 0, 'surpresa': 0, 'alegria': 0, 'medo': 0}
        new_texto.append((palavras, sentimento))
    return new_texto

def aplica_Stemmer(texto):
    stemmer = nltk.stem.RSLPStemmer()
    # Escolhido o RSLPS pois é especifico da lingua portugesa
    frases_sem_Stemming = []
    for(palavras, sentimento) in texto:
        com_Stemming = [str(stemmer.stem(p)) for p in palavras.split() if p not in lista_Stop]
        frases_sem_Stemming.append((com_Stemming, sentimento))
    return frases_sem_Stemming
###########################
emotions = ['raiva', 'nojo', 'alegria', 'medo', 'tristeza', 'surpresa'];
with open('words_emotions.json') as f:
    word_emotions = json.load(f)
        
stemmer = nltk.stem.RSLPStemmer()
        
format_word_emotions = []
# Create json object with tweet and sentiment
for word in word_emotions:
    for emotion in emotions:
        if word[emotion] == 1:
            com_Stem = str(stemmer.stem(word['word']))
            format_word_emotions.append({'word': com_Stem, 'emotion': emotion})
###########################
lista_Stop = nltk.corpus.stopwords.words('portuguese')

sentiments = ['alegria', 'medo', 'tristeza', 'nojo', 'surpresa', 'raiva']
base_treinamento = []

for sentiment in sentiments:
    inputjson_filename = '{sentiment_filename}.json'.format(sentiment_filename = sentiment)
    outputjson_filename = '{sentiment_filename}_output.json'.format(sentiment_filename = sentiment)
    
    # Deserialize file to load json in data
    with open(inputjson_filename) as f:
        data = json.load(f)

    # Filter python objects with list comprehensions
    output_dict = [x for x in data if x['language'] == 'pt']

    # Transform python object back into json
    output_json = json.dumps(output_dict, ensure_ascii=False)

    # Regex
    pattern = r'(?<="tweet": )"(.*?)"'

    # Get all regex matches
    tweets = re.findall(pattern, output_json)
    
    for tweet in tweets:
        tweet = re.sub(u'[^a-zA-Z0-9áéíóúÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇ: ]', '', tweet)
        tweet = ' '.join(word for word in tweet.split(' ') if not word.startswith('@'))
        base_treinamento.append((tweet, sentiment))

frases_com_Stem_treinamento = aplica_Stemmer(base_treinamento)
frases_com_Stem_e_sentimentos_treinamento = sentiment_Set(frases_com_Stem_treinamento, format_word_emotions)
palavras_treinamento = busca_Palavras(frases_com_Stem_e_sentimentos_treinamento)
frequencia_treinamento = busca_frequencia(palavras_treinamento)
palavras_unicas_treinamento = busca_palavras_unicas(frequencia_treinamento)

if __name__ == "__main__":
    app.run(debug=True)