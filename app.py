################ IMPORTS ######################
from flask import Flask, request, jsonify
from flask.helpers import url_for
import re
from celery import Celery
from flask_cors import CORS

# For sending GET requests from the API
import requests
# For saving access tokens and for file management when creating and adding to the dataset
import os
#To add wait time between requests
import time
import math

import collections

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

def create_url(keyword, max_results, since, until):
    
    search_url = "https://api.twitter.com/2/tweets/search/recent" #Change to the endpoint you want to collect data from

    since = since + 'T00:00:00.000Z'
    until = until + 'T00:00:00.000Z'

    #change params based on the endpoint you are using
    query_params = {'query': keyword,
                    'max_results': max_results,
                    'end_time': until,
                    'start_time': since,
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

#Initialize the flask App
app = Flask(__name__)
CORS(app)
app.config['DEBUG'] = True

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
      url = create_url(keyword, max_results, sinceValue, untilValue)
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

    tweets_string = ""
    for tweet in tweets_for_classify:
        tweet_without_special_chars = re.sub(u'[^a-zA-Z0-9áéíóúÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇ: ]', '', tweet)
        tweets_string = tweets_string + tweet_without_special_chars + " "
        
    responseClassify = requests.post('https://brimo-r.herokuapp.com/classify', data={'tweets': tweets_string})
    print("Endpoint Response Code: " + str(responseClassify.status_code))
    if responseClassify.status_code != 200:
        raise Exception(responseClassify.status_code, responseClassify.text)
    
    responseCorpus = requests.post('https://brimo-r.herokuapp.com/corpus', data={'tweets': tweets_string})
    print("Endpoint Response Code: " + str(responseCorpus.status_code))
    if responseCorpus.status_code != 200:
        raise Exception(responseCorpus.status_code, responseCorpus.text)

    output = {'classify': responseClassify.json(), 'corpus': collections.Counter(responseCorpus.json()[0].split()).most_common(30), 'tweets': tweets_for_classify}
    return {'status': 'Tweets prontos para análise!',
            'result': output}

if __name__ == "__main__":
    app.run(debug=True)