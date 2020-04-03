from pymongo import MongoClient
from random import randint
import datetime
import json
from bson import json_util
from bson.json_util import dumps
from bson.json_util import dumps, CANONICAL_JSON_OPTIONS
from dotenv import load_dotenv
import os

load_dotenv(verbose=True)

pathout = "../data/json_news_tagged/"
pathoutbundle = "../data/json_bundle_reviews/"

def createjson(item):
  out = dict()
  out["content"] = item["content"]
  out["date"] = item["date"]
  out["rate"] = item["rate"]
  out["rate_text"] = item["reate_text"]
  out["source"] = item["source"]
  out["title"] = item["title"]
  out["user"] = item["user"]
  return out

def save_news_into_json_bundle(news,count):
  filename = pathoutbundle + "large-bundle.json"
  out = []
  
    
  with open(filename, 'w',encoding="utf-8") as file:
    file.write('[')
    # Start from one as type_documents_count also starts from 1.
    for i, document in enumerate(news, 1):
        file.write(json.dumps(document, default=json_util.default, ensure_ascii=False))
        if i != count:
            file.write(',')
    file.write(']')

def main():
    MONGO_URL = os.getenv("database_url")

    client = MongoClient(MONGO_URL, unicode_decode_error_handler='ignore')

    db = client["reviews-scraping"]

    #query = {"date": {
    #    "$gte":   datetime.datetime.strptime("2018-11-01", '%Y-%m-%d'),
    #    "$lt":     datetime.datetime.strptime("2019-12-31", '%Y-%m-%d')}
    #}

    query = {"rate":{"$ne" : None}}
    print(query)

    news = db["Reviews"].find(query)
    count2 = db["Reviews"].find(query).count()

    print(count2)

    #save_news_into_jsons( count)
    save_news_into_json_bundle(news, count2)

if __name__ == "__main__":
    main()



#tar -cJf reviews-174k-bundle.tar.xz large-bundle.json
#tar -cJf reviews-180k-bundle-after-corona.tar.xz large-bundle-corona.json 

#extract tar -xf reviews-180k-bundle-after-corona.tar.xz
# mongoexport --uri="mongodb://admin:conectateSuperMongo_@104.248.33.254:27017"   --collection=Reviews  --out=reviews.json
# mongoexport --host="104.248.33.254" -u admin --port=27017 --collection=Reviews --db=reviews-scraping --out=events.json
