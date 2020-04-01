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
pathoutbundle = "../data/json_news_tagged_bundle/"

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

def save_news_into_json_bundle(news):
  filename = pathoutbundle + "large-bundle.json"
  out = []
  for item in news:
    item = createjson(item)
    out = out + [item]
    
  with open(filename, 'w',encoding="utf-8") as outfile:
    json.dump(out, outfile, ensure_ascii=False)

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

    count = db["Reviews"].find(query)
    count2 = db["Reviews"].find(query).count()

    print(count2)

    #save_news_into_jsons( count)
    save_news_into_json_bundle(count)

if __name__ == "__main__":
    main()



#tar -cJf reviews-174k-bundle.tar.xz large-bundle.json
#tar -cJf reviews-180k-bundle-after-corona.tar.xz large-bundle-corona.json 

#extract tar -xf reviews-180k-bundle-after-corona.tar.xz
