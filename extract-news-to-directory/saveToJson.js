const MongoClient = require('mongodb').MongoClient;
require('dotenv').config()
const url = process.env["database_url"];
const fs = require("fs");

const pathout = "../data/json_news_tagged/"
const pathoutbundle = "../data/json_bundle_reviews/"

MongoClient.connect(url, function(err, db) {
  if (err) throw err;
  var dbo = db.db("reviews-scraping");
  var query = {};
  dbo.collection("Reviews").find(query).toArray(function(err, result) {
    if (err) throw err;
    console.log(url)
    let data = JSON.stringify(result);
    fs.writeFileSync(pathoutbundle + "large-bundle.json", data);
        
    db.close();
  });
});