const MongoClient = require('mongodb').MongoClient;
require('dotenv').config()
const url = process.env["database_url"];
const fs = require("fs");

MongoClient.connect(url, function(err, db) {
  if (err) throw err;
  var dbo = db.db("reviews-scaping");
  var query = {};
  dbo.collection("Reviews").find(query).toArray(function(err, result) {
    if (err) throw err;
    console.log(url)
    console.log(result);
    db.close();
  });
});