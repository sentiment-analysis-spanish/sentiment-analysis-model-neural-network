cd data/json_bundle_reviews
zip reviews-bundle.zip negative_reviews.json positive_reviews.json

cd ..
cd neural_network_config
zip model_and_tokenizer-1mil.zip model.h5 model.json temp-model.h5 tokenizer.pickle 
