cd data/json_bundle_reviews
#gunzip -ck large-bundle-corona.json > reviews-180k-bundle-after-corona.json.gz
zip reviews-500k-bundle-20200409.zip large-bundle-clean.json 
zip reviews-500k-bundle-20200409-full-bundle.zip large-bundle.json 

cd ..
cd neural_network_config
zip model_and_tokenizer-500k.zip model.h5 model.json temp-model.h5 tokenizer.pickle 
