ovhai job run \
-v sentiment_fr:/app/data/sentiment_sentences \
-v sentiment_fr_output:/app/data/output \
-v sentiment_fr_run:/app/runs \
--image gsalouovh/sentiment:main \
--gpu 4 --cpu 16 --mem 24 --name sentiment_fr_camembert_training
