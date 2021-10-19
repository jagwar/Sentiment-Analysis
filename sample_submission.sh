ovhai job run \
-v sentiment_fr:/app/data/sentiment_sentences \
-v sentiment_fr_output:/app/data/output \
-v sentiment_fr_run:/app/runs \
--image gsalouovh/sentiment:main \
-g 1
