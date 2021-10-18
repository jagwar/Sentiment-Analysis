eai job submit \
--data $ORG_NAME.$ACCOUNT_NAME.sentiment_fr:/app/data/sentiment_sentences \
--data $ORG_NAME.$ACCOUNT_NAME.sentiment_fr_output:/app/data/output \
--data $ORG_NAME.$ACCOUNT_NAME.sentiment_fr_run:/app/runs \
--image registry.console.elementai.com/$ORG_NAME.$ACCOUNT_NAME/training-cam \
--gpu 4 --cpu 16 --mem 24 --name sentiment_fr_camembert_training