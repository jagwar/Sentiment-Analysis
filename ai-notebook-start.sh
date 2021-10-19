    ovhai notebook run one-for-all jupyterlab \
	--name Sentiment-fr-volume \
	--framework-version v98-ovh.beta.1 \
	--flavor ai1-1-gpu \
	--gpu 1 \
	--volume sentiment-fr@GRA/:/app/data/sentiment_sentences:RW \
	--volume sentiment_tmp@GRA/:/app/runs:RW \
	--volume sentiment_output@GRA/:/app/data/output:RW
