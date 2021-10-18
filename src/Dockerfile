FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
RUN pip install tensorboard
COPY classifier.py /app/classifier.py
WORKDIR /app
CMD ["sh", "-c", "umask 0002; python classifier.py \
--model_type camembert \
--pretrained_name camembert-base \
--num_train_epochs 3 \
--max_seq_length 256 \
--eval_all_checkpoints"]