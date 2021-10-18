import os
from datetime import datetime
import glob
import subprocess
import argparse
import logging
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, SequentialSampler
from tqdm import tqdm, trange
from transformers.data.processors import SingleSentenceClassificationProcessor
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    CamembertConfig,
    CamembertForSequenceClassification,
    CamembertTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

DATA_PATH = os.environ.get('DATA', '/app/data/sentiment_sentences')
OUTPUT_PATH = os.environ.get('OUTPUT', '/app/data/output')
LOG_DIR = os.environ.get('LOG_DIR', '/app/runs')

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
RUN_DIR = os.path.join(LOG_DIR, current_time)

WEIGHTS_NAME = "pytorch_model.bin"

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "camembert": (CamembertConfig, CamembertForSequenceClassification, CamembertTokenizer)
}

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (classes[0] for classes in MODEL_CLASSES.values())
    ),
    (),
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " +
        ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument(
        "--pretrained_name",
        default=None,
        type=str,
        required=True,
        help="Shortcut name of pretrained weights selected in the list: " +
        ", ".join(ALL_MODELS),
    )

    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )

    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--label_index",
        default=0,
        type=int,
        help="Index of label column in the data tsv file.",
    )

    parser.add_argument(
        "--text_index",
        default=1,
        type=int,
        help="Index of text/sentences column in the data tsv file.",
    )

    parser.add_argument(
        "--skip_first_row",
        action="store_true",
        help="Skip first row, set to True if the dataset has a header.",
    )

    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )

    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )

    args = parser.parse_args()

    model_type = args.model_type.lower()
    pretrained_name = args.pretrained_name
    num_train_epochs = args.num_train_epochs
    max_seq_length = args.max_seq_length
    label_index = args.label_index
    text_index = args.text_index
    do_lower_case = args.do_lower_case
    eval_all_checkpoints = args.eval_all_checkpoints
    skip_first_row = args.skip_first_row
    per_gpu_train_batch_size = args.per_gpu_train_batch_size
    per_gpu_eval_batch_size = args.per_gpu_eval_batch_size

    run(
        model_type=model_type,
        pretrained_name=pretrained_name,
        num_train_epochs=num_train_epochs,
        do_lower_case=do_lower_case,
        max_seq_length=max_seq_length,
        label_index=label_index,
        text_index=text_index,
        skip_first_row=skip_first_row,
        eval_all_checkpoints=eval_all_checkpoints,
        per_gpu_train_batch_size=per_gpu_train_batch_size,
        per_gpu_eval_batch_size=per_gpu_eval_batch_size,
        args=args
    )


def run(
    model_type="bert",
    pretrained_name="bert-base-uncased",
    num_train_epochs=3.0,
    do_lower_case=True,
    max_seq_length=128,
    label_index=0,
    text_index=1,
    skip_first_row=False,
    eval_all_checkpoints=False,
    per_gpu_train_batch_size=8,
    per_gpu_eval_batch_size=8,
    args=argparse.Namespace()
):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]

    model_name_or_path = pretrained_name

    # retrieve the latest checkpoint if it exists
    checkpoints = list(
        os.path.dirname(c) for c in glob.glob(OUTPUT_PATH + "/**/" + WEIGHTS_NAME, recursive=True)
    )
    if len(checkpoints) > 0:
        model_name_or_path = max(checkpoints, key=lambda x: x.split("-")[-1])

    # Init the tokenizer and load features
    tokenizer = tokenizer_class.from_pretrained(
        model_name_or_path,
        do_lower_case=do_lower_case
    )

    # Load training data, dataset is assumed to be named train.tsv
    train_dataset, labels_list = load_and_cache_examples(
        pretrained_name,
        tokenizer,
        max_seq_length,
        label_index,
        text_index,
        skip_first_row,
        evaluate=False
    )

    logger.info("Classification for %d labels (%s)",
                len(labels_list), labels_list)
    config = config_class.from_pretrained(
        model_name_or_path,
        num_labels=len(labels_list)
    )

    model = model_class.from_pretrained(
        model_name_or_path,
        config=config
    )

    # Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    n_gpu = torch.cuda.device_count()
    logger.info("Launch train with %d GPU and device %s", n_gpu, device)
    global_step, tr_loss = train(
        model_name_or_path,
        train_dataset,
        model,
        model_type,
        pretrained_name,
        tokenizer,
        num_train_epochs,
        max_seq_length,
        label_index,
        text_index,
        skip_first_row,
        per_gpu_train_batch_size=per_gpu_train_batch_size,
        per_gpu_eval_batch_size=per_gpu_eval_batch_size,
        n_gpu=n_gpu,
        device=device,
        args=args
    )

    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving

    # Create output directory if needed
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    logger.info("Saving model checkpoint to %s", OUTPUT_PATH)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(OUTPUT_PATH, "training_args.bin"))

    # Evaluation
    results = {}

    tokenizer = tokenizer_class.from_pretrained(
        OUTPUT_PATH, do_lower_case=do_lower_case)
    checkpoints = [OUTPUT_PATH]
    if eval_all_checkpoints:
        checkpoints = list(
            os.path.dirname(c) for c in sorted(glob.glob(OUTPUT_PATH + "/**/" + WEIGHTS_NAME, recursive=True))
        )
        logging.getLogger("transformers.modeling_utils").setLevel(
            logging.WARN)  # Reduce logging
    logger.info("Evaluate the following checkpoints: %s", checkpoints)
    for checkpoint in checkpoints:
        global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
        prefix = checkpoint.split(
            "/")[-1] if checkpoint.find("checkpoint") != -1 else ""

        model = model_class.from_pretrained(checkpoint)
        model.to(device)
        eval_dataset, _ = load_and_cache_examples(
            pretrained_name,
            tokenizer,
            max_seq_length,
            label_index,
            text_index,
            skip_first_row,
            evaluate=True
        )

        result = evaluate(eval_dataset, model, model_type, tokenizer,
                          n_gpu=n_gpu, device=device, prefix=prefix, args=args)
        result = dict((k + "_{}".format(global_step), v)
                      for k, v in result.items())
        results.update(result)

    return results


def set_seed(seed, n_gpu=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def load_and_cache_examples(pretrained_name, tokenizer, max_seq_length, label_index, text_index, skip_first_row, evaluate=False):
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        DATA_PATH,
        "cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
            pretrained_name,
            str(max_seq_length)
        ),
    )

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s",
                    cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", DATA_PATH)

        preprocessor = SingleSentenceClassificationProcessor.create_from_csv(
            os.path.join(DATA_PATH, "dev.tsv" if evaluate else "train.tsv"),
            column_label=label_index,
            column_text=text_index,
            skip_first_row=skip_first_row
        )

        preprocessor.labels = sorted(preprocessor.labels)

        features = preprocessor.get_features(
            tokenizer,
            max_length=max_seq_length
        )

        logger.info("Saving features into cached file %s",
                    cached_features_file)
        torch.save(features, cached_features_file)

    dataset, labels_list = tensors_to_dataset(features)
    return dataset, labels_list


def tensors_to_dataset(tensors, output_mode="classification"):
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(
        [f.input_ids for f in tensors], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in tensors], dtype=torch.long)

    # TODO should the token_type_ids be added ?
    # in the singlesentenceclassifier processor the tolen_types_ids are not extracted, why ?
    # all_token_type_ids = torch.tensor(
    #     [f.token_type_ids for f in tensors], dtype=torch.long)
    labels_list = []
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in tensors], dtype=torch.long)
        labels_list = list(all_labels.unique().numpy())
    elif output_mode == "regression":
        all_labels = torch.tensor(
            [f.label for f in tensors], dtype=torch.float)

    dataset = TensorDataset(
        # all_input_ids, all_attention_mask, all_token_type_ids, all_labels) cf TODO above
        all_input_ids, all_attention_mask, all_labels)
    return dataset, labels_list


def train(
    model_name_or_path,
    train_dataset,
    model,
    model_type,
    pretrained_name,
    tokenizer,
    num_train_epochs,
    max_seq_length,
    label_index,
    text_index,
    skip_first_row,
    n_gpu=0,
    device=torch.device("cpu"),
    per_gpu_train_batch_size=8,
    per_gpu_eval_batch_size=8,
    gradient_accumulation_steps=1,
    save_steps=500,
    logging_steps=500,
    max_grad_norm=1.0,
    weight_decay=0.0,
    learning_rate=2e-5,
    adam_epsilon=1e-8,
    warmup_steps=0,
    seed=42,
    args=argparse.Namespace()
):
    """ Train the model """
    tb_writer = SummaryWriter(log_dir=RUN_DIR)

    train_batch_size = per_gpu_train_batch_size * max(1, n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=train_batch_size)

    t_total = len(
        train_dataloader) // gradient_accumulation_steps * num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(
            os.path.join(model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(
            os.path.join(model_name_or_path, "scheduler.pt")))

    # multi-gpu training
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel & accumulation) = %d",
        train_batch_size * gradient_accumulation_steps
    )
    logger.info("  Gradient Accumulation steps = %d",
                gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) //
                                         gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (
            len(train_dataloader) // gradient_accumulation_steps)

        logger.info(
            "  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch",
                    steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(epochs_trained, int(
        num_train_epochs), desc="Epoch")

    set_seed(seed, n_gpu=n_gpu)  # Added here for reproductibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(device) for t in batch)
            # TODO cf todo in tensor_to_datasets (no token_type_ids)
            # inputs = {
            #     "input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            # if model_type != "distilbert":
            #     inputs["token_type_ids"] = (
            #         batch[2] if model_type in [
            #             "bert", "xlnet", "albert"] else None
            #     )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

            inputs = {
                "input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}

            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if logging_steps > 0 and global_step % logging_steps == 0:
                    logs = {}

                    # evaluate the current state
                    eval_dataset, _ = load_and_cache_examples(
                        pretrained_name,
                        tokenizer,
                        max_seq_length,
                        label_index,
                        text_index,
                        skip_first_row,
                        evaluate=True
                    )
                    results = evaluate(
                        eval_dataset,
                        model,
                        model_type,
                        tokenizer,
                        per_gpu_eval_batch_size=per_gpu_eval_batch_size,
                        n_gpu=n_gpu,
                        device=device,
                        args=args
                    )

                    for key, value in results.items():
                        eval_key = "eval_{}".format(key)
                        logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))

                if save_steps > 0 and global_step % save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(
                        OUTPUT_PATH, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(
                        output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(
                        output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(
                        output_dir, "scheduler.pt"))
                    logger.info(
                        "Saving optimizer and scheduler states to %s", output_dir)

    tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(
    eval_dataset,
    model,
    model_type,
    tokenizer,
    prefix="",
    n_gpu=0,
    device=torch.device("cpu"),
    per_gpu_eval_batch_size=8,
    args=argparse.Namespace()
):
    results = {}

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size
    )

    # multi-gpu eval
    if n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            # TODO cf todo in tensor_to_datasets (no token_type_ids)
            # inputs = {
            #     "input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            # if model_type != "distilbert":
            #     inputs["token_type_ids"] = (
            #         batch[2] if model_type in [
            #             "bert", "xlnet", "albert"] else None
            #     )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            inputs = {
                "input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}

            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)

    result = {"acc": (preds == out_label_ids).mean()}
    results.update(result)

    output_eval_file = os.path.join(
        OUTPUT_PATH, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def launch_tensorboard():
    ps = subprocess.Popen(['tensorboard', '--bind_all', '--logdir', RUN_DIR])
    return ps


if __name__ == "__main__":
    launch_tensorboard()
    main()
