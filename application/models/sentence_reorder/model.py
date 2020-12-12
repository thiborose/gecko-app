# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import glob
import logging
import os
import random
import csv
import codecs
import functools

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Dataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig,
                    BertForSequenceClassification, BertTokenizer)

from transformers import AdamW, get_linear_schedule_with_warmup

from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import DataProcessor, InputExample, InputFeatures

logger = logging.getLogger(__name__)

#ALL_MODELS = sum((tuple(
#    conf.pretrained_config_archive_map.keys()) for conf in (
#        BertConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
}


# GLOBAL VARS
data_dir = 'application/models/sentence_reorder/paragraph/'
output_dir = 'application/models/sentence_reorder/model/'

max_seq_length = 105
do_train = False
do_test = True
do_eval = False
evaluate_during_training = False
eval_all_checkpoints = False
do_lower_case = False
per_gpu_train_batch_size = 8
per_gpu_eval_batch_size = 8
gradient_accumulation_steps = 1
learning_rate = 5e-5
weight_decay = 0.0
adam_epsilon = 1e-8
max_grad_norm = 1.0
num_train_epochs = 3.0
max_steps = -1
warmup_steps = 0
logging_steps = 100
save_steps = 2000
no_cuda = False
overwrite_output_dir = True # was false
overwrite_cache = True # was false
seed = 42
fp16 = False
fp16_opt_level = '01'
local_rank = -1

# Model global variables
model = None
model_class = None



def set_seed():
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


class MyTestDataset(Dataset):
    """Dataset during test time"""

    def __init__(self, tensor_data, sents):
        assert len(tensor_data) == len(sents)
        self.tensor_data = tensor_data
        self.rows = sents
        
    def __len__(self):
        return len(self.tensor_data)

    def __getitem__(self, idx):
        return (self.tensor_data[idx], self.rows[idx])
    
def evaluate_test(model, tokenizer, prefix=""):
    
    if local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = PairProcessor()
    output_mode = "classification"
    
    cached_features_file = os.path.join(data_dir, 'cached_{}_{}_{}_{}'.format(
        'test',
        'bert',
        str(max_seq_length),
        'pair_order'))
    
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        lines = torch.load(cached_features_file + '_lines')
    else:
        logger.info("Creating features from dataset file at %s", data_dir)
        label_list = processor.get_labels()

        examples, lines = processor.get_test_examples(data_dir)
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=max_seq_length,
            output_mode=output_mode,
            pad_on_left=False,                 # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0,
        )
        if local_rank in [-1, 0]:
            logger.info(
                "Saving features into cached file %s", 
                cached_features_file)
            torch.save(features, cached_features_file)
            torch.save(lines, cached_features_file + '_lines')
        
    if local_rank == 0 and not evaluate:
        torch.distributed.barrier() 

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(
        all_input_ids, 
        all_attention_mask, 
        all_token_type_ids, 
        all_labels
        )

    
    eval_outputs_dirs = (output_dir,)
    file_h = codecs.open(data_dir + "test_results.tsv", "w", "utf-8")
    outF = csv.writer(file_h, delimiter='\t')

    results = {}
    for eval_output_dir in eval_outputs_dirs:
        eval_dataset = MyTestDataset(dataset, lines)
        
        if not os.path.exists(eval_output_dir) and local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        
        eval_dataloader = DataLoader(
            eval_dataset, 
            sampler=eval_sampler, 
            batch_size=eval_batch_size
            )

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
            row = batch[1]
            print(row)
            rows = {
                'guid': row[0],
                'text_a': row[1],
                'text_b': row[2],
                'labels': row[3],
                'pos_a': row[4],
                'pos_b':row[5]
            }
            del row
            batch = tuple(t.to(device) for t in batch[0])

            with torch.no_grad():
                inputs = {
                        'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2],
                        'labels':         batch[3]}

                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            logits = logits.detach().cpu().numpy()
            tmp_pred = np.argmax(logits, axis=1)
            for widx in range(logits.shape[0]):
                outF.writerow([rows['guid'][widx], rows['text_a'][widx], \
                rows['text_b'][widx], rows['labels'][widx], \
                rows['pos_a'][widx], rows['pos_b'][widx], \
                logits[widx][0], logits[widx][1], tmp_pred[widx]])
            if preds is None:
                preds = logits
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits, axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=1)

        result = compute_metrics("mnli", preds, out_label_ids)
        results.update(result)

        file_h.close()
        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results

def evaluate(model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    
    eval_outputs_dirs = (output_dir,)

    results = {}
    for eval_output_dir in eval_outputs_dirs:
        eval_dataset = load_and_cache_examples(tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

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
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'labels':         batch[3]
                        }
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=1)

        result = compute_metrics("mnli", preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def load_and_cache_examples(tokenizer, evaluate=False):
    if local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  

    processor = PairProcessor()
    output_mode = 'classification'
    
    cached_features_file = os.path.join(
        data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        'bert',
        str(max_seq_length),
        'pair_order'))

    if os.path.exists(cached_features_file):
        logger.info(
            "Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info(
            "Creating features from dataset file at %s", data_dir)
        label_list = processor.get_labels()
        examples = processor.get_dev_examples(data_dir) if evaluate else processor.get_train_examples(data_dir)
        features = convert_examples_to_features(examples,
                                    tokenizer,
                                    label_list=label_list,
                                    max_length=max_seq_length,
                                    output_mode=output_mode,
                                    pad_on_left=False,                 # pad on the left for xlnet
                                    pad_token=tokenizer.convert_tokens_to_ids(
                                        [tokenizer.pad_token])[0],
                                    pad_token_segment_id=0,
                                    )
        if local_rank in [-1, 0]:
            logger.info(
                "Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
        
    if local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in features], dtype=torch.long)

    all_labels = torch.tensor(
        [f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(
        all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset

class PairProcessor(DataProcessor):
    """Pair Processor for the pair ordering task."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
    
    def get_test_examples(self, data_dir):
        return self._create_test_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")
    
    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (_, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[1].lower()
                text_b = line[2].lower()
                label = line[3]
            except IndexError:
                print('cannot read the line: ' + line)
                continue
            examples.append(InputExample(
                                        guid=guid, 
                                        text_a=text_a, 
                                        text_b=text_b, 
                                        label=label
                                    ))
        return examples
    
    def _create_test_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples, rows = [], []
        for (_, line) in enumerate(lines):
            print(line)
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[1].lower()
                text_b = line[2].lower()
                label = line[3]
            except IndexError:
                print('cannot read the line: ' + line)
                continue
            examples.append(InputExample(
                                    guid=guid, 
                                    text_a=text_a, 
                                    text_b=text_b, 
                                    label=label
                            ))
            rows.append(line)
        return examples, rows


def load_model():
    global n_gpu

    global device

    global model

    global tokenizer

    global model_class

    if os.path.exists(output_dir) and os.listdir(output_dir) and do_train and not overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(output_dir))


    # Setup CUDA, GPU & distributed training
    if local_rank == -1 or no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.distributed.init_process_group(backend='nccl')
        n_gpu = 1
    device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    local_rank, device, n_gpu, bool(local_rank != -1), fp16)

    # Set seed
    output_mode = "classification"

    # Load pretrained model and tokenizer
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()  

    model_type = 'bert'
    _, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    
    tokenizer = tokenizer_class.from_pretrained('bert-base-uncased')
    model = model_class.from_pretrained('bert-base-uncased')

    if local_rank == 0:
        torch.distributed.barrier()  

    model.to(device)


def compute_probabilities():
    global model_class
    clean_cache()
    # Evaluation
    results = {}
    if (do_eval or do_test) and local_rank in [-1, 0]:
        checkpoints = [output_dir]
        if eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(device)
            if do_test:
                result = evaluate_test(model, tokenizer, prefix=global_step)
            elif do_eval:
                result = evaluate(model, tokenizer, prefix=global_step)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    return results

def clean_cache():
    os.system('rm application/models/sentence_reorder/paragraph/cached_test_bert_105_pair_order')
    os.system('rm application/models/sentence_reorder/paragraph/cached_test_bert_105_pair_order_lines')