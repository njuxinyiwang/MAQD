import argparse
import json
import logging
import os
import random
import shutil
import sys
import pdb
import apex
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from data_helper import BlkPosInterface, SimpleListDataset
import pickle
import torch.nn as nn
import torch.nn.functional as F


class ContextError(Exception):
    def __init__(self):
        pass


class Once:
    def __init__(self, rank):
        self.rank = rank

    def __enter__(self):
        if self.rank > 0:
            sys.settrace(lambda *args, **keys: None)
            frame = sys._getframe(1)
            frame.f_trace = self.trace
        return True

    def trace(self, frame, event, arg):
        raise ContextError

    def __exit__(self, type, value, traceback):
        if type == ContextError:
            return True
        else:
            return False


class OnceBarrier:
    def __init__(self, rank):
        self.rank = rank

    def __enter__(self):
        if self.rank > 0:
            sys.settrace(lambda *args, **keys: None)
            frame = sys._getframe(1)
            frame.f_trace = self.trace
        return True

    def trace(self, frame, event, arg):
        raise ContextError

    def __exit__(self, type, value, traceback):
        if self.rank >= 0:
            torch.distributed.barrier()
        if type == ContextError:
            return True
        else:
            return False


class Cache:
    def __init__(self, rank):
        self.rank = rank

    def __enter__(self):
        if self.rank not in [-1, 0]:
            torch.distributed.barrier()
        return True

    def __exit__(self, type, value, traceback):
        if self.rank == 0:
            torch.distributed.barrier()
        return False


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


class Prefetcher:
    def __init__(self, dataloader, stream):
        self.dataloader = dataloader
        self.stream = torch.cuda.Stream()

    def __iter__(self):
        self.iter = iter(self.dataloader)
        self.preload()
        return self

    def preload(self):
        try:
            self.next = next(self.iter)
        except StopIteration:
            self.next = None
            return
        with torch.cuda.stream(self.stream):
            next_list = list()
            for v in self.next:
                if type(v) == torch.Tensor:
                    next_list.append(v.cuda(non_blocking=True))
                else:
                    next_list.append(v)
            self.next = tuple(next_list)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        if self.next is not None:
            result = self.next
            self.preload()
            return result
        else:
            raise StopIteration

    def __len__(self):
        return len(self.dataloader)


class TrainerCallback:
    def __init__(self):
        pass

    def on_argument(self, parser):
        pass

    def load_model(self):
        pass

    def load_data(self):
        pass

    def collate_fn(self):
        return None, None, None

    def on_train_epoch_start(self, epoch):
        pass

    def on_train_step(self, step, train_step, inputs, extra, loss, outputs):
        pass

    def on_train_epoch_end(self, epoch):
        pass

    def on_dev_epoch_start(self, epoch):
        pass

    def on_dev_step(self, step, inputs, extra, outputs):
        pass

    def on_dev_epoch_end(self, epoch):
        pass

    def on_test_epoch_start(self, epoch):
        pass

    def on_test_step(self, step, inputs, extra, outputs):
        pass

    def on_test_epoch_end(self, epoch):
        pass

    def process_train_data(self, data):
        pass

    def process_dev_data(self, data):
        pass

    def process_test_data(self, data):
        pass

    def on_save(self, path):
        pass

    def on_load(self, path):
        pass


class Trainer:
    def __init__(self, callback: TrainerCallback):
        self.callback = callback
        self.callback.trainer = self
        logging.basicConfig(level=logging.INFO)

    def parse_args(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--train', action='store_true')
        self.parser.add_argument('--dev', action='store_true')
        self.parser.add_argument('--test', action='store_true')

        self.parser.add_argument('--debug', action='store_true')
        self.parser.add_argument("--per_gpu_train_batch_size", default=1, type=int)
        self.parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int)
        self.parser.add_argument("--learning_rate", default=3e-5, type=float)
        self.parser.add_argument("--selector_learning_rate", default=3e-4, type=float)
        self.parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
        self.parser.add_argument("--weight_decay", default=0.0, type=float)
        self.parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        self.parser.add_argument("--max_grad_norm", default=1.0, type=float)
        self.parser.add_argument("--epochs", default=10, type=int)
        self.parser.add_argument("--warmup_ratio", default=0.1, type=float)
        self.parser.add_argument("--logging_steps", type=int, default=500)
        self.parser.add_argument("--save_steps", type=int, default=1000000)
        self.parser.add_argument("--seed", type=int, default=42)
        self.parser.add_argument("--num_workers", type=int, default=0)
        self.parser.add_argument("--local_rank", type=int, default=-1)
        self.parser.add_argument("--fp16", action="store_true")
        self.parser.add_argument("--fp16_opt_level", type=str, default="O1")
        self.parser.add_argument("--no_cuda", action="store_true")
        self.parser.add_argument("--load_checkpoint", default=None, type=str)
        self.parser.add_argument("--ignore_progress", action='store_true')
        self.parser.add_argument("--dataset_ratio", type=float, default=1.0)
        self.parser.add_argument("--no_save", action="store_true")
        self.parser.add_argument("--intro_save", default="../data/", type=str)
        self.parser.add_argument("--num_sentences", default=15, type=int)
        self.parser.add_argument('--senemb_path', default='../../../data/senemb', type=str)
        self.parser.add_argument('--rerank_path', default='../../../data/new_rerank', type=str)
        self.parser.add_argument("--lam_notnone", default=10, type=float)
        self.parser.add_argument("--lam_none", default=1, type=float)
        self.parser.add_argument("--lam_rank", default=1, type=float)
        self.parser.add_argument("--reward_limit", default=9999, type=float)

        # self.parser.add_argument("--model_name", default="bert", type=str)
        self.callback.on_argument(self.parser)
        self.args = self.parser.parse_args()
        keys = list(self.args.__dict__.keys())
        for key in keys:
            value = getattr(self.args, key)
            if type(value) == str and os.path.exists(value):
                setattr(self.args, key, os.path.abspath(value))
        if not self.args.train:
            self.args.epochs = 1
        self.train = self.args.train
        self.dev = self.args.dev
        self.test = self.args.test
        self.debug = self.args.debug
        self.per_gpu_train_batch_size = self.args.per_gpu_train_batch_size
        self.per_gpu_eval_batch_size = self.args.per_gpu_eval_batch_size
        self.learning_rate = self.args.learning_rate
        self.selector_learning_rate = self.args.selector_learning_rate
        self.gradient_accumulation_steps = self.args.gradient_accumulation_steps
        self.weight_decay = self.args.weight_decay
        self.adam_epsilon = self.args.adam_epsilon
        self.max_grad_norm = self.args.max_grad_norm
        self.epochs = self.args.epochs
        self.warmup_ratio = self.args.warmup_ratio
        self.logging_steps = self.args.logging_steps
        self.save_steps = self.args.save_steps
        self.seed = self.args.seed
        self.num_workers = self.args.num_workers
        self.local_rank = self.args.local_rank
        self.fp16 = self.args.fp16
        self.fp16_opt_level = self.args.fp16_opt_level
        self.no_cuda = self.args.no_cuda
        self.load_checkpoint = self.args.load_checkpoint
        self.ignore_progress = self.args.ignore_progress
        self.dataset_ratio = self.args.dataset_ratio
        self.no_save = self.args.no_save
        self.callback.args = self.args
        self.model_name = self.args.model_name
        self.intro_save = self.args.intro_save

    def set_env(self):
        if self.debug:
            sys.excepthook = IPython.core.ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)
        if self.local_rank == -1 or self.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")
            self.n_gpu = 0 if self.no_cuda else torch.cuda.device_count()
        else:
            torch.cuda.set_device(self.local_rank)
            device = torch.device("cuda", self.local_rank)
            torch.distributed.init_process_group(backend="nccl")
            self.n_gpu = 1
        set_seed(self.seed, self.n_gpu)
        self.device = device
        with self.once_barrier():
            if not os.path.exists('r'):
                os.mkdir('r')
            runs = os.listdir('r')
            i = max([int(c) for c in runs], default=-1) + 1
            os.mkdir(os.path.join('r', str(i)))
            src_names = [source for source in os.listdir() if source.endswith('.py')]
            for src_name in src_names:
                shutil.copy(src_name, os.path.join('r', str(i), src_name))
            os.mkdir(os.path.join('r', str(i), 'output'))
            os.mkdir(os.path.join('r', str(i), 'tmp'))
        runs = os.listdir('r')
        i = max([int(c) for c in runs])
        os.chdir(os.path.join('r', str(i)))
        with self.once_barrier():
            json.dump(sys.argv, open('output/args.json', 'w'))
        logging.info("Process rank: {}, device: {}, n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            self.local_rank, device, self.n_gpu, bool(self.local_rank != -1), self.fp16))
        # self.train_batch_size = self.per_gpu_train_batch_size
        # self.eval_batch_size = self.per_gpu_eval_batch_size
        self.train_batch_size = self.per_gpu_train_batch_size * max(1, self.n_gpu)
        self.eval_batch_size = self.per_gpu_eval_batch_size * max(1, self.n_gpu)
        self.test_batch_size = 1

        if self.fp16:
            apex.amp.register_half_function(torch, "einsum")
        self.stream = torch.cuda.Stream()

    def set_model(self):
        self.model, self.selector = self.callback.load_model()
        self.model.to(self.device)
        self.selector.to(self.device)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": self.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.adam_epsilon)
        self.selector_optimizer = torch.optim.AdamW(self.selector.parameters(), lr=self.selector_learning_rate,
                                                    eps=self.adam_epsilon)
        if self.fp16:
            self.model, self.optimizer = apex.amp.initialize(self.model, self.optimizer, opt_level=self.fp16_opt_level)
            self.selector, self.selector_optimizer = apex.amp.initialize(self.selector, self.selector_optimizer,
                                                                         opt_level=self.fp16_opt_level)
        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.selector = torch.nn.DataParallel(self.selector)
        if self.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank],
                                                                   output_device=self.local_rank,
                                                                   find_unused_parameters=True)
            self.selector = torch.nn.parallel.DistributedDataParallel(self.selector, device_ids=[self.local_rank],
                                                                      output_device=self.local_rank,
                                                                      find_unused_parameters=True)

    def once(self):
        return Once(self.local_rank)

    def once_barrier(self):
        return OnceBarrier(self.local_rank)

    def cache(self):
        return Cache(self.local_rank)

    def load_data(self):
        self.train_step = 1
        self.epochs_trained = 0
        self.steps_trained_in_current_epoch = 0
        self.intro_train_step = 1
        train_dataset, dev_dataset, test_dataset = self.callback.load_data()
        # train_dataset, dev_dataset = self.callback.load_data()
        train_fn, dev_fn, test_fn = self.callback.collate_fn()
        if train_dataset:
            if self.dataset_ratio < 1:
                train_dataset = torch.utils.data.Subset(train_dataset,
                                                        list(range(int(len(train_dataset) * self.dataset_ratio))))
            self.train_dataset = train_dataset
            self.train_sampler = RandomSampler(self.train_dataset) if self.local_rank == -1 else DistributedSampler(
                self.train_dataset)
            self.train_dataloader = Prefetcher(
                DataLoader(self.train_dataset, sampler=self.train_sampler, batch_size=self.train_batch_size,
                           collate_fn=train_fn, num_workers=self.num_workers), self.stream)
            self.t_total = len(self.train_dataloader) // self.gradient_accumulation_steps * self.epochs
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                             num_warmup_steps=int(self.t_total * self.warmup_ratio),
                                                             num_training_steps=self.t_total)
            self.selector_scheduler = get_linear_schedule_with_warmup(self.selector_optimizer,
                                                                      num_warmup_steps=int(
                                                                          self.t_total * self.warmup_ratio),
                                                                      num_training_steps=self.t_total)
        if dev_dataset:
            if self.dataset_ratio < 1:
                dev_dataset = torch.utils.data.Subset(dev_dataset,
                                                      list(range(int(len(dev_dataset) * self.dataset_ratio))))
            self.dev_dataset = dev_dataset
            self.dev_sampler = SequentialSampler(self.dev_dataset) if self.local_rank == -1 else DistributedSampler(
                self.dev_dataset)
            self.dev_dataloader = Prefetcher(
                DataLoader(self.dev_dataset, sampler=self.dev_sampler, batch_size=self.eval_batch_size,
                           collate_fn=dev_fn, num_workers=self.num_workers), self.stream)
        if test_dataset:
            if self.dataset_ratio < 1:
                test_dataset = torch.utils.data.Subset(test_dataset,
                                                       list(range(int(len(test_dataset) * self.dataset_ratio))))
            self.test_dataset = test_dataset
            self.test_sampler = SequentialSampler(self.test_dataset) if self.local_rank == -1 else DistributedSampler(
                self.test_dataset)
            self.test_dataloader = Prefetcher(
                DataLoader(self.test_dataset, sampler=self.test_sampler, batch_size=self.test_batch_size,
                           collate_fn=test_fn, num_workers=self.num_workers), self.stream)

    def restore_checkpoint(self, path, ignore_progress=False):
        if self.no_save:
            return
        model_to_load = self.model.module if hasattr(self.model, "module") else self.model
        model_to_load.load_state_dict(torch.load(os.path.join(path, 'pytorch_model.bin'), map_location=self.device))
        self.optimizer.load_state_dict(torch.load(os.path.join(path, "optimizer.pt"), map_location=self.device))
        self.scheduler.load_state_dict(torch.load(os.path.join(path, "scheduler.pt"), map_location=self.device))

        selector_to_load = self.selector.module if hasattr(self.model, "module") else self.selector
        selector_to_load.load_state_dict(
            torch.load(os.path.join(path, 'selector_pytorch_model.bin'), map_location=self.device))
        self.selector_optimizer.load_state_dict(
            torch.load(os.path.join(path, "selector_optimizer.pt"), map_location=self.device))

        self.callback.on_load(path)
        if not ignore_progress:
            self.train_step = int(path.split("-")[-1])
            self.epochs_trained = self.train_step // (len(self.train_dataloader) // self.gradient_accumulation_steps)
            self.steps_trained_in_current_epoch = self.train_step % (
                        len(self.train_dataloader) // self.gradient_accumulation_steps)
        logging.info("  Continuing training from checkpoint, will skip to saved train_step")
        logging.info("  Continuing training from epoch %d", self.epochs_trained)
        logging.info("  Continuing training from train step %d", self.train_step)
        logging.info("  Will skip the first %d steps in the first epoch", self.steps_trained_in_current_epoch)

    def save_checkpoint(self):
        if self.no_save:
            return
        output_dir = os.path.join('output', "checkpoint-{}".format(self.train_step))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        selector_to_save = self.selector.module if hasattr(self.selector, "module") else self.selector
        torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(self.scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

        torch.save(selector_to_save.state_dict(), os.path.join(output_dir, 'selector_pytorch_model.bin'))
        torch.save(self.selector_optimizer.state_dict(), os.path.join(output_dir, "selector_optimizer.pt"))
        torch.save(self.selector_scheduler.state_dict(), os.path.join(output_dir, "selector_scheduler.pt"))
        self.callback.on_save(output_dir)

    def run(self):
        self.parse_args()
        self.set_env()
        with self.once():
            self.writer = SummaryWriter()
        self.set_model()
        self.load_data()
        if self.load_checkpoint is not None:
            self.restore_checkpoint(self.load_checkpoint, self.ignore_progress)
        best_performance = 0
        best_step = -1
        BERT_MAX_LEN = 512
        for epoch in range(self.epochs):
            # if epoch < self.epochs_trained:
            #     continue
            with self.once():
                logging.info('epoch %d', epoch)
            if self.train:
                tr_loss, tr_s_loss, logging_loss, logging_s_loss = 0.0, 0.0, 0.0, 0.0
                self.model.zero_grad()
                self.model.train()
                self.selector.zero_grad()
                self.selector.train()
                self.callback.on_train_epoch_start(epoch)
                if self.local_rank >= 0:
                    self.train_sampler.set_epoch(epoch)
                print("==========Training==========")
                for step, batch in enumerate(tqdm(self.train_dataloader, disable=self.local_rank > 0)):
                    if step < self.steps_trained_in_current_epoch:
                        continue
                    inputs, inputs_codred, extra = self.callback.process_train_data(batch)
                    num_dsre = len(inputs['lst_input_ids'])
                    num_codred = len(inputs_codred['lst_tokens_codred'])
                    if num_codred != 0:
                        lst_final_scores = []
                        lst_sample_mask = []
                        for i in range(len(inputs_codred['lst_target_inte_codred'])):
                            input_ids = list()
                            token_type_ids = list()
                            attention_mask = list()
                            tokens_codreds = inputs_codred['lst_tokens_codred'][i]
                            target_inte_codreds = inputs_codred['lst_target_inte_codred'][i]
                            ht_sentence_codreds = inputs_codred['lst_ht_sentence_codred'][i]
                            selected_intervals_codreds = inputs_codred['lst_selected_intervals_codred'][i]
                            rerank_scores_codreds = inputs_codred['lst_rerank_scores_codred'][i]
                            final_scores = []
                            sample_mask = []
                            for tokens_codred, target_inte_codred, ht_sentence_codred, selected_intervals_codred, rerank_scores_codred in zip(
                                    tokens_codreds, target_inte_codreds, ht_sentence_codreds,
                                    selected_intervals_codreds, rerank_scores_codreds):
                                k1_rerank_scores, k2_rerank_scores = rerank_scores_codred[0], rerank_scores_codred[1]
                                k1_rerank_scores = torch.tensor(k1_rerank_scores).transpose(0, 1).to(self.device)
                                k1_final_scores = self.selector(k1_rerank_scores)
                                k2_rerank_scores = torch.tensor(k2_rerank_scores).transpose(0, 1).to(self.device)
                                k2_final_scores = self.selector(k2_rerank_scores)

                                # final_scores.append((F.softmax(k1_final_scores, dim=-1), F.softmax(k2_final_scores, dim=-1)))
                                final_scores.append((k1_final_scores, k2_final_scores))
                                k1_sample_mask = torch.tensor([0] * len(k1_final_scores)).to(self.device)
                                k2_sample_mask = torch.tensor([0] * len(k2_final_scores)).to(self.device)
                                sorted_k1_indices = sorted(range(len(k1_final_scores)),
                                                           key=lambda i: k1_final_scores[i], reverse=True)
                                sorted_k1_intervals = [selected_intervals_codred[0][k] for k in sorted_k1_indices]
                                k1_length = len(ht_sentence_codred[0])
                                k1_final_intervals = [(target_inte_codred[0][0], target_inte_codred[0][1])]
                                k1_num_sentences = min(self.args.num_sentences - 2, len(sorted_k1_intervals))
                                for idx in range(k1_num_sentences):
                                    if sorted_k1_intervals[idx][0] == target_inte_codred[0][0] and \
                                            sorted_k1_intervals[idx][1] == target_inte_codred[0][1]:
                                        continue
                                    if len(tokens_codred[0][sorted_k1_intervals[idx][0]:sorted_k1_intervals[idx][
                                        1]]) + k1_length < BERT_MAX_LEN // 2 - 2:
                                        k1_final_intervals.append(sorted_k1_intervals[idx])
                                        k1_length += len(
                                            tokens_codred[0][sorted_k1_intervals[idx][0]:sorted_k1_intervals[idx][1]])
                                    else:
                                        break
                                for inte in k1_final_intervals:
                                    if [inte[0], inte[1]] in selected_intervals_codred[0]:
                                        k1_sample_mask[selected_intervals_codred[0].index([inte[0], inte[1]])] = 1
                                sorted_k1_final_intervals = sorted(k1_final_intervals, key=lambda x: x[0])
                                k1_final_sentences = []
                                for (st, en) in sorted_k1_final_intervals:
                                    if st == target_inte_codred[0][0] and en == target_inte_codred[0][1]:
                                        k1_final_sentences += ht_sentence_codred[0]
                                    else:
                                        k1_final_sentences += tokens_codred[0][st:en]

                                sorted_k2_indices = sorted(range(len(k2_final_scores)),
                                                           key=lambda i: k2_final_scores[i], reverse=True)
                                sorted_k2_intervals = [selected_intervals_codred[1][k] for k in sorted_k2_indices]
                                k2_length = len(ht_sentence_codred[1])
                                k2_final_intervals = [(target_inte_codred[1][0], target_inte_codred[1][1])]
                                k2_num_sentences = min(self.args.num_sentences - 2, len(sorted_k2_intervals))
                                for idx in range(k2_num_sentences):
                                    if sorted_k2_intervals[idx][0] == target_inte_codred[1][0] and \
                                            sorted_k2_intervals[idx][1] == target_inte_codred[1][1]:
                                        continue
                                    if len(tokens_codred[1][sorted_k2_intervals[idx][0]:sorted_k2_intervals[idx][
                                        1]]) + k2_length < BERT_MAX_LEN // 2 - 1:
                                        k2_final_intervals.append(sorted_k2_intervals[idx])
                                        k2_length += len(
                                            tokens_codred[1][sorted_k2_intervals[idx][0]:sorted_k2_intervals[idx][1]])
                                    else:
                                        break
                                for inte in k2_final_intervals:
                                    if [inte[0], inte[1]] in selected_intervals_codred[1]:
                                        k2_sample_mask[selected_intervals_codred[1].index([inte[0], inte[1]])] = 1
                                sample_mask.append((k1_sample_mask, k2_sample_mask))
                                sorted_k2_final_intervals = sorted(k2_final_intervals, key=lambda x: x[0])
                                k2_final_sentences = []
                                for (st, en) in sorted_k2_final_intervals:
                                    if st == target_inte_codred[1][0] and en == target_inte_codred[1][1]:
                                        k2_final_sentences += ht_sentence_codred[1]
                                    else:
                                        k2_final_sentences += tokens_codred[1][st:en]

                                tmp_token = ['[CLS]'] + k1_final_sentences + ['[SEP]'] + k2_final_sentences + ['[SEP]']
                                tmp_token_ids = self.callback.tokenizer.convert_tokens_to_ids(tmp_token)
                                if len(tmp_token_ids) < BERT_MAX_LEN:
                                    tmp_token_ids = tmp_token_ids + [0] * (BERT_MAX_LEN - len(tmp_token_ids))
                                tmp_attention_mask = [1] * len(tmp_token) + [0] * (BERT_MAX_LEN - len(tmp_token))
                                tmp_token_type_ids = [0] * (len(k1_final_sentences) + 2) + [1] * (
                                            len(k2_final_sentences) + 1) + [0] * (BERT_MAX_LEN - len(tmp_token))
                                input_ids.append(tmp_token_ids)
                                token_type_ids.append(tmp_token_type_ids)
                                attention_mask.append(tmp_attention_mask)

                            input_ids_t = torch.tensor(input_ids, dtype=torch.int64)
                            token_type_ids_t = torch.tensor(token_type_ids, dtype=torch.int64)
                            attention_mask_t = torch.tensor(attention_mask, dtype=torch.int64)
                            lst_final_scores.append(final_scores)
                            lst_sample_mask.append(sample_mask)
                            inputs['lst_input_ids'].append(input_ids_t)
                            inputs['lst_token_type_ids'].append(token_type_ids_t)
                            inputs['lst_attention_mask'].append(attention_mask_t)
                        inputs['lst_dplabel'] = inputs['lst_dplabel'] + inputs_codred['lst_dplabel_codred']
                        inputs['lst_rs'] = inputs['lst_rs'] + inputs_codred['lst_rs_codred']

                        extra['lst_rs'] = extra['lst_rs'] + extra['lst_rs_codred']
                    outputs = self.model(**inputs)
                    loss = outputs[0]
                    if num_codred != 0:
                        s_loss = 0.0
                        lst_reward = []
                        for idx, final_scores in enumerate(lst_final_scores):
                            pred_prob = outputs[3][num_dsre + idx][0].detach()
                            thr = outputs[4][num_dsre + idx][0]
                            dplabel = inputs['lst_dplabel'][num_dsre + idx]
                            for idx_path, _ in enumerate(final_scores):
                                reward = pred_prob[idx_path][dplabel[idx_path]] - thr[idx_path][0]
                                lst_reward.append(reward)
                        k = 0
                        reward_mean = torch.mean(torch.stack(lst_reward))
                        for idx, final_scores in enumerate(lst_final_scores):
                            dplabel = inputs['lst_dplabel'][num_dsre + idx]
                            sample_mask = lst_sample_mask[idx]
                            for idx_path, final_score in enumerate(final_scores):
                                reward = lst_reward[k] - reward_mean
                                if dplabel[idx_path] != 0:
                                    reward = reward * self.args.lam_notnone
                                else:
                                    reward = min(reward, self.args.reward_limit)
                                    reward = reward * self.args.lam_none
                                k1_final_scores, k2_final_scores = final_score[0], final_score[1]
                                k1_sample_mask, k2_sample_mask = sample_mask[idx_path][0], sample_mask[idx_path][1]
                                log_k1_prob = torch.log(F.softmax(k1_final_scores, dim=-1))
                                log_k2_prob = torch.log(F.softmax(k2_final_scores, dim=-1))

                                sampled_log_k1_prob = log_k1_prob * k1_sample_mask
                                sampled_k1_loss = -torch.sum(sampled_log_k1_prob * reward.unsqueeze(-1))
                                k1_non_sample_mask = 1 - k1_sample_mask
                                k1_non_sample_scores = k1_final_scores * k1_non_sample_mask
                                k1_sample_min_score = torch.min(k1_final_scores[k1_sample_mask.bool()])
                                k1_ranking_loss = torch.sum(torch.relu(k1_non_sample_scores - k1_sample_min_score))

                                loss1 = sampled_k1_loss + self.args.lam_rank * k1_ranking_loss

                                sampled_log_k2_prob = log_k2_prob * k2_sample_mask
                                sampled_k2_loss = -torch.sum(sampled_log_k2_prob * reward.unsqueeze(-1))
                                k2_non_sample_mask = 1 - k2_sample_mask
                                k2_non_sample_scores = k2_final_scores * k2_non_sample_mask
                                k2_sample_min_score = torch.min(k2_final_scores[k2_sample_mask.bool()])
                                k2_ranking_loss = torch.sum(torch.relu(k2_non_sample_scores - k2_sample_min_score))
                                loss2 = sampled_k2_loss + self.args.lam_rank * k2_ranking_loss

                                s_loss = s_loss + 0.5 * loss1 + 0.5 * loss2
                                k = k + 1

                    if step % 5000 == 0:
                        print(loss)
                    if self.n_gpu > 1:
                        loss = loss.mean()
                        if num_codred != 0:
                            s_loss = s_loss.mean()
                    if self.gradient_accumulation_steps > 1:
                        loss = loss / self.gradient_accumulation_steps
                        if num_codred != 0:
                            s_loss = s_loss / self.gradient_accumulation_steps
                    if self.local_rank < 0 or (step + 1) % self.gradient_accumulation_steps == 0:
                        if self.fp16:
                            with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                                scaled_loss.backward()
                            if num_codred != 0:
                                with apex.amp.scale_loss(s_loss, self.selector_optimizer) as scaled_s_loss:
                                    scaled_s_loss.backward()
                        else:
                            loss.backward()
                            if num_codred != 0:
                                s_loss.backward()
                    else:
                        with self.model.no_sync():
                            if self.fp16:
                                with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                                    scaled_loss.backward()
                                if num_codred != 0:
                                    with apex.amp.scale_loss(s_loss, self.selector_optimizer) as scaled_s_loss:
                                        scaled_s_loss.backward()
                            else:
                                loss.backward()
                                if num_codred != 0:
                                    s_loss.backward()
                    tr_loss += loss.item()
                    if num_codred != 0:
                        tr_s_loss += s_loss.item()
                    if (step + 1) % self.gradient_accumulation_steps == 0:
                        if self.fp16:
                            torch.nn.utils.clip_grad_norm_(apex.amp.master_params(self.optimizer), self.max_grad_norm)
                            if num_codred != 0:
                                torch.nn.utils.clip_grad_norm_(apex.amp.master_params(self.selector_optimizer),
                                                               self.max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                            if num_codred != 0:
                                torch.nn.utils.clip_grad_norm_(self.selector.parameters(), self.max_grad_norm)
                        self.optimizer.step()
                        self.scheduler.step()
                        if num_codred != 0:
                            self.selector_optimizer.step()
                            self.selector_scheduler.step()

                        self.model.zero_grad()
                        self.selector.zero_grad()
                        self.train_step += 1
                        with self.once():
                            if self.train_step % self.logging_steps == 0:
                                self.writer.add_scalar("lr", self.scheduler.get_lr()[0], self.train_step)
                                self.writer.add_scalar("loss", (tr_loss - logging_loss) / self.logging_steps,
                                                       self.train_step)
                                if num_codred != 0:
                                    self.writer.add_scalar("s_loss", (tr_s_loss - logging_s_loss) / self.logging_steps,
                                                           self.train_step)
                                logging_loss = tr_loss
                                if num_codred != 0:
                                    logging_s_loss = tr_s_loss
                    # torch.cuda.empty_cache()
                    self.callback.on_train_step(step, self.train_step, inputs, extra, loss.item(), outputs)
                with self.once():
                    self.save_checkpoint()
                self.callback.on_train_epoch_end(epoch)
            if self.dev:
                with torch.no_grad():
                    self.model.eval()
                    self.selector.eval()
                    self.callback.on_dev_epoch_start(epoch)
                    for step, batch in enumerate(tqdm(self.dev_dataloader, disable=self.local_rank > 0)):
                        inputs_codred, extra = self.callback.process_dev_data(batch)
                        input_ids = list()
                        token_type_ids = list()
                        attention_mask = list()
                        tokens_codreds = inputs_codred['lst_tokens_codred']
                        target_inte_codreds = inputs_codred['lst_target_inte_codred']
                        ht_sentence_codreds = inputs_codred['lst_ht_sentence_codred']
                        selected_intervals_codreds = inputs_codred['lst_selected_intervals_codred']
                        rerank_scores_codreds = inputs_codred['lst_rerank_scores_codred']
                        for tokens_codred, target_inte_codred, ht_sentence_codred, selected_intervals_codred, rerank_scores_codred in zip(
                                tokens_codreds, target_inte_codreds, ht_sentence_codreds, selected_intervals_codreds,
                                rerank_scores_codreds):
                            k1_rerank_scores, k2_rerank_scores = rerank_scores_codred[0], rerank_scores_codred[1]
                            k1_rerank_scores = torch.tensor(k1_rerank_scores).transpose(0, 1).to(self.device)
                            k1_final_scores = self.selector(k1_rerank_scores)
                            k2_rerank_scores = torch.tensor(k2_rerank_scores).transpose(0, 1).to(self.device)
                            k2_final_scores = self.selector(k2_rerank_scores)

                            sorted_k1_indices = sorted(range(len(k1_final_scores)), key=lambda i: k1_final_scores[i],
                                                       reverse=True)
                            sorted_k1_intervals = [selected_intervals_codred[0][k] for k in sorted_k1_indices]
                            k1_length = len(ht_sentence_codred[0])
                            k1_final_intervals = [(target_inte_codred[0][0], target_inte_codred[0][1])]
                            k1_num_sentences = min(self.args.num_sentences - 2, len(sorted_k1_intervals))
                            for idx in range(k1_num_sentences):
                                if sorted_k1_intervals[idx][0] == target_inte_codred[0][0] and sorted_k1_intervals[idx][
                                    1] == target_inte_codred[0][1]:
                                    continue
                                if len(tokens_codred[0][sorted_k1_intervals[idx][0]:sorted_k1_intervals[idx][
                                    1]]) + k1_length < BERT_MAX_LEN // 2 - 2:
                                    k1_final_intervals.append(sorted_k1_intervals[idx])
                                    k1_length += len(
                                        tokens_codred[0][sorted_k1_intervals[idx][0]:sorted_k1_intervals[idx][1]])
                                else:
                                    break
                            sorted_k1_final_intervals = sorted(k1_final_intervals, key=lambda x: x[0])
                            k1_final_sentences = []
                            for (st, en) in sorted_k1_final_intervals:
                                if st == target_inte_codred[0][0] and en == target_inte_codred[0][1]:
                                    k1_final_sentences += ht_sentence_codred[0]
                                else:
                                    k1_final_sentences += tokens_codred[0][st:en]

                            sorted_k2_indices = sorted(range(len(k2_final_scores)), key=lambda i: k2_final_scores[i],
                                                       reverse=True)
                            sorted_k2_intervals = [selected_intervals_codred[1][k] for k in sorted_k2_indices]
                            k2_length = len(ht_sentence_codred[1])
                            k2_final_intervals = [(target_inte_codred[1][0], target_inte_codred[1][1])]
                            k2_num_sentences = min(self.args.num_sentences - 2, len(sorted_k2_intervals))
                            for idx in range(k2_num_sentences):
                                if sorted_k2_intervals[idx][0] == target_inte_codred[1][0] and sorted_k2_intervals[idx][
                                    1] == target_inte_codred[1][1]:
                                    continue
                                if len(tokens_codred[1][sorted_k2_intervals[idx][0]:sorted_k2_intervals[idx][
                                    1]]) + k2_length < BERT_MAX_LEN // 2 - 1:
                                    k2_final_intervals.append(sorted_k2_intervals[idx])
                                    k2_length += len(
                                        tokens_codred[1][sorted_k2_intervals[idx][0]:sorted_k2_intervals[idx][1]])
                                else:
                                    break
                            sorted_k2_final_intervals = sorted(k2_final_intervals, key=lambda x: x[0])
                            k2_final_sentences = []
                            for (st, en) in sorted_k2_final_intervals:
                                if st == target_inte_codred[1][0] and en == target_inte_codred[1][1]:
                                    k2_final_sentences += ht_sentence_codred[1]
                                else:
                                    k2_final_sentences += tokens_codred[1][st:en]

                            tmp_token = ['[CLS]'] + k1_final_sentences + ['[SEP]'] + k2_final_sentences + ['[SEP]']
                            tmp_token_ids = self.callback.tokenizer.convert_tokens_to_ids(tmp_token)
                            if len(tmp_token_ids) < BERT_MAX_LEN:
                                tmp_token_ids = tmp_token_ids + [0] * (BERT_MAX_LEN - len(tmp_token_ids))
                            tmp_attention_mask = [1] * len(tmp_token) + [0] * (BERT_MAX_LEN - len(tmp_token))
                            tmp_token_type_ids = [0] * (len(k1_final_sentences) + 2) + [1] * (
                                    len(k2_final_sentences) + 1) + [0] * (BERT_MAX_LEN - len(tmp_token))
                            input_ids.append(tmp_token_ids)
                            token_type_ids.append(tmp_token_type_ids)
                            attention_mask.append(tmp_attention_mask)

                        input_ids_t = torch.tensor(input_ids, dtype=torch.int64)
                        token_type_ids_t = torch.tensor(token_type_ids, dtype=torch.int64)
                        attention_mask_t = torch.tensor(attention_mask, dtype=torch.int64)
                        input_ids_t = torch.stack([input_ids_t])
                        token_type_ids_t = torch.stack([token_type_ids_t])
                        attention_mask_t = torch.stack([attention_mask_t])

                        inputs = {
                            'lst_input_ids': input_ids_t,
                            'lst_token_type_ids': token_type_ids_t,
                            'lst_attention_mask': attention_mask_t,
                            'train': False
                        }
                        # with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=True, profile_memory=True, with_stack=True, with_modules=True) as prof:
                        outputs = self.model(**inputs)
                        # print(prof.key_averages().table(sort_by="cuda_time_total"))
                        # prof.export_chrome_trace('./codred_profile.json')
                        self.callback.on_dev_step(step, inputs, extra, outputs)
                    performance = self.callback.on_dev_epoch_end(epoch)
                    if performance > best_performance:
                        best_performance = performance
                        best_step = self.train_step

        if self.test:
            with torch.no_grad():
                if best_step > 0 and self.train:
                    self.restore_checkpoint(os.path.join('output', "checkpoint-{}".format(best_step)))
                self.model.eval()
                self.selector.eval()
                self.callback.on_test_epoch_start(epoch)
                for step, batch in enumerate(tqdm(self.test_dataloader, disable=self.local_rank > 0)):
                    inputs_codred, extra = self.callback.process_test_data(batch)
                    input_ids = list()
                    token_type_ids = list()
                    attention_mask = list()
                    tokens_codreds = inputs_codred['lst_tokens_codred']
                    target_inte_codreds = inputs_codred['lst_target_inte_codred']
                    ht_sentence_codreds = inputs_codred['lst_ht_sentence_codred']
                    selected_intervals_codreds = inputs_codred['lst_selected_intervals_codred']
                    rerank_scores_codreds = inputs_codred['lst_rerank_scores_codred']
                    for tokens_codred, target_inte_codred, ht_sentence_codred, selected_intervals_codred, rerank_scores_codred in zip(
                            tokens_codreds, target_inte_codreds, ht_sentence_codreds, selected_intervals_codreds,
                            rerank_scores_codreds):
                        k1_rerank_scores, k2_rerank_scores = rerank_scores_codred[0], rerank_scores_codred[1]
                        k1_rerank_scores = torch.tensor(k1_rerank_scores).transpose(0, 1).to(self.device)
                        k1_final_scores = self.selector(k1_rerank_scores)
                        k2_rerank_scores = torch.tensor(k2_rerank_scores).transpose(0, 1).to(self.device)
                        k2_final_scores = self.selector(k2_rerank_scores)

                        sorted_k1_indices = sorted(range(len(k1_final_scores)), key=lambda i: k1_final_scores[i],
                                                   reverse=True)
                        sorted_k1_intervals = [selected_intervals_codred[0][k] for k in sorted_k1_indices]
                        k1_length = len(ht_sentence_codred[0])
                        k1_final_intervals = [(target_inte_codred[0][0], target_inte_codred[0][1])]
                        k1_num_sentences = min(self.args.num_sentences - 2, len(sorted_k1_intervals))
                        for idx in range(k1_num_sentences):
                            if sorted_k1_intervals[idx][0] == target_inte_codred[0][0] and sorted_k1_intervals[idx][
                                1] == target_inte_codred[0][1]:
                                continue
                            if len(tokens_codred[0][sorted_k1_intervals[idx][0]:sorted_k1_intervals[idx][
                                1]]) + k1_length < BERT_MAX_LEN // 2 - 2:
                                k1_final_intervals.append(sorted_k1_intervals[idx])
                                k1_length += len(
                                    tokens_codred[0][sorted_k1_intervals[idx][0]:sorted_k1_intervals[idx][1]])
                            else:
                                break
                        sorted_k1_final_intervals = sorted(k1_final_intervals, key=lambda x: x[0])
                        k1_final_sentences = []
                        for (st, en) in sorted_k1_final_intervals:
                            if st == target_inte_codred[0][0] and en == target_inte_codred[0][1]:
                                k1_final_sentences += ht_sentence_codred[0]
                            else:
                                k1_final_sentences += tokens_codred[0][st:en]

                        sorted_k2_indices = sorted(range(len(k2_final_scores)), key=lambda i: k2_final_scores[i],
                                                   reverse=True)
                        sorted_k2_intervals = [selected_intervals_codred[1][k] for k in sorted_k2_indices]
                        k2_length = len(ht_sentence_codred[1])
                        k2_final_intervals = [(target_inte_codred[1][0], target_inte_codred[1][1])]
                        k2_num_sentences = min(self.args.num_sentences - 2, len(sorted_k2_intervals))
                        for idx in range(k2_num_sentences):
                            if sorted_k2_intervals[idx][0] == target_inte_codred[1][0] and sorted_k2_intervals[idx][
                                1] == target_inte_codred[1][1]:
                                continue
                            if len(tokens_codred[1][sorted_k2_intervals[idx][0]:sorted_k2_intervals[idx][
                                1]]) + k2_length < BERT_MAX_LEN // 2 - 1:
                                k2_final_intervals.append(sorted_k2_intervals[idx])
                                k2_length += len(
                                    tokens_codred[1][sorted_k2_intervals[idx][0]:sorted_k2_intervals[idx][1]])
                            else:
                                break
                        sorted_k2_final_intervals = sorted(k2_final_intervals, key=lambda x: x[0])
                        k2_final_sentences = []
                        for (st, en) in sorted_k2_final_intervals:
                            if st == target_inte_codred[1][0] and en == target_inte_codred[1][1]:
                                k2_final_sentences += ht_sentence_codred[1]
                            else:
                                k2_final_sentences += tokens_codred[1][st:en]

                        tmp_token = ['[CLS]'] + k1_final_sentences + ['[SEP]'] + k2_final_sentences + ['[SEP]']
                        tmp_token_ids = self.callback.tokenizer.convert_tokens_to_ids(tmp_token)
                        if len(tmp_token_ids) < BERT_MAX_LEN:
                            tmp_token_ids = tmp_token_ids + [0] * (BERT_MAX_LEN - len(tmp_token_ids))
                        tmp_attention_mask = [1] * len(tmp_token) + [0] * (BERT_MAX_LEN - len(tmp_token))
                        tmp_token_type_ids = [0] * (len(k1_final_sentences) + 2) + [1] * (
                                len(k2_final_sentences) + 1) + [0] * (BERT_MAX_LEN - len(tmp_token))
                        input_ids.append(tmp_token_ids)
                        token_type_ids.append(tmp_token_type_ids)
                        attention_mask.append(tmp_attention_mask)

                    input_ids_t = torch.tensor(input_ids, dtype=torch.int64)
                    token_type_ids_t = torch.tensor(token_type_ids, dtype=torch.int64)
                    attention_mask_t = torch.tensor(attention_mask, dtype=torch.int64)

                    input_ids_t = torch.stack([input_ids_t])
                    token_type_ids_t = torch.stack([token_type_ids_t])
                    attention_mask_t = torch.stack([attention_mask_t])

                    inputs = {
                        'lst_input_ids': input_ids_t,
                        'lst_token_type_ids': token_type_ids_t,
                        'lst_attention_mask': attention_mask_t,
                        'train': False
                    }
                    outputs = self.model(**inputs)
                    self.callback.on_test_step(step, inputs, extra, outputs)
                self.callback.on_test_epoch_end(epoch)
        with self.once():
            self.writer.close()
        json.dump(True, open('output/f.json', 'w'))

    def distributed_broadcast(self, l):
        assert type(l) == list or type(l) == dict
        if self.local_rank < 0:
            return l
        else:
            torch.distributed.barrier()
            process_number = torch.distributed.get_world_size()
            json.dump(l, open(f'tmp/{self.local_rank}.json', 'w'))
            torch.distributed.barrier()
            objs = list()
            for i in range(process_number):
                objs.append(json.load(open(f'tmp/{i}.json')))
            if type(objs[0]) == list:
                ret = list()
                for i in range(process_number):
                    ret.extend(objs[i])
            else:
                ret = dict()
                for i in range(process_number):
                    for k, v in objs.items():
                        assert k not in ret
                        ret[k] = v
            torch.distributed.barrier()
            return ret

    def distributed_merge(self, l):
        assert type(l) == list or type(l) == dict
        if self.local_rank < 0:
            return l
        else:
            torch.distributed.barrier()
            process_number = torch.distributed.get_world_size()
            json.dump(l, open(f'tmp/{self.local_rank}.json', 'w'))
            torch.distributed.barrier()
            if self.local_rank == 0:
                objs = list()
                for i in range(process_number):
                    objs.append(json.load(open(f'tmp/{i}.json')))
                if type(objs[0]) == list:
                    ret = list()
                    for i in range(process_number):
                        ret.extend(objs[i])
                else:
                    ret = dict()
                    for i in range(process_number):
                        for k, v in objs.items():
                            assert k not in ret
                            ret[k] = v
            else:
                ret = None
            torch.distributed.barrier()
            return ret

    def distributed_get(self, v):
        if self.local_rank < 0:
            return v
        else:
            torch.distributed.barrier()
            if self.local_rank == 0:
                json.dump(v, open('tmp/v.json', 'w'))
            torch.distributed.barrier()
            v = json.load(open('tmp/v.json'))
            torch.distributed.barrier()
            return v

    def _write_estimation(self, buf, relevance_blk, f):
        for i, blk in enumerate(buf):
            f.write(f'{blk.pos} {relevance_blk[i].item()}\n')

    def _score_blocks(self, qbuf, relevance_token):
        ends = qbuf.block_ends()
        relevance_blk = torch.ones(len(ends), device='cpu')
        for i in range(len(ends)):
            if qbuf[i].blk_type > 0:  # query
                relevance_blk[i] = (relevance_token[ends[i - 1]:ends[i]]).mean()
        return relevance_blk

    def _collect_estimations_from_dir(self, est_dir):
        ret = {}
        for shortname in os.listdir(est_dir):
            filename = os.path.join(est_dir, shortname)
            if shortname.startswith('estimations_'):
                with open(filename, 'r') as fin:
                    for line in fin:
                        l = line.split()
                        pos, estimation = int(l[0]), float(l[1])
                        ret[pos].estimation = estimation
                os.replace(filename, os.path.join(est_dir, 'backup_' + shortname))
        return ret


class AttentionModule(nn.Module):
    def __init__(self, input_size):
        super(AttentionModule, self).__init__()
        self.attn_weights = nn.Linear(input_size, input_size)
        self.bias = nn.Parameter(torch.zeros(input_size))

    def forward(self, x):
        raw_weights = self.attn_weights(x) + self.bias
        attn_scores = F.softmax(raw_weights, dim=-1)
        return x * attn_scores


class RerankSelector(nn.Module):
    def __init__(self, num_rerankers, hidden_dim=512, dropout_rate=0.5):
        super(RerankSelector, self).__init__()
        self.layer_norm = nn.LayerNorm(num_rerankers)
        self.attention = AttentionModule(num_rerankers)
        self.fc1 = nn.Linear(num_rerankers, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, scores):
        # scores: batch_size x num_rerankers
        normalized_scores = self.layer_norm(scores)
        # normalized_scores = self.attention(normalized_scores)
        x = torch.relu(self.fc1(normalized_scores))
        x = self.dropout(x)
        x = self.fc2(x)  # Output: batch_size x 1 (final score for each text segment)
        return x

# class RerankSelector(nn.Module):
#     def __init__(self, num_rerankers, hidden_dim=512, dropout_rate=0.5):
#         super(RerankSelector, self).__init__()
#         self.layer_norm = nn.LayerNorm(num_rerankers)
#         self.fc1 = nn.Linear(num_rerankers, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, 1)
#         self.dropout = nn.Dropout(p=dropout_rate)
#
#     def forward(self, scores):
#         # scores: batch_size x num_rerankers
#         normalized_scores = self.layer_norm(scores)
#         x = torch.relu(self.fc1(normalized_scores))
#         x = self.dropout(x)
#         x = self.fc2(x)  # Output: batch_size x 1 (final score for each text segment)
#         return x