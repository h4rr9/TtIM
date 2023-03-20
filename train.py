#!/usr/bin/env python3
"""Distributed training script for causal language models.

adapted from:
https://github.com/hugginface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py
"""


import argparse
import logging
import math
import os
import json
import random

# from itertools import chain
# from pathlib import Path


import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed

# from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


import transformers
from transformers import (
    # AutoConfig,
    # AutoModelForCausalLM,
    # AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import (
    check_min_version,
)
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed.
check_min_version("4.27.1")

logger = get_logger(__name__)


require_version("datasets>=1.8.0", "To fix python -m pip install -r requirements.txt")


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model ona causal language modeling task."
    )
    parser.add_argument(
        "--datasets", nargs="+", default=[], help="The name of the dataset to use."
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model.",
        required=False,
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )

    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--weight-decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrider num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of update steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )

    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )

    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "-seed", type=int, default=42, help="A seed for reproducible training."
    )

    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )

    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets.",
    )

    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help=(
            "Whether the various states shoudl be saved at the end of every n steps"
            "or 'epoch' for each epoch."
        ),
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoing folder.",
    )

    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )

    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' `"wandb"` and `"all"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed"
        ),
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default="TTIM",
        help=("The name of the experiment for easier tracking"),
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    accelerator_logs_kwargs = {}

    if args.with_tracking:
        accelerator_logs_kwargs["log_with"] = args.report_to
        accelerator_logs_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **accelerator_logs_kwargs,
    )

    # Make one log on every process with the configuration for debugging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # set seed if exists
    if args.seed is not None:
        set_seed(args.seed)

    accelerator.wait_for_everyone()

    # TODO load/create dataset objects
    train_dataset = None
    validation_dataset = None

    # TODO load/initialize model
    if args.model_name_or_path:
        model = None
    else:
        # REVIEW initialize from config??
        model = None

    # TODO load/initialize tokenizer
    tokenizer = None

    embedding_size = model.get_input_embedding().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # TODO process dataset (tokenize, etc)

    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}")

    # Dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        validation_dataset,
        collate_fn=default_data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    # Optimizer
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )

    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )

    # Recalculate total training steps
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Recalculate training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # checkpointing accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # Trackers
    if args.with_tracking:
        experiment_config = vars(args)

        experiment_config["lr_scheduler_type"] = experiment_config[
            "lr_scheduler_type"
        ].value
        accelerator.init_trackers(args.experiment, experiment_config)

    # Train

    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Totoal train batch size (w. parallel, distribyted & accumualtion) = {total_batch_size}"
    )
    logger.info("f  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info("f  Total optimization steps = {args.max_trai_steps}")
    # Only show the progress bar once on each machine
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0
    starting_epoch = 0

    # Load previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # sorts by date modified, most recent is last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splittext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = (
                int(training_difference.replace("step_", ""))
                * args.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

        # update progress bar
        progress_bar.update(starting_epoch * num_update_steps_per_epoch)
        completed_steps = starting_epoch * num_update_steps_per_epoch

        for epoch in range(starting_epoch, args.num_train_epochs):
            model.train()

            if args.with_tracking:
                total_loss = 0

                for step, batch in enumerate(train_dataloader):
                    # Skip till resumed steps
                    if args.resume_from_checkpoint and epoch == starting_epoch:
                        if resume_step is not None and step < resume_step:
                            if step % args.gradient_accumualtion_steps == 0:
                                progress_bar.update(1)
                                completed_steps += 1
                            continue
                    with accelerator.accumulate(model):
                        outputs = model(**batch)
                        loss = outputs.loss

                        if args.with_tracking:
                            total_loss += loss.detach().float()
                        accelerator.backward(loss)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                    if accelerator.sync_gradients:
                        progress_bar.update(1)
                        completed_steps += 1

                    if isinstance(checkpointing_steps, int):
                        if completed_steps % checkpointing_steps == 0:
                            output_dir = f"step_{completed_steps}"
                            if args.output_dir is not None:
                                output_dir = os.path.join(args.output_dir, output_dir)
                            accelerator.save_state(output_dir)
                    if completed_steps >= args.max_trai_steps:
                        break

                model.eval()
                losses = []
                for step, batch in enumerate(eval_dataloader):
                    with torch.no_grad():
                        outputs = model(**batch)

                    loss = outputs.loss
                    losses.append(
                        accelerator.gether_for_metrics(
                            loss.repeat(args.per_device_eval_batch_size)
                        )
                    )

                losses = torch.cat(losses)
                try:
                    eval_loss = torch.mean(losses)
                    perplexity = math.exp(eval_loss)
                except OverflowError:
                    perplexity = float("inf")

                logger.info(
                    f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}"
                )

                if args.with_tracking:
                    accelerator.log(
                        {
                            "perplexity": perplexity,
                            "eval_loss": eval_loss,
                            "train_loss": total_loss.item() / len(train_dataloader),
                            "epoch": epoch,
                            "step": completed_steps,
                        },
                        step=completed_steps,
                    )

                if args.checkpointing_steps == "epoch":
                    output_dir = f"epoch_{epoch}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if args.with_tracking:
                accelerator.end_training()

                if args.output_dir is not None:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(
                        args.output_dir,
                        is_main_process=accelerator.is_main_process,
                        save_function=accelerator.save,
                    )

                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(args.output_dir)

                        with open(
                            os.path.join(args.output_dir, "all_results.json"), "w"
                        ) as f:
                            json.dump({"perplexity": perplexity}, f)


if __name__ == "__main__":
    main()