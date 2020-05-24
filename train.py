from . import process_data
from . import Parrot, ParrotConfig
import argparse
import json
import numpy as np
import torch
import torch.nn as nn


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training script for Parrot chatbot.")

    parser.add_argument("--source", "--data_dir", "-d",
                        required=True,
                        type=str)

    parser.add_argument("--d_model", "-d_model", "-dm",
                        type=int,
                        help="The number of hidden neurons in the model",
                        default=256)

    parser.add_argument("--d_ff", "-dff",
                        type=int,
                        help="The number of hidden neurons for the "
                             "feed-forward layers",
                        default=512)

    parser.add_argument("--num_layers", "-num_layers", "-nl",
                        type=int,
                        help="The number of encoder and decoder layers",
                        default=6)

    parser.add_argument("--num_heads", "-num_heads", "-nh",
                        type=int,
                        help="The number of heads to split into for multi-head "
                             "attention",
                        default=8)

    parser.add_argument("--dropout", "-dropout", "-dr",
                        type=float,
                        help="The probability of dropout regularization",
                        default=0.1)

    parser.add_argument("--no_tie_weights", "-no_tie_weights", "-ntw",
                        type=bool,
                        help="Disable the use the embedding weights as output "
                             "projections",
                        action="store_false")

    parser.add_argument("--initial_learning_rate", "-initial_learning_rate",
                        "--initial_lr", "-initial_lr", "-ilr",
                        type=float,
                        help="The initial learning rate for the optimizer",
                        default=0.0)

    parser.add_argument("--batch_size", "-batch_size", "-bs",
                        type=int,
                        help="The number of training examples handled by "
                             "each iteration of the training loop",
                        default=32)

    parser.add_argument("--epochs", "-epochs", "-ep",
                        type=int,
                        help="The number of epochs to train for",
                        default=1)

    parser.add_argument("--optimizer", "-optimizer", "--optim", "-optim",
                        "-opt",
                        type=str,
                        help="The gradient descent algorithm to use",
                        choices=["sgd", "adam"],
                        default="adam")

    parser.add_argument("--save_checkpoint", "-save_checkpoint", "-sc",
                        type=bool,
                        help="Whether or not to save checkpoints",
                        action="store_true")

    parser.add_argument("--checkpoint_dir", "-checkpoint_dir", "-cd",
                        type=str,
                        help="The directory to store the checkpoints",
                        default=".")

    parser.add_argument("--checkpoint_freq", "-checkpoint_freq", "-cf",
                        type=int,
                        help="The number of iterations until the next "
                             "checkpoint is saved.",
                        default=None)

    parser.add_argument("--checkpoint_name", "-checkpoint_name", "-cn",
                        type=str,
                        help="Name of the checkpoint file",
                        default="parrot_checkpoint")

    parser.add_argument("--from_checkpoint", "-from_checkpoint", "-fc",
                        type=str,
                        help="The path to the checkpoint to load; "
                             "if this is provided, the checkpoint will be "
                             "loaded",
                        default=None)

    parser.add_argument("--from_configurations", "-from_configurations",
                        "--from_configs", "-from_configs", "-fc",
                        type=str,
                        help="The path to the configurations file to load into "
                             "a ParrotConfig class; should be a JSON file; if "
                             "this is provided, this will be the config used",
                        default=None)

    parser.add_argument("--export_configurations", "-export_configurations",
                        "-ec",
                        type=str,
                        help="The path to save the current configurations as a "
                             "JSON file for future use.",
                        default=None)

    parser.add_argument("--parallel", "-parallel", "-pll",
                        type=bool,
                        help="Turn on parallel training on the GPUs (only turn "
                             "this on if there are multiple GPUs present)",
                        action="store_true")

    args = parser.parse_args()

    train_dataloader, val_dataloader, tokenizer, longest_seq_len = process_data(
        args.source, args.batch_size)

    if args.from_configurations is not None:
        with open(args.from_configurations) as loaded_configurations:
            configs = json.load(loaded_configurations)

    else:
        configs = {
            "num_words": tokenizer.num_words,
            "d_model": args.d_model,
            "d_ff": args.d_ff,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "use_embed_weights": True if not args.no_tie_weights else False,
            "set_seq_len": longest_seq_len,
            "optimizer": args.optimizer,
        }

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_config = ParrotConfig.from_dict(configs)
    model = Parrot(config=model_config, device=device, parallel=args.parallel)


    if args.export_configurations is not None:
        model_config.to_dict(to_json=True, out_path=args.export_configurations)

    # if the checkpoint frequency isn't given and checkpoint is turned on,
    # we will save ~20 checkpoints:
    if args.save_checkpoint and args.checkpoint_freq is None:
        args.checkpoint_freq = np.ceil(len(train_dataloader) / 20)

    model.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=args.epochs,
        checkpoint=args.save_checkpoint,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.checkpoint_name
    )

    dir_path = args.checkpoint_dir if args.checkpoint_dir.endswith("/") \
        else args.checkpoint_dir + "/"

    save_path = f"{dir_path}{args.checkpoint_name}_final.tar"

    torch.save(model.get_stored_weights(), save_path)
