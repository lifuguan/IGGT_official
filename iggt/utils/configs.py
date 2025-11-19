import argparse, os
from mmengine.config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    
    parser.add_argument("--config", default="configs/base.py" ,help="model config file path")
    parser.add_argument(
        "--resume_model_path",
        type=str,
        default=None,
        help="Path to resume trained models.",
    )
    parser.add_argument("--wandb", type=bool, help="Wandb Logging.")

    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args 


def merge_args(cfg, args):
    # if args.ckpt_path is not None:
    #     cfg.model["from_pretrained"] = args.ckpt_path
    #     if cfg.get("discriminator") is not None:
    #         cfg.discriminator["from_pretrained"] = args.ckpt_path
    #     args.ckpt_path = None
    for k, v in vars(args).items():
        if v is not None:
            cfg[k] = v

    return cfg


def read_config(config_path):
    cfg = Config.fromfile(config_path)
    return cfg


def parse_configs():
    args = parse_args()
    cfg = read_config(args.config)
    cfg = merge_args(cfg, args)
    return cfg

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
