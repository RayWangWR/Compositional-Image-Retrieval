import argparse


def str2bool(v):
    return v.lower() in ('true', '1')


parser = argparse.ArgumentParser()

# -------------------------------------------------------------------------
# Data input settings
parser.add_argument("--root", type=str, default="data/", help="path to dataset")
parser.add_argument("--output-dir", type=str, default="", help="output directory")
parser.add_argument("--config-file", type=str, default="", help="path to config file")
parser.add_argument("--dataset_name", type=str, default="B2W", help="path to config file for dataset ")
parser.add_argument("--dataset-config-file", type=str, default="", help="path to config file for dataset setup", )
# -------------------------------------------------------------------------
# Model input settings
parser.add_argument("--prompt_path", type=str, default="", help="checkpoint directory", )
parser.add_argument("--model_path", type=str, default="ViT-B-16", help="eval-only mode", )
# -------------------------------------------------------------------------
# Model params
parser.add_argument("--seed", type=int, default=-1, help="only positive value enables a fixed seed")
parser.add_argument("--trainer", type=str, default="", help="name of trainer")
parser.add_argument("--backbone", type=str, default="ViT-B-16", help="name of CNN backbone")
parser.add_argument("--head", type=str, default="", help="name of head")
# Prompt Learner
parser.add_argument("--TIRG", type=str2bool, default=True, help="use tirg?")
parser.add_argument("--class_token_position", type=str, default="end", choices=['end', 'middle', 'front'])
parser.add_argument("--csc", type=str2bool, default=False, help="class specific context")
parser.add_argument("--pos", type=str, default='end', help="front/end/middle")
parser.add_argument("--n_ctx", type=int, default=1, help="number of ctx words")
parser.add_argument("--l_ctx", type=int, default=1, help="number of ctx words")
parser.add_argument("--r_ctx", type=int, default=1, help="number of ctx words")
parser.add_argument("--n_ref", type=int, default=1, help="number of ref words")
parser.add_argument("--margin", type=float, default=0.1, help="margin")
parser.add_argument("--beta", type=float, default=0.5, help="beta")
parser.add_argument('--ctx_init', default=False, type=str2bool, help='Use template as init')
# -------------------------------------------------------------------------
# Setting params
parser.add_argument("--domain", type=str, default='single', help="whether domain")
parser.add_argument("--n_shot", type=str, default='2', help="number of shots per cls")
# -------------------------------------------------------------------------
# Optimization / training params
parser.add_argument('--test_size', default=256, type=int, help='Test image size')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size (Adjust base on GPU memory)')
parser.add_argument('--freq', default=100, type=int, help='Test image size')
parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
parser.add_argument('--dropout', default=0.0, type=float, help='Dropout')
parser.add_argument('--epochs', default=1000, type=int, help='Epochs')
parser.add_argument("--load-epoch", type=int, help="load model weights at this epoch for evaluation")
parser.add_argument('--resume', default=False, type=str2bool, help='Continue?')
# Other training environmnet settings
parser.add_argument('--cuda', default=True, type=str2bool, help='Use GPU or CPU')
parser.add_argument('--n_works', default=0, type=int, help='Number of worker threads in dataloader')
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# Load Models
params, unparsed = parser.parse_known_args()
