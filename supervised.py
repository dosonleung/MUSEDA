import os
import json
import argparse
import numpy as np
from collections import OrderedDict
import torch

from src.utils import bool_flag, initialize_exp
from src.models import build_model
from src.trainer import Trainer
from src.evaluation import Evaluator

os.environ['KMP_DUPLICATE_LIB_OK']='True'

VALIDATION_METRIC_SUP = 'precision_at_1-nn'
VALIDATION_METRIC_UNSUP = 'mean_cosine-nn-S2T-10000'

# main
parser = argparse.ArgumentParser(description='Supervised training')
parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
parser.add_argument("--export", type=str, default="txt", help="Export embeddings after training (txt / pth)")

# data
parser.add_argument("--src_lang", type=str, default='en', help="Source language")
parser.add_argument("--tgt_lang", type=str, default='es', help="Target language")
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")
# training refinement
parser.add_argument("--n_refinement", type=int, default=5, help="Number of refinement iterations (0 to disable the refinement procedure)")
# dictionary creation parameters (for refinement)
parser.add_argument("--dico_train", type=str, default="default", help="Path to training dictionary (default: use identical character strings)")
parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
parser.add_argument("--dico_method", type=str, default='csls_knn_10', help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
parser.add_argument("--dico_build", type=str, default='S2T&T2S', help="S2T,T2S,S2T|T2S,S2T&T2S")
parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dictionary generation")
parser.add_argument("--dico_max_rank", type=int, default=10000, help="Maximum dictionary words rank (0 to disable)") #限制构造CSLS词典的最大值
parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")
# reload pre-trained embeddings
parser.add_argument("--src_emb", type=str, default='', help="Reload source embeddings")
parser.add_argument("--tgt_emb", type=str, default='', help="Reload target embeddings")
parser.add_argument("--src_weight", type=str, default='', help="Reload source weights")
parser.add_argument("--tgt_weight", type=str, default='', help="Reload target weights")
parser.add_argument("--weight_scale", type=float, default=0.1, help="eta for PCSLS weight scale")
parser.add_argument("--hist_mu", type=float, default=0.2, help="mu for hist-pcsls for t step")
#parser.add_argument("--full_src_emb", type=str, default='', help="Export source full embedding")
#parser.add_argument("--full_tgt_emb", type=str, default='', help="Export target full embedding")

parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")


# parse parameters
params = parser.parse_args()

# check parameters
assert not params.cuda or torch.cuda.is_available()
assert params.dico_train in ["identical_char", "default"] or os.path.isfile(params.dico_train)
assert params.dico_build in ["S2T", "T2S", "S2T|T2S", "S2T&T2S"]
assert params.dico_max_size == 0 or params.dico_max_size < params.dico_max_rank
assert params.dico_max_size == 0 or params.dico_max_size > params.dico_min_size
#assert os.path.isfile(params.full_src_emb)
#assert os.path.isfile(params.full_tgt_emb)
assert os.path.isfile(params.src_emb)
assert os.path.isfile(params.tgt_emb)
assert params.dico_eval == 'default' or os.path.isfile(params.dico_eval)
assert params.export in ["", "txt", "pth"]

# build logger / model / trainer / evaluator
logger = initialize_exp(params)
# src_emb, tgt_emb, mapping, _ = build_model(params, False) #emb is nn.Embedding
src_emb, tgt_emb, mapping, src_weights, tgt_weights, _ = build_model(params, False)#emb is nn.Embedding
trainer = Trainer(src_emb, tgt_emb, mapping, src_weights, tgt_weights, None, params)
trainer.load_training_dico(params.dico_train)
evaluator = Evaluator(trainer)

# load a training dictionary. if a dictionary path is not provided, use a default
# one ("default") or create one based on identical character strings ("identical_char")
#trainer.load_training_dico(params.dico_train)

# define the validation metric
VALIDATION_METRIC = VALIDATION_METRIC_UNSUP if params.dico_train == 'identical_char' else VALIDATION_METRIC_SUP
logger.info("Validation metric: %s" % VALIDATION_METRIC)

"""
Learning loop for Procrustes Iterative Learning
"""
last_pcsls = None
for n_iter in range(params.n_refinement + 1):

    logger.info('Starting iteration %i...' % n_iter)

    # build a dictionary from aligned embeddings (unless
    # it is the first iteration and we use the init one)
    if n_iter > 0 or not hasattr(trainer, 'dico'):
        last_pcsls = trainer.build_dictionary(last_pcsls)

    # using weight procrustes
    trainer.weight_procrustes()

    to_log = OrderedDict({'n_iter': n_iter})
    evaluator.all_eval(to_log)

    # JSON log / save best model / end of epoch
    logger.info("__log__:%s" % json.dumps(to_log))
    trainer.save_best(to_log, VALIDATION_METRIC)
    logger.info('End of iteration %i.\n\n' % n_iter)


# export embeddings
if params.export:
    # 最好的那一次不一定是现在的mapping
    trainer.reload_best()
    # 根据best_mapping导出
    trainer.export()


