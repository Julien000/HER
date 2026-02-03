

import numpy as np
import warnings
import torch
import os
import argparse

from data.datasets import load_feats_datasets 
from module import LODA_TRAINER



class Config():
    def __init__(self,args):
        self.batch_size = args.batch_size
        self.epochs = args.max_epoch
        self.bert_path = "./bert-base-english/"
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.event_num = 30
        self.root_dir= "/home/tj/multi_fake/multi_fake/wenbo"
        self.data_root=args.data_root
        self.model=args.model
        dir=f"{args.workdir}/debug" if args.debug else args.workdir
        self.out_dir= f"{self.root_dir}/{dir}/{args.dataset}/{self.model}/"
        self.dataset= args.dataset
        self.show_TSNE= args.show_TSNE
        self.args= args
        os.makedirs(self.out_dir, exist_ok=True)
        self.log_dir= f"{self.out_dir}log/"
        os.makedirs(self.log_dir, exist_ok=True)


def train_val_test(args):
   
    config = Config(args)
    feats_data_dir =config.data_root+f"{config.dataset}_feats_datasets.pt"
    # feats_data_dir =config.data_root+f"{config.dataset}_feats_TJ_datasets.pt"
    print("dataset load ok")
    data_info, train_loader, validate_loader = load_feats_datasets(feats_data_dir, batch_size=config.batch_size, test_shuffle=config.args.test_shuffle)
    config.data_info = data_info
    config.event_num = data_info['event_num']

    # Start training
    trainer=LODA_TRAINER(config, train_loader)
    print("=================  Loading Model  =================")
    trainer.load_model_weight(config.args.ckpt)
    print("=================  Staring Testing  =================")
    trainer.Testing( validate_loader )
    print("=================  Testing ending   =================")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    # training settings
    parser.add_argument('--max_epoch', default=50, type=int)
    parser.add_argument('--model', default="Our_Model", help="choose model for testing")
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--ckpt', default="/lwb/MFND/EventDomain/results/ESM_TMN2P_train&test/last.ckpt", help="Checkpoint path")

    # datasets settings
    # parser.add_argument('--dataset', default='TNM2P', choices=["PNM2T", "TNM2P", "TNP2M", "TPM2N"],  help="Dataset name")
    parser.add_argument('--dataset', default='TNM2P',  help="Dataset name") 
    parser.add_argument('--data_root', default="/home/tj/multi_fake/multi_fake/wenbo/feats_dataV2/",  help="dataset root directory")
    parser.add_argument('--text_only', action='store_true', help="Whether use text only")
    parser.add_argument('--show_TSNE', action='store_true', help="show_TSNE")
    parser.add_argument('--workdir', default="workdir", help="Setting workingdir!")
    
    # constant settings
    parser.add_argument('--seed', default=2021, type=int)
    parser.add_argument('--test_shuffle', default=True, type=bool)
    parser.add_argument('--maskrate', default=0.0, type=float)
    parser.add_argument('--maskbranch', default='text', type=str)

    # debug settings
    parser.add_argument('--debug', action='store_true', help="Whether use training!")
    args = parser.parse_args()
    
    warnings.filterwarnings('ignore')
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_val_test(args)