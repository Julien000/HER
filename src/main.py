

import numpy as np
import warnings
import torch
import os
import argparse
import pickle
from data.datasets import load_feats_datasets 
import torch
from torch.utils.data import DataLoader
from module import LODA_TRAINER

warnings.filterwarnings('ignore')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
seed = 2021
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


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

def load_raw_datasets(feats_data_dir, batch_size=32):
    
    train_dataset = pickle.load(open(f'{feats_data_dir}_train_512.pkl', 'rb'))
    validate_dataset = pickle.load(open(f'{feats_data_dir}_test_512.pkl', 'rb'))
    print(len(train_dataset), len(validate_dataset))

    print("dataset load ok")
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size)
    print('process data  Loader success')
    return '', train_loader, validate_loader


def train_val_test(args):
   
    config = Config(args)
    feats_data_dir =config.data_root+f"{config.dataset}_feats_datasets.pt"
    print("dataset load ok")
    data_info, train_loader, validate_loader = load_feats_datasets(feats_data_dir, batch_size=config.batch_size)

    config.data_info = data_info
    config.event_num = data_info['event_num']

    # Start training
    trainer=LODA_TRAINER(config, train_loader)
    print("=================  Staring Training  =================")
    trainer.fit(train_loader, validate_loader, config.epochs)
    print("=================  Training ending   =================")


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
    # parser.add_argument('--data_root', default="/home/tj/multi_fake/multi_fake/wenbo/feats_data/",  help="dataset root directory")
    parser.add_argument('--data_root', default="/home/tj/multi_fake/multi_fake/wenbo/feats_dataV2/",  help="dataset root directory")
    parser.add_argument('--text_only', action='store_true', help="Whether use text only")
    parser.add_argument('--show_TSNE', action='store_true', help="show_TSNE")
    parser.add_argument('--workdir', default="workdir", help="Setting workingdir!")
    # constant settings
    # constant settings
    parser.add_argument('--seed', default=2021, type=int)
    parser.add_argument('--test_shuffle', default=False, type=bool)
    parser.add_argument('--maskrate', default=0.0, type=float)
    parser.add_argument('--maskbranch', default='text', type=str)
    # debug settings
    parser.add_argument('--debug', action='store_true', help="Whether use training!")
    args = parser.parse_args()

    train_val_test(args)