
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
from clip import load

class FakeNewsFeaturesDataset(Dataset):
    def __init__(self, info, ann, Train=False):
        self.data_info=info
        self.ann=ann
        self.size= info['train_size'] if Train else info['test_size'] 
        
        self.Train=Train


    def __len__(self):
        return self.size 
    
    def __getitem__(self, idx):
        # 返回每条数据的 input_three, text, image, domain_label, label, 以及 class 标签
        if self.Train:
            event = torch.tensor(self.data_info['event_map'][self.ann['event'][idx].item()])
        else:
            event=self.ann['event'][idx]
        return {
            "img_feat":  self.ann['img_feats'][idx],
            "txt_feat": self.ann['txt_faets'][idx],
            "label":  self.ann['labels'][idx],
            "domain": self.ann['domain'][idx],
            "event":  event,
        }

class FakeNewsDatasetCLIP(Dataset):
    def __init__(self, info, ann, Train=False):
        self.data_info=info
        self.ann=ann
        self.size= info['train_size'] if Train else info['test_size']
        self.Train=Train

        # with torch.no_grad():
        #     self.clip_func, self.transform_func = load('ViT-B/32', device='cuda')
        # for param in self.clip_func.parameters():
        #     param.requires_grad = False

        self.ann=ann
        self.size= info['train_size'] if Train else info['test_size']  
        self.Train=Train

    def __len__(self):
        return self.size 

    def __getitem__(self, idx):
        # 返回每条数据的 input_three, text, image, domain_label, label, 以及 class 标签
        if self.Train:
            event = torch.tensor(self.data_info['event_map'][self.ann['event'][idx].item()])
        else:
            event=self.ann['event'][idx]
        return {
            "img_feat":  self.ann['img_feats'][idx],
            "txt_feat": self.ann['txt_faets'][idx],
            "label":  self.ann['labels'][idx],
            "domain": self.ann['domain'][idx],
            "event":  event,
            "image": self.ann['images'][idx],
            "text": self.ann['texts'][idx],
            "idx": idx,
        }


def load_feats_datasets(file_path="/home/tj/multi_fake/multi_fake/wenbo/feats_data/TNM2P_feats_datasets.pt", batch_size=32, DEBUG=False, test_shuffle=False):
    
    raw_data = torch.load(file_path)

    data_info = raw_data['info']
    train_sets = raw_data['train_sets']
    test_sets = raw_data['test_sets']

    sorted_event_ids = sorted(list(data_info['train_event_labelset']))
    # Create mapping from original IDs to sequential indices
    event_mapping = {original_id: index for index, original_id in enumerate(sorted_event_ids)}
    data_info['event_map']=event_mapping
    data_info['event_num'] = len(data_info['train_event_labelset'])
    
    train_datasets = FakeNewsFeaturesDataset(data_info, train_sets, Train=True)
    test_datasets = FakeNewsFeaturesDataset(data_info, test_sets)

    if DEBUG:
        train_dataloader = DataLoader(test_datasets, batch_size=batch_size)
    else:
        train_dataloader = DataLoader(train_datasets, batch_size=batch_size)

    test_dataloader = DataLoader(test_datasets, batch_size=batch_size, shuffle=test_shuffle)

    return data_info, train_dataloader, test_dataloader

def load_CLIP_datasets(file_path="/home/tj/multi_fake/multi_fake/wenbo/feats_data/TNM2P_feats_datasets.pt", batch_size=32, DEBUG=False, test_shuffle=False):
    
    raw_data = torch.load(file_path)

    data_info = raw_data['info']
    train_sets = raw_data['train_sets']
    test_sets = raw_data['test_sets']

    sorted_event_ids = sorted(list(data_info['train_event_labelset']))
    # Create mapping from original IDs to sequential indices
    event_mapping = {original_id: index for index, original_id in enumerate(sorted_event_ids)}
    data_info['event_map']=event_mapping
    data_info['event_num'] = len(data_info['train_event_labelset'])
    
    train_datasets = FakeNewsDatasetCLIP(data_info, train_sets, Train=True)
    test_datasets = FakeNewsDatasetCLIP(data_info, test_sets)


    train_dataloader = DataLoader(train_datasets, batch_size=batch_size)

    test_dataloader = DataLoader(test_datasets, batch_size=batch_size)

    return data_info, train_dataloader, test_dataloader
