import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np


class FloodnetCachedDataset(Dataset):
    "Reads cached dataset which contains both VQA and SEG data"
    def __init__(self, full_cache_file, char2idx, idx2char, max_question_len):

        
        with open(full_cache_file, 'rb') as f:
            # np.memmap(hyperparameters['DATA_FILE'], dtype=np.int32, mode='r+')
            self.full_cache_file = np.load(f, allow_pickle=True) ### USE memmap

        
        #print(self.full_cache_file)
        
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.max_question_len = max_question_len
            
    def __getitem__(self, index):
        data_point = self.full_cache_file[index]
        image = data_point['image']
        masks = data_point['masks']
        question_tokens = data_point['question_tokens']
        question_type_id = data_point['question_type_id']
        answer_id = data_point['answer_id']
        
        image = torch.Tensor(image) #.permute(2, 0, 1)
        masks = torch.Tensor(masks)
        question_tokens = torch.LongTensor(question_tokens)
        pad_mask = (question_tokens == self.char2idx['<pad>'])
        
        return image, masks, question_tokens, pad_mask, question_type_id, answer_id

    def __len__(self):
        return len(self.full_cache_file)
        
    def generate_question_tokens(self, question_sentence):
        question_tokens = convert_to_idx(question_sentence, self.char2idx)
        question_len = len(question_tokens)

        # Padding
        if question_len < self.max_question_len:
            n_to_be_added = self.max_question_len - question_len
            question_tokens += [self.char2idx['<pad>']] * n_to_be_added
        else:
            question_tokens = question_tokens[:self.max_question_len]
    
        question_tokens = torch.LongTensor(question_tokens)
        pad_mask = (question_tokens == self.char2idx['<pad>'])
        
        return question_tokens, pad_mask