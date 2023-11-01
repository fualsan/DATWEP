import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

import os
from PIL import Image

from . import character_mappings as CM

class FloodnetDataset():
    def __init__(
        self, 
        train_images_path,
        mask_images_path,
        img_transforms, 
        num_segmentation_classes, 
        questions_list, 
        answers_dict, 
        char2idx, 
        idx2char, 
        question_type_dict, 
        max_question_len, 
        use_average_len
    ):
        self.train_images_path = train_images_path
        self.mask_images_path = mask_images_path
        self.img_transforms = img_transforms
        self.num_segmentation_classes = num_segmentation_classes
        self.questions_list = questions_list
        self.answers_dict = answers_dict
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.question_type_dict = question_type_dict
        #self.max_question_len = max_question_len
        self.average_question_len = -1 # run optimize_question_len()
        self.use_average_len = use_average_len
        self.added_pad_count = [] # per data point
        self.max_pad_count = -1 # loop over all data to find it
        
        # Optimize question length
        if max_question_len == 'auto':
            self.max_question_len = -1
            self.optimize_question_len()
        else:
            self.max_question_len = max_question_len

    def get_only_answers(self, idx):
        """
        Added for fast computations of answer classes
        
        Used in:
        y_full = []
        
        for i in tqdm(range(len(full_train_dataset))):
            y_full.append(full_train_dataset[i][-1])
        """
        question_dict = self.questions_list[idx]
        
        question_answer = question_dict['Ground_Truth']
        
        answer_id = self.answers_dict[question_answer]

        return answer_id

    def __getitem__(self, idx):
        # Select question from index
        question_dict = self.questions_list[idx]
        
        image_id = question_dict['Image_ID']
        question_sentence = question_dict['Question']
        question_type = question_dict['Question_Type']
        question_answer = question_dict['Ground_Truth']
        
        # Image
        image_full_path = os.path.join(self.train_images_path, image_id)
        mask_id = image_id[:-4] + '_lab.png'
        mask_full_path = os.path.join(self.mask_images_path, mask_id)
        
        # TODO: Maybe replace this with a read function from numpy
        image_raw = Image.open(image_full_path).convert('RGB')
        try:
            mask_raw = Image.open(mask_full_path).convert('P')
            _mask = np.array(mask_raw)
            mask_nan = False
        except:
            #print(f'Mask error at: {idx}')
            mask_nan = True
            _mask = np.zeros(shape=(image_raw.height, image_raw.width), dtype=np.uint8)
        
        _image = np.array(image_raw)

        all_masks = []
        for mask_id in range(self.num_segmentation_classes):
            all_masks.append(np.array(_mask==mask_id, dtype=np.uint8)*255)

        transformed = self.img_transforms(image=_image, masks=all_masks)
        transformed_image = transformed['image']
        transformed_masks = transformed['masks']

        # Stack masks 
        transformed_masks_stacked = np.stack(transformed_masks)

        # Normalize
        image = transformed_image/255.0
        masks = transformed_masks_stacked/255.0

        image = image.transpose(2, 0, 1)
        
        # Input tokens
        question_tokens = CM.convert_to_idx(question_sentence, self.char2idx)
        question_len = len(question_tokens)

        # Padding
        if question_len < self.max_question_len:
            n_to_be_added = self.max_question_len - question_len
            question_tokens += [self.char2idx['<pad>']] * n_to_be_added
            self.added_pad_count.append(n_to_be_added)
            # Max padding count 
            if n_to_be_added > self.max_pad_count:
                self.max_pad_count = n_to_be_added
        else:
            question_tokens = question_tokens[:self.max_question_len]
            self.added_pad_count.append(0)
        
        # Label
        answer_id = self.answers_dict[question_answer]
        
        # Label (optional, pre-training)
        question_type_id = self.question_type_dict[question_type]

        image = torch.Tensor(image) #.permute(2, 0, 1)
        masks = torch.Tensor(masks)
        question_tokens = torch.LongTensor(question_tokens)
        pad_mask = (question_tokens == self.char2idx['<pad>'])
        
        return image, masks, question_tokens, pad_mask, question_type_id, answer_id
        #return data

    def __len__(self):
        return len(self.questions_list)
    
    def optimize_question_len(self):
        __question_len_list = []
        for question_dict in self.questions_list:
            question_sentence = question_dict['Question']
            question_tokens = CM.convert_to_idx(question_sentence, self.char2idx)
            question_len = len(question_tokens)   
            
            __question_len_list.append(question_len)
            
            if question_len > self.max_question_len:
                self.max_question_len = question_len
                
            
        self.average_question_len = sum(__question_len_list) // len(__question_len_list)
        
        print(f'Average question length: {self.average_question_len}')
        print(f'Maximum question length: {self.max_question_len}')
        
        if self.use_average_len:
            self.max_question_len = self.average_question_len
            print(f'Maximum question length (averaged): {self.max_question_len}')
            
    def print_average_pad_count(self):
        avg_pad_count = sum(self.added_pad_count) / len(self.added_pad_count)
        print(f'Average padding count: {avg_pad_count:.2f}')
        print(f'Maximum padding count: {self.max_pad_count}')

    def generate_question_tokens(self, question_sentence):
        question_tokens = CM.convert_to_idx(question_sentence, self.char2idx)
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