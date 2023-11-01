import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class PositionalEncoding(nn.Module):
    'FROM: https://nlp.seas.harvard.edu/2018/04/03/attention.html'
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        # Dim = [1, max_len, d_model]
        self.register_buffer('pe', pe)

        self.pe_tensor = torch.tensor(
            self.pe[:, :, :], requires_grad=False
        )
    
    def forward(self, x):
        device = x.device
        x = x + self.pe_tensor[:, :x.size(1), :].to(device)
        return self.dropout(x)


class VQAClassifier(nn.Module):
    def __init__(
        self, 
        n_char_tokens, 
        emb_dim, 
        pos_emb_dropout, 
        max_question_len, 
        n_vqa_answer_classes, 
        #n_vqa_question_type_classes,
        unet, 
        unet_features_dim, 
        image_features_pool_dim,
        freeze_unet,
        text_dropout_prob,
        combined_dropout_prob,
    ):
        super().__init__()
        self.n_char_tokens = n_char_tokens
        self.emb_dim = emb_dim
        self.pos_emb_dropout = pos_emb_dropout
        self.max_question_len = max_question_len
        self.n_vqa_answer_classes = n_vqa_answer_classes
        #self.n_vqa_question_type_classes = n_vqa_question_type_classes
        self.text_features_dim = 255
        self.unet_features_dim = unet_features_dim # Sizes of all UNet filters
        self.image_features_pool_dim = image_features_pool_dim # Sizes of all UNet images
        self.freeze_unet = freeze_unet
        self.text_dropout_prob = text_dropout_prob
        self.combined_dropout_prob = combined_dropout_prob

        self.emb = nn.Embedding(n_char_tokens, emb_dim)
        self.pos_emb = PositionalEncoding(d_model=emb_dim, dropout=pos_emb_dropout, max_len=max_question_len)
        self.fc_attn = nn.Linear(emb_dim, 1)
        self.fc1 = nn.Linear(emb_dim*max_question_len, self.text_features_dim)
        
        self.softmax = nn.Softmax(dim=1)
        self.flatten = nn.Flatten()
        self.text_dropout = nn.Dropout(text_dropout_prob)
        self.combined_dropout = nn.Dropout(combined_dropout_prob)
        
        # Image encoder
        self.image_encoder = unet
        self.image_features_dim = sum(unet_features_dim)
        self.image_features_pool_dim = image_features_pool_dim
        
        # Freeze image encoder
        if freeze_unet:
            self.freeze_image_encoder()
        
        # Question classifier
        total_features = self.image_features_dim+self.text_features_dim
        print(f'Total (image+text) features: {total_features}')
        
        # Answer classification
        self.fc_answer1 = nn.Linear(total_features, total_features//2)
        self.fc_answer2 = nn.Linear(total_features//2, n_vqa_answer_classes)
        
        """
        # Question type classification
        self.fc_question_type1 = nn.Linear(total_features, total_features//2)
        self.fc_question_type2 = nn.Linear(total_features//2, n_vqa_question_type_classes)
        """
        
    def freeze_layer(self, layer):
        for p in layer.parameters():
            p.requires_grad = False
    
    def freeze_image_encoder(self):
        for p in self.image_encoder.parameters():
            p.requires_grad = False
            
    def unfreeze_image_encoder(self):
        for p in self.image_encoder.parameters():
            p.requires_grad = True
    
    def freeze_emb_layers(self):
        self.freeze_layer(self.emb)
        self.freeze_layer(self.fc_attn)
        self.freeze_layer(self.fc1)
        
        print('Embbeding layers are frozed!')
                
    """
    def freeze_qt_layers(self):
        self.freeze_layer(self.fc_question_type1)
        self.freeze_layer(self.fc_question_type2)

        print('Question classification is frozed!')
    """
        
    def forward(self, image, text, mask):        
        emb_out = self.emb(text)
        
        # Positional encodings 
        # (BATCH, SEQ, EMB_SIZE)
        emb_out = self.pos_emb(emb_out)
        
        # Apply attention
        # (BATCH, SEQ, 1)
        attn_scores = self.softmax(self.fc_attn(emb_out))
        
        # Padding mask 
        # (BATCH, SEQ)
        attn_scores = attn_scores.squeeze(2) * (~mask).int()
        
        # Apply attention
        emb_out = emb_out * attn_scores.unsqueeze(2)

        # Feed-forward, text feature
        #text_features = F.relu(self.fc1(self.flatten(emb_out)), inplace=True)
        text_features = self.fc1(self.flatten(emb_out))
        # Apply dropout to text features
        text_features = self.text_dropout(text_features)
        
        # Image feature
        # final_output is predicted segmentation mask
        final_output, (x_res1, x_res2, x_res3) = self.image_encoder(image)
        
        # Pool the outputs of images at each scale
        # CAREFUL WITH SQUEEZE() FUNCTION!
        x_res1 = F.avg_pool2d(x_res1, kernel_size=self.image_features_pool_dim[0]).squeeze(2).squeeze(2)
        x_res2 = F.avg_pool2d(x_res2, kernel_size=self.image_features_pool_dim[1]).squeeze(2).squeeze(2)
        x_res3 = F.avg_pool2d(x_res3, kernel_size=self.image_features_pool_dim[2]).squeeze(2).squeeze(2)

        #print(f'{x_res1.shape=}, {x_res2.shape=}, {x_res3.shape=}, {text_features.shape=}')
        
        # Concat features (image+text)
        combined_features = torch.cat((x_res1, x_res2, x_res3, text_features), dim=1)
        # Apply dropout to combined features
        combined_features = self.combined_dropout(combined_features)
        
        # Question type classifier
        #question_type_probs = F.relu(self.fc_question_type1(combined_features), inplace=True)
        #question_type_probs = self.fc_question_type2(question_type_probs)
        
        # MLP multimodal classifier for both image and text
        answer_probs = F.relu(self.fc_answer1(combined_features), inplace=True)
        answer_probs = self.fc_answer2(answer_probs)
        
        #return answer_probs, question_type_probs, attn_scores, final_output
        return answer_probs, attn_scores, final_output