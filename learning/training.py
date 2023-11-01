import torch
import torch.nn as nn
import torch.nn.functional as F

from .segmentation_metrics import calculate_segmentation_metrics

from tqdm import tqdm
import numpy as np

def train_iter(
    epoch, 
    dataloader, 
    model, 
    device, 
    optimizer, 
    answer_loss_fn, 
    seg_loss_fn, 
    alpha, 
    adjust_alpha, 
    epsilon, 
    _adjust_weights, # not to be confused with the function
    adjust_weights_fn,
    dynamic_weights,
    adjust_weights_lr
):
    print(f'train_iter current alpha: {alpha:.4f}')
    
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    
    model.train()
    pbar = tqdm(dataloader, unit='batch')
    pbar.set_description(f'Epoch: {epoch}, Train')

    training_loss = 0.0
    n_answer_correct = 0
    #n_question_type_correct = 0
    acc_count = 0
    count = 0
    
    epoch_history = {
        'seg_loss': [],
        'vqa_loss': [],
        'vqa_acc': [], # SEG accuracy is in another function
    }
    
    for image, masks, question_tokens, pad_mask, question_type_ids, answer_ids in pbar:
        # SEG
        image, masks = image.to(device), masks.to(device) 
        # VQA
        question_tokens, pad_mask, answer_ids = question_tokens.to(device), pad_mask.to(device), answer_ids.to(device)
        
        answer_predictions, attn_scores, mask_predictions = model(image, question_tokens, pad_mask)
        # Answer loss
        answer_loss = F.cross_entropy(
            answer_predictions, 
            answer_ids, 
            weight=torch.tensor(list(dynamic_weights.values()), device=device),
        )

        # RECORD VQA LOSS
        epoch_history['vqa_loss'].append(answer_loss.item())

        # Segmentation loss
        seg_loss = seg_loss_fn(mask_predictions, masks)

        # RECORD SEG LOSS
        epoch_history['seg_loss'].append(seg_loss.item())

        total_loss = (alpha*(answer_loss)) + ((1.0-alpha)*seg_loss)

        #### UPDATE ALPHA #####
        #######################
        if adjust_alpha:
            regularization_term = (alpha-sigmoid(alpha)) / np.abs((alpha-sigmoid(alpha))) * (1-sigmoid(alpha)+sigmoid(alpha)**2) 
            dALPHA_dTotalLoss = (answer_loss.item()-seg_loss.item()) + 0.75*regularization_term            
            alpha = alpha - (epsilon*(dALPHA_dTotalLoss))
        
            if alpha < 0.0001:
                alpha = 0.0010
            if alpha > 0.9999:
                alpha = 0.9990
            
        #######################
        #######################
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        
        ####### ADJUST WEIGHTS ###########
        ##################################
        if _adjust_weights:
            adjust_weights_fn(
                pred=answer_predictions.detach().cpu(), 
                target=answer_ids.cpu(), 
                weight=dynamic_weights, 
                lr_weight_adjust=adjust_weights_lr
            )
        ##################################
        ##################################
        
        training_loss += total_loss.item()
        
        # Answer accuracy
        pred_answer_ids = answer_predictions.argmax(dim=-1)
        
        acc_count += answer_ids.size(0) # The same for question type predictions (batch size)
        n_answer_correct += (answer_ids == pred_answer_ids).sum().item()
        
        count += 1
        
        # Calculate averages
        avg_train_loss = training_loss/count
        avg_train_answer_acc = (100 * n_answer_correct) / acc_count
        
        epoch_history['vqa_acc'].append(avg_train_answer_acc)
        
        pbar.set_postfix_str(f'Loss: {avg_train_loss:.4f}, Acc answer: %{avg_train_answer_acc:.2f}')
        
        
    segmentation_metrics = calculate_segmentation_metrics(dataloader, model, device, threshold=0.5, eplison=1e-8)
    
    metrics = {
        'avg_total_loss': avg_train_loss,
        'avg_answer_acc': avg_train_answer_acc,
        'avg_dice_score': segmentation_metrics['avg_dice_score'],
        'miou_score': segmentation_metrics['miou_score'],
        'epoch_history': epoch_history | segmentation_metrics['epoch_history'], # merge two dicts together
        'alpha': alpha,
        'epsilon': epsilon, 
    }
    
    return metrics


@torch.no_grad()
def validate_iter(epoch, dataloader, model, device, answer_loss_fn, seg_loss_fn, alpha, dynamic_weights):

    print(f'validate_iter current alpha: {alpha:.4f}')
    
    model.eval()
    pbar = tqdm(dataloader, unit='batch')
    pbar.set_description(f'Epoch: {epoch}, Val')

    test_loss = 0.0
    n_answer_correct = 0
    #n_question_type_correct = 0
    acc_count = 0
    count = 0
    
    epoch_history = {
        'seg_loss': [],
        'vqa_loss': [],
        'vqa_acc': [], # SEG accuracy is in another function
    }
    
    for image, masks, question_tokens, pad_mask, question_type_ids, answer_ids in pbar:
        # SEG
        image, masks = image.to(device), masks.to(device) 
        # VQA
        question_tokens, pad_mask, answer_ids = question_tokens.to(device), pad_mask.to(device), answer_ids.to(device)
        
        ### Forward pass only with not gradient calculations ###
        ########################################################

        #answer_predictions, question_type_predictions, attn_scores, mask_predictions = model(image, question_tokens, pad_mask)
        answer_predictions, attn_scores, mask_predictions = model(image, question_tokens, pad_mask)
        # Answer loss
        answer_loss = F.cross_entropy(
            answer_predictions, 
            answer_ids, 
            weight=torch.tensor(list(dynamic_weights.values()), device=device),
        )

        # RECORD VQA LOSS
        epoch_history['vqa_loss'].append(answer_loss.item())

        # Segmentation loss
        seg_loss = seg_loss_fn(mask_predictions, masks)

        # RECORD SEG LOSS
        epoch_history['seg_loss'].append(seg_loss.item())

        total_loss = (alpha*(answer_loss)) + ((1.0-alpha)*seg_loss)                
        
        test_loss += total_loss.item()
        
        # Answer accuracy
        pred_answer_ids = answer_predictions.argmax(dim=-1)

        acc_count += answer_ids.size(0) # The same for question type predictions (batch size)
        n_answer_correct += (answer_ids == pred_answer_ids).sum().item()
        
        count += 1
        #######################################################
        #######################################################
        
        # Calculate averages
        avg_test_loss = test_loss/count
        avg_test_answer_acc = (100 * n_answer_correct) / acc_count
        
        epoch_history['vqa_acc'].append(avg_test_answer_acc)
        
        pbar.set_postfix_str(f'Loss: {avg_test_loss:.4f}, Acc answer: %{avg_test_answer_acc:.2f}')
        
        
    segmentation_metrics = calculate_segmentation_metrics(dataloader, model, device, threshold=0.5, eplison=1e-8)
    
    metrics = {
        'avg_total_loss': avg_test_loss,
        'avg_answer_acc': avg_test_answer_acc,
        'avg_dice_score': segmentation_metrics['avg_dice_score'],
        'miou_score': segmentation_metrics['miou_score'],
        'epoch_history': epoch_history | segmentation_metrics['epoch_history'], # merge two dicts together
    }
    
    return metrics





