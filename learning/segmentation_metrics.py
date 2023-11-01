import torch

@torch.no_grad()
def calculate_segmentation_metrics(loader, model, device, threshold=0.5, eplison=1e-8):
    """Dice, IoU, Pixel Accuracy"""
    model.eval()

    num_correct = 0
    num_pixels = 0
    dice_score = 0
    mIoU_score = 0 
    
    epoch_history = {
        'dice_score': [],
        'mIoU_score': [],
    }
    
    count = 1 
    
    for image, mask, question_tokens, pad_mask, question_type_ids, answer_ids in loader:
        image = image.to(device)
        mask = mask.to(device)
        
        # THIS IS EXTRA!
        # TODO: Use multimodal masking (i.e: mask image or text completely)
        question_tokens, pad_mask = question_tokens.to(device), pad_mask.to(device)
                    
        answer_predictions, attn_scores, mask_predictions = model(image, question_tokens, pad_mask)
        
        preds = torch.sigmoid(mask_predictions)
        preds = (preds > threshold).float()
        
        # Select non-zero masks
        non_zero_ids = [(mask[batch_id, :, :, :].sum() != 0).item() for batch_id in range(mask.shape[0])]
        mask = mask[non_zero_ids, :, :, :]
        preds = preds[non_zero_ids, :, :, :]
        
        # Pixel accuracy
        num_correct += (preds == mask).sum()
        num_pixels += torch.numel(preds)
        
        # Dice score
        dice_score += (2 * (preds * mask).sum()) / (
            (preds + mask).sum() + eplison
        )
        
        # RECORD DICE SCORE
        epoch_history['dice_score'].append((dice_score/count).item()*100.0)
        
        # intersection is equivalent to True Positive (TP) count
        # union is the mutually inclusive area of all labels & predictions 
        intersection = (preds * mask).sum()
        total = (preds + mask).sum()
        union = total - intersection 

        #IoU = (intersection + smooth)/(union + smooth)
        mIoU_score += intersection/(union + eplison)
        
        # RECORD mIoU SCORE
        epoch_history['mIoU_score'].append((mIoU_score/count).item()*100.0)
        
        count += 1
        
    metrics = {
        'avg_dice_score': (dice_score/len(loader)).item()*100.0,
        'miou_score': (mIoU_score/len(loader)).item()*100.0,
        'epoch_history': epoch_history,
    }
    
    return metrics