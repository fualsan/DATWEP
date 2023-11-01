import torch

from . import dynamic_weights as dw

def adjust_weights(pred, target, weight, lr_weight_adjust):
    
    for t in weight.keys():
        grad = dw.grad_wrt_weight(
            pred=pred.detach().clone(), 
            target_wrt=t, 
            target=target,
            batch_size=pred.shape[0], 
            weight=weight
        )
        
        if torch.isnan(grad):
            print(f'WARNING GRADIENT IS NAN! (FOR: {t})')
            
        # *************
        # NOTE: Some gradients are exploding and causing loss for some classes to be negative
        # this in return cheats the weights and does not reduce the loss for some classes
        # *************
        grad = torch.clamp(grad, min=-1.5, max=1.5)
        #grad = torch.clamp(grad, min=-2.5, max=2.5) # MODIFIED!
        
        #print(f'grad: {grad}')
        # Apply grads
        weight[t] = weight[t] - (lr_weight_adjust * grad)
        #weight[t] = weight[t] + (lr_weight_adjust * grad)