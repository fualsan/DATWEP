import torch

def count_errors(pred, target):
    n_errors = 0
    n_errors_per_class = {k.item():{'n_errors': 0, 'total_count': 0} for k in target.unique()}

    for p, t in zip(pred, target):
        if p.argmax() != t:
            n_errors +=1
            n_errors_per_class[t.item()]['n_errors'] += 1
            n_errors_per_class[t.item()]['total_count'] += 1
        else:
            n_errors_per_class[t.item()]['total_count'] += 1
        
    print('    Errors per class (n_errors/total_count):')
    print('    ', end='')
    for _class, _error_dict in n_errors_per_class.items():
        print(f'{_class}: ({_error_dict["n_errors"]}/{_error_dict["total_count"]})', end=' ')
    print(f'--> total: ({n_errors}/{len(target)})')


def softmax(logit):
    logit_exp = torch.exp(logit)
    partition = logit_exp.sum(1, keepdims=True)
    return logit_exp / partition


def nll_weighted_mean(pred_probs, batch_size, target, weight):
    # Select (one shot encoded) and log the probabilities, WITH NEGATIVE
    pred_probs_log = -(pred_probs[range(batch_size), target].log())
    # Weight each class
    weighs_per_logit_out = torch.tensor([weight[t.item()] for t in target])
    pred_probs_log_weighted = pred_probs_log * weighs_per_logit_out
    return pred_probs_log_weighted.sum() / weighs_per_logit_out.sum()
    
    
def nll_weighted_sum(pred_probs, batch_size, target, weight):
    # Select (one shot encoded) and log the probabilities, WITH NEGATIVE
    pred_probs_log = -(pred_probs[range(batch_size), target].log())
    # Weight each class
    weighs_per_logit_out = torch.tensor([weight[t.item()] for t in target])
    pred_probs_log_weighted = pred_probs_log * weighs_per_logit_out
    return pred_probs_log_weighted.sum()


def CELoss(pred, batch_size, target, weight, reduction='mean'):
    pred_probs = softmax(pred)
    if reduction == 'mean':
        return nll_weighted_mean(pred_probs, batch_size, target, weight)
    elif reduction == 'sum':
        return nll_weighted_sum(pred_probs, batch_size, target, weight)
    else:
        print(f"ERROR! {reduction} is not a valid option!")


def loss_per_class(pred, batch_size, target, weight, verbose=True):
    weighs_per_logit_out = torch.tensor([weight[t.item()] for t in target])
    denom = weighs_per_logit_out.sum()
    
    _loss_per_class = {}
    
    for selected_target in target.unique():
        _target = target[target == selected_target]
        _pred = pred[target == selected_target, :]
        _batch_size = (target == selected_target).sum()
        
        loss = CELoss(_pred, _batch_size, _target, weight, reduction='sum')
        loss = loss/denom
        
        if verbose:
            print(f'Loss: {loss:.4f} for target: {selected_target.item()}')
            
        _loss_per_class[selected_target.item()] = loss
        
    return _loss_per_class
    
    
def grad_wrt_weight(pred, target_wrt, target, batch_size, weight):
    pred_probs = softmax(pred)
    pred_probs = pred_probs[range(batch_size), target]
    
    weighs_per_logit_out = torch.tensor([weight[t.item()] for t in target])
    weights_sum = weighs_per_logit_out.sum()
        
    # a
    occurance = len(target[target == target_wrt])
    a1_1 = -occurance / (weights_sum**2)
    a1_2 = (-weighs_per_logit_out * (pred_probs.log())).sum()
    a = a1_1 * a1_2

    # print(pred_probs.log())
    # if torch.isnan(a):
    #     print(pred_probs)    
    #     quit()

    # b
    b1_1 = 1 / weights_sum
    b1_2 = -(pred_probs[target == target_wrt].log().sum())
    b = b1_1 * b1_2
    
    #print(f'******* a: {a.item():.4f}, b: {b.item():.4f}')
    #print(f'******* a1_1: {a1_1.item():.4f}, a1_2: {a1_2.item():.4f}')
    return a + b


    