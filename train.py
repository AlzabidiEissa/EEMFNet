import time
import json
import os
import wandb
import logging
import globals

import torch
import torch.nn.functional as F
import numpy as np
from typing import List
from sklearn.metrics import roc_auc_score, average_precision_score
from metrics import compute_pro, trapezoid
# from main import run
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, roc_curve

import torch.nn.functional as F


_logger = logging.getLogger('train')

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# def bce_iou_loss(pred, mask):
#     weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
#     wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
#     bce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))
#     pred  = torch.sigmoid(pred)
#     inter = ((pred*mask)*weit).sum(dim=(2,3))
#     union = ((pred+mask)*weit).sum(dim=(2,3))
#     iou  = 1-(inter+1)/(union-inter+1)
#     return (bce+iou).mean()
#     # return iou.mean()

# mIoU
def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

def training(model, trainloader, validloader, criterion, optimizer, scheduler, num_training_steps: int = 1000, loss_weights: List[float] = [0.6, 0.4],
             log_interval: int = 1, eval_interval: int = 1, savedir: str = None, use_wandb: bool = False, use_tpu: bool = False,
             device: str ='cpu',model_size: str = None) -> dict:
 

    impressions = np.array([0, 0, 0, 0, 0, 0])

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    l1_losses_m = AverageMeter()
    focal_losses_m = AverageMeter()
    bce_iou_losses_m = AverageMeter()

    # criterion
    l1_criterion, focal_criterion = criterion
    # l1_criterion, focal_criterion, bce_iou_criterion = criterion
    l1_weight, focal_weight = loss_weights

    # set train mode
    model.train()

    # set optimizer
    optimizer.zero_grad()

    # training
    best_score = 0
    best_score_AUC = 0
    step = 0
    globals.rate_p = 0.0
    train_mode = True
    epoch = 0
    while train_mode:   
    # for epoch in range(num_training_steps):   
        model.train()
        
        train_loss = 0.0

        # Gradient Accumulation
        accumulation_steps = 4

        for step, (inputs, masks, targets, maskmodes) in enumerate(trainloader):
            end = time.time()
            # batch
            inputs, masks, targets, maskmodes[0] = inputs.to(device), masks.to(device), targets.to(device), maskmodes[0].to(device)

            # print(maskmodes[0], maskmodes[1])
            inx_maskmodes = maskmodes[0].detach()[1::2].tolist()
            inxRep = {value:inx_maskmodes.count(value) for value in set(inx_maskmodes)}
            # print(list(inxRep.keys())==[5])
            if list(inxRep.keys())!=[6]:
              impressions[list(inxRep.keys())] = impressions[list(inxRep.keys())] + list(inxRep.values())
            # print(impressions)


            data_time_m.update(time.time() - end)

         
            # freq = torch.fft.fft2(inputs, norm='ortho')
            # freq = torch.fft.fftshift(freq, dim=(-2, -1))
            # # Combine real and imaginary parts
            # freq_combined = freq.real + 1j * freq.imag

            # # Inverse Fourier Transform
            # freq_inputs = torch.fft.ifft2(freq_combined, norm='ortho').real            
            # random_index = np.random.choice(range(3))
            # outputs = model(inputs, freq_inputs[:,random_index,:].unsqueeze(1))
            outputs = model(inputs)
            

            # freq = torch.fft.fft2(inputs, norm='ortho')
            # freq = torch.fft.fftshift(freq, dim=(-2, -1))
            # # Combine real and imaginary parts
            # freq_combined = freq.real + 1j * freq.imag

            # # Inverse Fourier Transform
            # freq_inputs = torch.fft.ifft2(freq_combined, norm='ortho').real
            # # print(inputs[0][0][0], freq_inputs[0][0][0])
            # outputs = model(inputs, freq_inputs)

            # bce_iou_loss = torch.tensor(0)            # bce_iou_criterion(outputs[:,1,:], masks)
        

            # weit  = 1+5*torch.abs(F.avg_pool2d(masks.float(), kernel_size=31, stride=1, padding=15)-masks.float())
            # wbce  = F.binary_cross_entropy_with_logits(outputs[:,1,:], masks.float(), reduction='none')
            # bce  = (weit*wbce).sum(dim=(1,2))/weit.sum(dim=(1,2))



            # bce_iou_loss = bce_iou_criterion(outputs[:,1,:], masks)
            # focal_loss = focal_criterion(outputs, masks) 
            # bce_iou_loss = bce.mean()

            outputs = F.softmax(outputs, dim=1)
            
            focal_loss = focal_criterion(outputs, masks) 
            # bce_iou_loss  = dice_loss(outputs[:,1,:].unsqueeze(1), masks.unsqueeze(1))
            # inter = ((outputs[:,1,:]*masks)*weit).sum(dim=(1,2))
            # union = ((outputs[:,1,:]+masks)*weit).sum(dim=(1,2))
            # iou  = 1-(inter+1)/(union-inter+1)
            # bce_iou_loss = (bce+iou).mean()           
            # bce_iou_loss = iou.mean()       
            
            
            # focal_loss = focal_criterion(outputs, masks) 

            # bce_iou_loss = bce_iou_loss(torch.argmax(outputs, dim=1).float().unsqueeze(1), masks.float().unsqueeze(1))
            l1_loss = l1_criterion(outputs[:,1,:], masks)
            # bce_iou_loss = bce_iou_criterion(outputs[:,1,:], masks)
            # l1_loss = l1_criterion(torch.argmax(outputs, dim=1).float(), masks.float())
            bce_iou_loss = torch.tensor(0)  
            # bce_iou_loss = bce_iou_criterion(outputs[:,1,:], masks)

            # EEMFNet_loss = EEMFNet_criterion(outputs[:,1,:], masks)
            # EEMFNet_loss = EEMFNet_criterion(outputs, masks)
            # EEMFNet_loss = torch.tensor(0)

            # loss = bce_iou_loss(outputs, masks.float().unsqueeze(1))
            # loss = 0.90 * ((l1_weight * l1_loss) + (focal_weight * focal_loss)) + 0.10 * bce_iou_loss
            loss =(l1_weight * l1_loss) + (focal_weight * focal_loss)

            if torch.isnan(loss).any():
                print(".......................Loss is NaN, stopping training.....................")
                return
            
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


            if (step+1) % accumulation_steps == 0 or (step + 1) == len(trainloader):
                    # update weight
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    _logger.warning('Epoch[{:>4d}/{}] '
                        'Step:{}, '
                        'Loss: {:>6.4f} '
                        'L1 Loss: {:>6.4f} '
                        'Focal Loss: {:>6.4f} '
                        'bce_iou Loss: {:>6.4f} '
                        'LR: {:.3e} '
                        'batch_time: {:.3f}s '
                        'data_time: {:.3f}s '.format(
                        epoch+1, num_training_steps,
                        step+1,
                        losses_m.avg,
                        l1_losses_m.avg,
                        focal_losses_m.avg,
                        bce_iou_losses_m.avg,
                        optimizer.param_groups[0]['lr'],
                        batch_time_m.avg,
                        data_time_m.avg))
            
            # optimizer.step()
            # optimizer.zero_grad()

            train_loss += loss.item()


            # log loss
            l1_losses_m.update(l1_loss.item())
            focal_losses_m.update(focal_loss.item())
            bce_iou_losses_m.update(bce_iou_loss.item())
            losses_m.update(loss.item())

            batch_time_m.update(time.time() - end)

            # wandb
            # if use_wandb:
            #     wandb.log({
            #         'lr':optimizer.param_groups[0]['lr'],
            #         'train_focal_loss':focal_losses_m.val,
            #         'train_l1_loss':l1_losses_m.val,
            #         # 'train_bce_iou_loss':bce_iou_losses_m.val,
            #         'train_loss':losses_m.val
            #     },
            #     step=step)

            # if (step+1) % log_interval == 0 or step == 0:
                # _logger.info('TRAIN [{:>4d}/{}] '
            # _logger.warning('Epoch[{:>4d}/{}] '
            #             'Step:{}, '
            #             'Loss: {:>6.4f} '
            #             'L1 Loss: {:>6.4f} '
            #             'Focal Loss: {:>6.4f} '
            #             'bce_iou Loss: {:>6.4f} '
            #             'LR: {:.3e} '
            #             'batch_time: {:.3f}s '
            #             'data_time: {:.3f}s '.format(
            #             epoch+1, num_training_steps,
            #             step+1,
            #             losses_m.avg,
            #             l1_losses_m.avg,
            #             focal_losses_m.avg,
            #             bce_iou_losses_m.avg,
            #             optimizer.param_groups[0]['lr'],
            #             batch_time_m.avg,
            #             data_time_m.avg))
            

            # del inputs, masks, targets, maskmodes, outputs, l1_loss, focal_loss, loss
            # torch.cuda.empty_cache()

        train_loss /= len(trainloader)
        # Loss.append(loss.item())
        # train_losses.append(train_loss)
        # if ((step+1) % eval_interval == 0 and step != 0) or (step+1) == num_training_steps:
        eval_metrics, fps = evaluate(
            model        = model, 
            dataloader   = validloader, 
            device       = device
        )
        model.train()

        eval_log = dict([(f'eval_{k}', v) for k, v in eval_metrics.items()])

        # wandb
        if use_wandb:
            # wandb.log({'train_loss':train_loss}, step=epoch)
            wandb.log({
                    'lr':optimizer.param_groups[0]['lr'],
                    'train_focal_loss':focal_losses_m.avg,
                    'train_l1_loss':l1_losses_m.avg,
                    'train_bce_iou_loss':bce_iou_losses_m.avg,
                    'train_loss':losses_m.avg
                }, step=epoch)            
            wandb.log(eval_log, step=epoch)


        # checkpoint
        globals.rate_p = globals.rate_p + 0.01

        if best_score < np.mean(list(eval_metrics.values())):
        # if best_score < np.mean(list(eval_metrics.values())[:2]):
            
            if  num_training_steps-step < 10:
                num_training_steps = num_training_steps + 10
            c = 0
            globals.globalimpressions1 = impressions
            # print("globalimpressions and rate", [impressions,globals.rate_p] )

            # save best score
            state = {'best_step':epoch+1}
            state.update({'model size:': model_size})
            state.update({'inference speed (s):':f"{fps:.3f} s"})
            
            state.update(eval_log)
            state.update({'globalimpressions1':impressions.tolist()})
            # print(send_value()[0][0].tolist())
            # json.dump(state, open(os.path.join(savedir, f"best_score{step}.json"),'w'), indent='\t')
            json.dump(state, open(os.path.join(savedir, 'best_score.json'),'w'), indent='\t')

            # save best model
            # torch.save(model.state_dict(), os.path.join(savedir, f'best_model{step}.pt'))
            torch.save(model.state_dict(), os.path.join(savedir, 'best_model.pt'))

            # _logger.info('Best Score {:.3%} to {:.3%}, inference speed {:.3f}(s):'.format(best_score, np.mean(list(eval_metrics.values())[:2]), fps))
            _logger.warning('Best Score {:.3%} to {:.3%}, inference speed {:.3f}(s):'.format(best_score, np.mean(list(eval_metrics.values())), fps))

            # best_score = np.mean(list(eval_metrics.values())[:2])
            best_score = np.mean(list(eval_metrics.values()))

            # model, optimizer = update_Memory (savedir)
            # set train mode
            # model.train()
            # set optimizer
            # optimizer.zero_grad()
            c2 = True

            if best_score_AUC < np.mean(list(eval_metrics.values())[1:]):

                best_score_AUC = np.mean(list(eval_metrics.values())[1:])
                json.dump(state, open(os.path.join(savedir, 'best_score_AUC.json'),'w'), indent='\t')
                torch.save(model.state_dict(), os.path.join(savedir, 'best_model_AUC.pt'))


        elif best_score_AUC < np.mean(list(eval_metrics.values())[1:]):
            
            if  num_training_steps-step < 10:
                num_training_steps = num_training_steps + 10


            # save best score
            state = {'best_step':step}
            state.update({'model size:': model_size})
            state.update({'inference speed (s):':f"{fps:.3f} s"})
        
            state.update(eval_log)
            state.update({'globalimpressions1':impressions.tolist()})
            json.dump(state, open(os.path.join(savedir, 'best_score_AUC.json'),'w'), indent='\t')
            torch.save(model.state_dict(), os.path.join(savedir, 'best_model_AUC.pt'))
            _logger.warning('Best Score_AUC {:.3%} to {:.3%}, inference speed {:.3f}(s):'.format(best_score, np.mean(list(eval_metrics.values())[1:]), fps))
            
            best_score_AUC = np.mean(list(eval_metrics.values())[1:])


        else:
            c = c + 1

        if c == 3 and c2:
            globals.globalimpressions1 = np.array([1, 1, 1, 1, 1, 1])
            print("globalimpressions and rate", [impressions, globals.rate_p] )

            # model, optimizer = update_Memory (savedir)
            # set train mode
            # model.train()
            # set optimizer
            # optimizer.zero_grad()


            c = 0
            c2 = False

        impressions = np.array([0, 0, 0, 0, 0, 0])

        # scheduler
        if scheduler:
            scheduler.step()


        epoch += 1

        if epoch == num_training_steps:
            train_mode = False
            break



    # print best score and step
    _logger.warning('Best Metric: {0:.3%} (step {1:})'.format(best_score, state['best_step']))

    # save latest model
    torch.save(model.state_dict(), os.path.join(savedir, f'latest_model.pt'))

    # save latest score
    state = {'latest_step':epoch}
    state.update(eval_log)
    json.dump(state, open(os.path.join(savedir, 'latest_score.json'),'w'), indent='\t')


def evaluate(model, dataloader, device: str = 'cpu'):
    # targets and outputs
    image_targets = []
    image_masks = []
    anomaly_score = []
    anomaly_map = []

    model.eval()
    with torch.no_grad():
        
        t_per_imge = 0.0

        # total_iou = 0.0
        for idx, (inputs, masks, targets,_) in enumerate(dataloader):

            inputs, masks, targets = inputs.to(device), masks.to(device), targets.to(device)

            torch.cuda.synchronize()
            start_t = time.time()

            # predict
            # freq = torch.fft.fft2(inputs, norm='ortho')
            # freq = torch.fft.fftshift(freq, dim=(-2, -1))
            # # Combine real and imaginary parts
            # freq_combined = freq.real + 1j * freq.imag

            # # Inverse Fourier Transform
            # freq_inputs = torch.fft.ifft2(freq_combined, norm='ortho').real   

            # random_index = np.random.choice(range(3))
            # outputs = model(inputs, freq_inputs[:,random_index,:].unsqueeze(1))
            outputs = model(inputs)

            # freq = torch.fft.fft2(inputs, norm='ortho')
            # freq = torch.fft.fftshift(freq, dim=(-2, -1))
            # # Combine real and imaginary parts
            # freq_combined = freq.real + 1j * freq.imag

            # # Inverse Fourier Transform
            # freq_inputs = torch.fft.ifft2(freq_combined, norm='ortho').real
            # # print(inputs[0][0][0], freq_inputs[0][0][0])
            # outputs = model(inputs, freq_inputs)


            outputs = F.softmax(outputs, dim=1)
            # outputs = F.sigmoid(outputs)


            # preds = outputs[:,1,:].cpu()
            # masks = masks.cpu()
            # # threshold = 0.5
            # precision, recall, thresholds = precision_recall_curve(masks.reshape(-1), preds.flatten())
            # f1_score = (2 * precision * recall) / (precision + recall + 1e-10)
            # threshold = thresholds[np.argmax(f1_score)]
            # # threshold = thresholds[torch.argmax(f1_score)]
            # # print('pF1 adaptive threshold value:', threshold)
            # # Binarize predictions
            # preds_bin = (preds > threshold).float()

            # # Compute metrics
            # intersection = (preds_bin * masks).sum(dim=(1, 2))
            # union = preds_bin.sum(dim=(1, 2)) + masks.sum(dim=(1, 2)) - intersection
            # iou = intersection / (union + 1e-6)
            # # Accumulate metrics
            # total_iou += iou.mean().item()

            anomaly_score_i = torch.topk(torch.flatten(outputs[:,1,:], start_dim=1), 100)[0].mean(dim=1)
            # anomaly_score_i = torch.topk(torch.flatten(outputs.squeeze(1), start_dim=1), 100)[0].mean(dim=1)

            # stack targets and outputs
            image_targets.extend(targets.cpu().tolist())
            # image_masks.extend(masks.numpy())
            image_masks.extend(masks.cpu().numpy())

            anomaly_score.extend(anomaly_score_i.cpu().tolist())
            anomaly_map.extend(outputs[:,1,:].cpu().numpy())
            # anomaly_map.extend(outputs.squeeze(1).cpu().numpy())
            # anomaly_map.extend(outputs[:,1,:].cpu().detach().numpy())
            
            torch.cuda.synchronize()
            end_t = time.time() 
            t_per_imge += end_t - start_t

        inference_speed = t_per_imge / len(dataloader)
        print('inference speed:', inference_speed, 's')

    # del inputs, masks, targets, outputs
    # torch.cuda.empty_cache()


    # metrics
    image_masks = np.array(image_masks)
    anomaly_map = np.array(anomaly_map)

    # auroc_image = roc_auc_score(image_targets, anomaly_score)
    auroc_pixel = roc_auc_score(image_masks.reshape(-1).astype(int), anomaly_map.reshape(-1))
    all_fprs, all_pros = compute_pro(
        anomaly_maps      = anomaly_map,
        ground_truth_maps = image_masks
    )
    aupro = trapezoid(all_fprs, all_pros)

    AP_score = average_precision_score(image_masks.reshape(-1).astype(int), anomaly_map.reshape(-1))

    metrics = {
        'AP_score':AP_score,
        'AUROC-pixel':auroc_pixel,
        'AUPRO-pixel':aupro

    }

    _logger.warning('TEST: AP_score: %.3f%% | AUROC-pixel: %.3f%% | AUPRO-pixel: %.3f%%' %
                (metrics['AP_score'], metrics['AUROC-pixel'], metrics['AUPRO-pixel']))

    return metrics, inference_speed

