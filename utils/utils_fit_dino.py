import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .utils import get_lr


def fit_one_epoch(model_train, model, loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, genval, genvalshi, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss      = 0
    total_accuracy  = 0

    val_loss            = 0
    val_total_accuracy  = 0
    val_total_precision = 0
    val_total_recall    = 0
    val_total_F1        = 0
    
    if local_rank == 0:
        print('----------Start Train----------')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        images, crop_images, targets, crop_targets = batch
        
        crop_images = torch.cat(crop_images)
        crop_targets = torch.cat(crop_targets)

        with torch.no_grad():
            if cuda:
                images       = images.cuda(local_rank)
                crop_images  = crop_images.cuda(local_rank)
                targets      = targets.cuda(local_rank)
                crop_targets = crop_targets.cuda(local_rank)
                
        
        if not fp16:
            outputs, outputs_crop, teacherout, cropout = model_train(images, crop_images)
            #teacherout                        = torch.round(nn.Sigmoid()(teacherout))
            output                = loss(outputs, targets)
            
            outputs_crop = outputs_crop.chunk(2)
            crop_targets = crop_targets.chunk(2)
            cropout      = cropout.chunk(2)
            for i in range(len(outputs_crop)):
                output            = output + loss(outputs_crop[i], crop_targets[i])
                #output            = output + 0.1 * epoch / Epoch * teacher_loss(outputs_crop[i], teacherout, 0.1, 0.05)
                output            = output + 0.01 * epoch / Epoch * teacher_loss(cropout[i], teacherout, 0.1, 0.05)

            lambda_value = 0.01
            l2_reg       = torch.tensor(0., requires_grad=True)
            for param in model_train.parameters():
                l2_reg = l2_reg + torch.norm(param, p=2)
            
            output      = output + 0.5 * lambda_value * l2_reg

            model_train.update_moving_average() 
            model_train.update_moving_average_p()
            optimizer.zero_grad()
            output.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs, outputs_crop = model_train(images, crop_images)
                output                = loss(outputs, targets)
                for i in range(len(outputs_crop)):
                    output            = output + loss(outputs_crop[i], crop_targets[i])

                lambda_value = 0.01
                l2_reg       = torch.tensor(0., requires_grad=True)
                for param in model_train.parameters():
                    l2_reg = l2_reg + torch.norm(param, p=2)
                
                output      = output + 0.5 * lambda_value * l2_reg

            model_train.update_moving_average() 
            model_train.update_moving_average_p()
            optimizer.zero_grad()

            scaler.scale(output).backward()
            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
            equal       = torch.eq(torch.round(nn.Sigmoid()(outputs)), targets)
            accuracy    = torch.mean(equal.float())

        total_loss      += output.item()
        total_accuracy  += accuracy.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'acc'       : total_accuracy / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.eval()
    for iteration, batch in enumerate(genval):
        if iteration >= epoch_step_val:
            break
        
        images, _, targets, _ = batch

        with torch.no_grad():
            if cuda:
                images       = images.cuda(local_rank)
                targets      = targets.cuda(local_rank)
            
            optimizer.zero_grad()
            outputs, _, _, _  = model_train(images, images)
            output      = loss(outputs, targets)

            equal       = torch.eq(torch.round(nn.Sigmoid()(outputs)), targets)
            accuracy    = torch.mean(equal.float())
            predict     = torch.round(nn.Sigmoid()(outputs))
            
            tp          = ((predict == 1) & (targets == 1)).sum().item()
            fp          = ((predict == 1) & (targets == 0)).sum().item()
            fn          = ((predict == 0) & (targets == 1)).sum().item()

            precision   = tp / (tp + fp + 1e-10)
            recall      = tp / (tp + fn + 1e-10)
            f1          = (precision * recall) / (precision + recall + 1e-10)

        val_loss            += output.item()
        val_total_accuracy  += accuracy.item()
        val_total_precision += precision
        val_total_recall    += recall
        val_total_F1        += f1

        if local_rank == 0:
            pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1), 
                                'acc'       : val_total_accuracy / (iteration + 1),
                                'pre'       : val_total_precision / (iteration + 1),
                                'rec'       : val_total_recall / (iteration + 1),
                                'f1'        : val_total_F1 / (iteration + 1)})
            pbar.update(1)
                
    if local_rank == 0:
        pbar.close()
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val, val_total_accuracy / epoch_step_val, val_total_precision / epoch_step_val, val_total_recall / epoch_step_val, val_total_F1 / epoch_step_val)
        print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f || Val Acc: %.3f || Val Pre: %.3f || Val Rec: %.3f || Val F1: %.3f' % (total_loss / epoch_step, val_loss / epoch_step_val, val_total_accuracy / epoch_step_val, val_total_precision / epoch_step_val, val_total_recall / epoch_step_val, val_total_F1 / epoch_step_val))
        print('----------Finish Validation----------')
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_loss" + str(val_loss / epoch_step_val) + "_epoch_weights.pth"))
        
        if len(loss_history.val_acc) <= 1 or (val_total_accuracy / epoch_step_val) >= max(loss_history.val_acc):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_acc" + str(val_total_accuracy / epoch_step_val) + "_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
    
    val_loss            = 0
    val_total_accuracy  = 0
    val_total_precision = 0
    val_total_recall    = 0
    val_total_F1        = 0
    if genvalshi is None:
        return
    pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    for iteration, batch in enumerate(genvalshi):
        if iteration >= epoch_step_val:
            break
        
        images, _, targets, _ = batch

        with torch.no_grad():
            if cuda:
                images       = images.cuda(local_rank)
                targets      = targets.cuda(local_rank)
            
            optimizer.zero_grad()
            outputs, _, _, _ = model_train(images, images)
            output     = loss(outputs, targets)

            equal       = torch.eq(torch.round(nn.Sigmoid()(outputs)), targets)
            accuracy    = torch.mean(equal.float())
            predict     = torch.round(nn.Sigmoid()(outputs))
            
            tp          = ((predict == 1) & (targets == 1)).sum().item()
            fp          = ((predict == 1) & (targets == 0)).sum().item()
            fn          = ((predict == 0) & (targets == 1)).sum().item()

            precision   = tp / (tp + fp + 1e-10)
            recall      = tp / (tp + fn + 1e-10)
            f1          = (precision * recall) / (precision + recall + 1e-10)

        val_loss            += output.item()
        val_total_accuracy  += accuracy.item()
        val_total_precision += precision
        val_total_recall    += recall
        val_total_F1        += f1

        if local_rank == 0:
            pbar.set_postfix(**{'shipsear_val_loss'  : val_loss / (iteration + 1), 
                                'shipsear_acc'       : val_total_accuracy / (iteration + 1),
                                'shipsear_pre'       : val_total_precision / (iteration + 1),
                                'shipsear_rec'       : val_total_recall / (iteration + 1),
                                'shipsear_f1'        : val_total_F1 / (iteration + 1)})
            
            pbar.update(1)
        
        with open(save_dir + '/shipsear_' + str(epoch+1) + '.txt', 'a') as f:
                f.write('Val Loss: %.3f || Val Acc: %.3f || Val Pre: %.3f || Val Rec: %.3f || Val F1: %.3f \n' % (val_loss / (iteration + 1), val_total_accuracy / (iteration + 1), val_total_precision / (iteration + 1), val_total_recall / (iteration + 1), val_total_F1 / (iteration + 1)))

def teacher_loss(crop_out, teacher_out, tps, tpt):
    t = F.softmax((teacher_out / tpt), dim=-1)
    #c = F.softmax((crop_out / tps), dim=-1)

    loss = torch.sum(-t * F.log_softmax(crop_out / tps, dim=-1), dim=-1)
    return loss.mean()

def merge_tensors(tensor1, tensor2):
    tensor1 = np.array(tensor1)
    tensor2 = np.array(tensor2)

    # 检查两个数组的形状是否相同
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must have the same shape.")

    # 初始化合并后的结果数组
    merged_tensor = np.zeros_like(tensor1, dtype=float)

    # 遍历数组的每个元素
    for i in range(len(tensor1)):
        if tensor1[i] == tensor2[i]:
            # 如果元素相同，则保持不变
            merged_tensor[i] = tensor1[i]
        else:
            # 如果元素不同，则计算平均值
            merged_tensor[i] = 0.9 * tensor1[i] + 0.1 * tensor2[i]

    return merged_tensor

def valid_one_epoch(idx, model_train, model, loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, genval, genvalshi, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss      = 0
    total_accuracy  = 0

    val_loss            = 0
    val_total_accuracy  = 0
    val_total_precision = 0
    val_total_recall    = 0
    val_total_F1        = 0
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    if local_rank == 0:
        #pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.eval()
    for iteration, batch in enumerate(genval):
        if iteration >= epoch_step_val:
            break
        
        images, _, targets, _ = batch

        with torch.no_grad():
            if cuda:
                images       = images.cuda(local_rank)
                targets      = targets.cuda(local_rank)
            
            optimizer.zero_grad()
            outputs, _, _, _  = model_train(images, images)
            output      = loss(outputs, targets)

            equal       = torch.eq(torch.round(nn.Sigmoid()(outputs)), targets)
            accuracy    = torch.mean(equal.float())
            predict     = torch.round(nn.Sigmoid()(outputs))
            
            tp          = ((predict == 1) & (targets == 1)).sum().item()
            fp          = ((predict == 1) & (targets == 0)).sum().item()
            fn          = ((predict == 0) & (targets == 1)).sum().item()
            tn          = ((predict == 0) & (targets == 0)).sum().item()
            
            TP         += tp
            FP         += fp
            TN         += tn
            FN         += fn

            precision   = tp / (tp + fp + 1e-10)
            recall      = tp / (tp + fn + 1e-10)
            f1          = (precision * recall) / (precision + recall + 1e-10)

        val_loss            += output.item()
        val_total_accuracy  += accuracy.item()
        val_total_precision += precision
        val_total_recall    += recall
        val_total_F1        += f1

        if local_rank == 0:
            pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1), 
                                'acc'       : val_total_accuracy / (iteration + 1),
                                'pre'       : val_total_precision / (iteration + 1),
                                'rec'       : val_total_recall / (iteration + 1),
                                'f1'        : val_total_F1 / (iteration + 1)})
            pbar.update(1)
                
    if local_rank == 0:
        pbar.close()
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val, val_total_accuracy / epoch_step_val, val_total_precision / epoch_step_val, val_total_recall / epoch_step_val, val_total_F1 / epoch_step_val)
        print('IDX:'+ str(idx))
        print('Total Loss: %.3f || Val Loss: %.3f || Val Acc: %.3f || Val Pre: %.3f || Val Rec: %.3f || Val F1: %.3f' % (total_loss / epoch_step, val_loss / epoch_step_val, val_total_accuracy / epoch_step_val, val_total_precision / epoch_step_val, val_total_recall / epoch_step_val, val_total_F1 / epoch_step_val))
        print('TP: {} TN: {} FP: {} FN: {}'.format(TP, TN, FP, FN))
        #print('----------Finish Validation----------')

    
    val_loss            = 0
    val_total_accuracy  = 0
    val_total_precision = 0
    val_total_recall    = 0
    val_total_F1        = 0
    pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    for iteration, batch in enumerate(genvalshi):
        if iteration >= epoch_step_val:
            break
        
        images, _, targets, _ = batch

        with torch.no_grad():
            if cuda:
                images       = images.cuda(local_rank)
                targets      = targets.cuda(local_rank)
            
            optimizer.zero_grad()
            outputs, _, _, _ = model_train(images, images)
            output     = loss(outputs, targets)

            equal       = torch.eq(torch.round(nn.Sigmoid()(outputs)), targets)
            accuracy    = torch.mean(equal.float())
            predict     = torch.round(nn.Sigmoid()(outputs))
            
            tp          = ((predict == 1) & (targets == 1)).sum().item()
            fp          = ((predict == 1) & (targets == 0)).sum().item()
            fn          = ((predict == 0) & (targets == 1)).sum().item()

            precision   = tp / (tp + fp + 1e-10)
            recall      = tp / (tp + fn + 1e-10)
            f1          = (precision * recall) / (precision + recall + 1e-10)

        val_loss            += output.item()
        val_total_accuracy  += accuracy.item()
        val_total_precision += precision
        val_total_recall    += recall
        val_total_F1        += f1

        if local_rank == 0:
            pbar.set_postfix(**{'shipsear_val_loss'  : val_loss / (iteration + 1), 
                                'shipsear_acc'       : val_total_accuracy / (iteration + 1),
                                'shipsear_pre'       : val_total_precision / (iteration + 1),
                                'shipsear_rec'       : val_total_recall / (iteration + 1),
                                'shipsear_f1'        : val_total_F1 / (iteration + 1)})
            
            pbar.update(1)
        
        with open(save_dir + '/shipsear_' + str(epoch+1) + str(idx) + 'idx.txt', 'a') as f:
                f.write('Val Loss: %.3f || Val Acc: %.3f || Val Pre: %.3f || Val Rec: %.3f || Val F1: %.3f \n' % (val_loss / (iteration + 1), val_total_accuracy / (iteration + 1), val_total_precision / (iteration + 1), val_total_recall / (iteration + 1), val_total_F1 / (iteration + 1)))


def awgn(signal, snr_dB):
    signal_power = torch.mean(torch.abs(signal) ** 2)  
    snr_linear = 10 ** (snr_dB / 10)  
    noise_power = signal_power / snr_linear  
    noise = torch.randn_like(signal) * torch.sqrt(noise_power)  
    noisy_signal = signal + noise  
    return noisy_signal

def add_noise_to_tensor(tensor, snr):
    noisy_tensors = []
    for batch_index in range(tensor.size(0)):  
        batch_noisy_tensors = []
        for i in range(tensor.size(1)):  
            noisy_subtensor = awgn(tensor[batch_index, i], snr)
            batch_noisy_tensors.append(noisy_subtensor)

        noisy_batch_tensor = torch.stack(batch_noisy_tensors, dim=0)
        noisy_tensors.append(noisy_batch_tensor)

    noisy_tensor = torch.stack(noisy_tensors, dim=0)
    return noisy_tensor




def valid_snr_one_epoch(idx, snr, model_train, model, loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, genval, genvalshi, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss      = 0
    total_accuracy  = 0

    val_loss            = 0
    val_total_accuracy  = 0
    val_total_precision = 0
    val_total_recall    = 0
    val_total_F1        = 0
    
    if local_rank == 0:
        #pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.eval()
    for iteration, batch in enumerate(genval):
        if iteration >= epoch_step_val:
            break
        
        images, _, targets, _ = batch
        
        images = add_noise_to_tensor(images, snr)
        #print(images.shape)

        with torch.no_grad():
            if cuda:
                images       = images.cuda(local_rank)
                targets      = targets.cuda(local_rank)
            
            optimizer.zero_grad()
            outputs, _, _, _  = model_train(images, images)
            output      = loss(outputs, targets)

            equal       = torch.eq(torch.round(nn.Sigmoid()(outputs)), targets)
            accuracy    = torch.mean(equal.float())
            predict     = torch.round(nn.Sigmoid()(outputs))
            
            tp          = ((predict == 1) & (targets == 1)).sum().item()
            fp          = ((predict == 1) & (targets == 0)).sum().item()
            fn          = ((predict == 0) & (targets == 1)).sum().item()

            precision   = tp / (tp + fp + 1e-10)
            recall      = tp / (tp + fn + 1e-10)
            f1          = (precision * recall) / (precision + recall + 1e-10)

        val_loss            += output.item()
        val_total_accuracy  += accuracy.item()
        val_total_precision += precision
        val_total_recall    += recall
        val_total_F1        += f1

        if local_rank == 0:
            pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1), 
                                'acc'       : val_total_accuracy / (iteration + 1),
                                'pre'       : val_total_precision / (iteration + 1),
                                'rec'       : val_total_recall / (iteration + 1),
                                'f1'        : val_total_F1 / (iteration + 1)})
            pbar.update(1)
                
    if local_rank == 0:
        pbar.close()
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val, val_total_accuracy / epoch_step_val, val_total_precision / epoch_step_val, val_total_recall / epoch_step_val, val_total_F1 / epoch_step_val)
        print('SNR:'+ str(snr))
        print('Total Loss: %.3f || Val Loss: %.3f || Val Acc: %.3f || Val Pre: %.3f || Val Rec: %.3f || Val F1: %.3f' % (total_loss / epoch_step, val_loss / epoch_step_val, val_total_accuracy / epoch_step_val, val_total_precision / epoch_step_val, val_total_recall / epoch_step_val, val_total_F1 / epoch_step_val))
        with open(save_dir + '/deepship_' + str(epoch+1) + str(snr) + 'idx.txt', 'a') as f:
            f.write('SNR:'+ str(snr) + '\n')
            f.write('Total Loss: %.3f || Val Loss: %.3f || Val Acc: %.3f || Val Pre: %.3f || Val Rec: %.3f || Val F1: %.3f \n' % (total_loss / epoch_step, val_loss / epoch_step_val, val_total_accuracy / epoch_step_val, val_total_precision / epoch_step_val, val_total_recall / epoch_step_val, val_total_F1 / epoch_step_val))
        #print('----------Finish Validation----------')
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
    
    val_loss            = 0
    val_total_accuracy  = 0
    val_total_precision = 0
    val_total_recall    = 0
    val_total_F1        = 0
    pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    for iteration, batch in enumerate(genvalshi):
        if iteration >= epoch_step_val:
            break
        
        images, _, targets, _ = batch
        images = add_noise_to_tensor(images, snr)

        with torch.no_grad():
            if cuda:
                images       = images.cuda(local_rank)
                targets      = targets.cuda(local_rank)
            
            optimizer.zero_grad()
            outputs, _, _, _ = model_train(images, images)
            output     = loss(outputs, targets)

            equal       = torch.eq(torch.round(nn.Sigmoid()(outputs)), targets)
            accuracy    = torch.mean(equal.float())
            predict     = torch.round(nn.Sigmoid()(outputs))
            
            tp          = ((predict == 1) & (targets == 1)).sum().item()
            fp          = ((predict == 1) & (targets == 0)).sum().item()
            fn          = ((predict == 0) & (targets == 1)).sum().item()

            precision   = tp / (tp + fp + 1e-10)
            recall      = tp / (tp + fn + 1e-10)
            f1          = (precision * recall) / (precision + recall + 1e-10)

        val_loss            += output.item()
        val_total_accuracy  += accuracy.item()
        val_total_precision += precision
        val_total_recall    += recall
        val_total_F1        += f1

        if local_rank == 0:
            pbar.set_postfix(**{'shipsear_val_loss'  : val_loss / (iteration + 1), 
                                'shipsear_acc'       : val_total_accuracy / (iteration + 1),
                                'shipsear_pre'       : val_total_precision / (iteration + 1),
                                'shipsear_rec'       : val_total_recall / (iteration + 1),
                                'shipsear_f1'        : val_total_F1 / (iteration + 1)})
            
            pbar.update(1)
        
        with open(save_dir + '/shipsear_' + str(epoch+1) + str(snr) + 'idx.txt', 'a') as f:
                f.write('Val Loss: %.3f || Val Acc: %.3f || Val Pre: %.3f || Val Rec: %.3f || Val F1: %.3f \n' % (val_loss / (iteration + 1), val_total_accuracy / (iteration + 1), val_total_precision / (iteration + 1), val_total_recall / (iteration + 1), val_total_F1 / (iteration + 1)))