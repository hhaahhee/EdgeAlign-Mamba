import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from utils import save_imgs


def calculate_binary_metrics(predictions, targets, threshold=0.5):
    """Calculate pooled pixel-level metrics for binary lesion segmentation."""
    y_pred = np.asarray(predictions).reshape(-1) >= threshold
    y_true = np.asarray(targets).reshape(-1) >= 0.5
    tp = int(np.logical_and(y_pred, y_true).sum())
    tn = int(np.logical_and(~y_pred, ~y_true).sum())
    fp = int(np.logical_and(y_pred, ~y_true).sum())
    fn = int(np.logical_and(~y_pred, y_true).sum())

    def safe_div(numerator, denominator):
        return float(numerator) / float(denominator) if denominator else 0.0

    return {
        'dice': safe_div(2 * tp, 2 * tp + fp + fn),
        'iou': safe_div(tp, tp + fp + fn),
        'sensitivity': safe_div(tp, tp + fn),
        'specificity': safe_div(tn, tn + fp),
        'accuracy': safe_div(tp + tn, tp + tn + fp + fn),
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
    }


def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    step,
                    logger, 
                    config,
                    writer):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()
        images, targets = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

        out = model(images)
        loss = criterion(out, targets)

        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step() 
    return step


def val_one_epoch(test_loader,
                    model,
                    criterion, 
                    epoch, 
                    logger,
                    config):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
            loss = criterion(out, msk)

            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out) 

    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        metrics = calculate_binary_metrics(preds, gts, config.threshold)
        log_info = (f"val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, "
                    f"iou: {metrics['iou']}, dice: {metrics['dice']}, "
                    f"accuracy: {metrics['accuracy']}, specificity: {metrics['specificity']}, "
                    f"sensitivity: {metrics['sensitivity']}, "
                    f"confusion: TN={metrics['tn']}, FP={metrics['fp']}, "
                    f"FN={metrics['fn']}, TP={metrics['tp']}")
        print(log_info)
        logger.info(log_info)

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)
    
    return np.mean(loss_list)


def test_one_epoch(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
            loss = criterion(out, msk)

            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out) 
            if i % config.save_interval == 0:
                save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        metrics = calculate_binary_metrics(preds, gts, config.threshold)
        metrics['loss'] = float(np.mean(loss_list))

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = (f"evaluation of best model, loss: {metrics['loss']:.4f}, "
                    f"iou: {metrics['iou']}, dice: {metrics['dice']}, "
                    f"accuracy: {metrics['accuracy']}, specificity: {metrics['specificity']}, "
                    f"sensitivity: {metrics['sensitivity']}, "
                    f"confusion: TN={metrics['tn']}, FP={metrics['fp']}, "
                    f"FN={metrics['fn']}, TP={metrics['tp']}")
        print(log_info)
        logger.info(log_info)

    return metrics
