import logging
import os
import shutil
import tempfile
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

import random
import progressbar
from lib.utils import Timer, AverageMeter, precision_at_one, fast_hist, per_class_iu, \
    get_prediction, get_torch_device, save_soft_predictions_color, save_soft_pred_gt_color, save_predictions, visualize_results, \
    permute_pointcloud, save_rotation_pred

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from MinkowskiEngine import SparseTensor


def print_info(iteration,
               max_iteration,
               data_time,
               iter_time,
               has_gt=False,
               losses=None,
               scores=None,
               ious=None,
               hist=None,
               ap_class=None,
               class_names=None):
  debug_str = "{}/{}: ".format(iteration + 1, max_iteration)
  debug_str += "Data time: {:.4f}, Iter time: {:.4f}".format(data_time, iter_time)

  if has_gt:
    acc = hist.diagonal() / hist.sum(1) * 100
    debug_str += "\tLoss {loss.val:.3f} (AVG: {loss.avg:.3f})\t" \
        "Score {top1.val:.3f} (AVG: {top1.avg:.3f})\t" \
        "mIOU {mIOU:.3f} mAP {mAP:.3f} mAcc {mAcc:.3f}\n".format(
            loss=losses, top1=scores, mIOU=np.nanmean(ious),
            mAP=np.nanmean(ap_class), mAcc=np.nanmean(acc))
    if class_names is not None:
      debug_str += "\nClasses: " + " ".join(class_names) + '\n'
    # too many
    debug_str += 'IOU: ' + ' '.join('{:.03f}'.format(i) for i in ious) + '\n'
    debug_str += 'mAP: ' + ' '.join('{:.03f}'.format(i) for i in ap_class) + '\n'
    debug_str += 'mAcc: ' + ' '.join('{:.03f}'.format(i) for i in acc) + '\n'

  logging.info(debug_str)


def average_precision(prob_np, target_np):
  num_class = prob_np.shape[1]
  label = label_binarize(target_np, classes=list(range(num_class)))
  # return np.zeros((num_class))
  # too slow
  with np.errstate(divide='ignore', invalid='ignore'):
    return average_precision_score(label, prob_np, None)

def MLCE_loss(pred, target):
  N = pred.shape[0]
  eps=1e-11
  MLCE = -target * pred
  MLCE = torch.sum(MLCE, 1)
  MLCE += torch.log(torch.sum(torch.exp(target), 1))
  loss = MLCE.mean()
  return loss


def test(model, renderer, data_loader, config, transform_data_fn=None, has_gt=True):
  device = get_torch_device(config.is_cuda)
  dataset = data_loader.dataset
  num_labels = dataset.NUM_LABELS
  global_timer, data_timer, iter_timer = Timer(), Timer(), Timer()
  losses, scores, ious = AverageMeter(), AverageMeter(), 0
  aps = np.zeros((0, num_labels))
  hist = np.zeros((num_labels, num_labels))

  logging.info('===> Start testing')

  global_timer.tic()
  data_iter = data_loader.__iter__()
  max_iter = len(data_loader)
  max_iter_unique = max_iter

  # Fix batch normalization running mean and std
  model.eval()
  pred_examples = []
  gt_examples = []

  # Clear cache (when run in val mode, cleanup training cache)
  torch.cuda.empty_cache()

  if config.save_prediction or config.test_original_pointcloud:
    if config.save_prediction:
      save_pred_dir = config.save_pred_dir_iter
      os.makedirs(save_pred_dir, exist_ok=True)
    else:
      save_pred_dir = tempfile.mkdtemp()

  with torch.no_grad():
    # data to cache
    data_loader.dataset.dataBag.setcapacity(data_loader.dataset.render_epoch_num)
    obj_names = renderer.getObjNames()

    # re-rendering
    logging.info('===> New obj Rendering : Test')
    bar = progressbar.ProgressBar()
    renderer.setMotion('unique')
    for i in bar(range(data_loader.dataset.render_epoch_num)):
      tar_obj = obj_names[random.randint(0,len(obj_names) - 1)]
      dfrom = int(data_loader.dataset.render_frame_ratio[0] * renderer.getMotionFrames('unique'))
      dend = int(data_loader.dataset.render_frame_ratio[1] * renderer.getMotionFrames('unique'))
      frame_idx = random.randint(dfrom, dend - 1)
      if i > data_loader.dataset.render_epoch_num * 0.9:
        renderer.setMotion('__ROM__human25')
      data_loader.dataset.dataBag.setdata(i, renderer.render2pcd(tar_obj, frame_idx, random.randint(0,39), r_mode='LBS0'))
    print("\n")
    data_iter = data_loader.__iter__()

    for iteration in range(max_iter):
      data_timer.tic()
      if config.return_transformation:
        coords, input, target, transformation = data_iter.next()
      else:
        coords, input, target = data_iter.next()
        transformation = None
      data_time = data_timer.toc(False)

      # Preprocess input
      iter_timer.tic()

      if config.wrapper_type != 'None':
        color = input[:, :3].int()
      if config.normalize_color:
        input[:, :3] = input[:, :3] / 255. - 0.5
      sinput = SparseTensor(input, coords).to(device)

      # Feed forward
      inputs = (sinput,) if config.wrapper_type == 'None' else (sinput, coords, color)
      soutput_raw = model(*inputs)
      soutput = F.softmax(soutput_raw.F, dim=1)

      output = soutput
      pred_soft = output.cpu().numpy()
      gt_soft = target.cpu().numpy()

      pred = get_prediction(dataset, output, target).int()
      iter_time = iter_timer.toc(False)

      if (config.save_prediction or config.test_original_pointcloud) and iteration % config.save_pred_freq == 0:
        pred_pcd, gt_pcd = save_soft_pred_gt_color(soutput_raw.C.numpy()[:, 1:], pred_soft, gt_soft, transformation, dataset, config, iteration, save_pred_dir)
        pred_examples.append(pred_pcd)
        gt_examples.append(gt_pcd)


      if has_gt:
        if config.evaluate_original_pointcloud:
          raise NotImplementedError('pointcloud')

        # target_np = target.numpy()
        target_np = get_prediction(dataset, target, output).int()
        target_np = target_np.numpy()

        num_sample = target_np.shape[0]

        target = target.to(device)

        # loss = criterion(output, target)
        loss = MLCE_loss(output, target)
        score = 100 - torch.norm(soutput - target).item()
        losses.update(float(loss ), num_sample)
        scores.update(score, num_sample)
        hist += fast_hist(pred.cpu().numpy().flatten(), target_np.flatten(), num_labels)
        ious = per_class_iu(hist) * 100

        prob = torch.nn.functional.softmax(output, dim=1)
        ap = average_precision(prob.cpu().detach().numpy(), target_np)
        aps = np.vstack((aps, ap))
        # Due to heavy bias in class, there exists class with no test label at all
        with warnings.catch_warnings():
          warnings.simplefilter("ignore", category=RuntimeWarning)
          ap_class = np.nanmean(aps, 0) * 100.

      if iteration % config.test_stat_freq == 0 and iteration > 0:
        reordered_ious = dataset.reorder_result(ious)
        reordered_ap_class = dataset.reorder_result(ap_class)
        class_names = dataset.get_classnames()
        print_info(
            iteration,
            max_iter_unique,
            data_time,
            iter_time,
            has_gt,
            losses,
            scores,
            reordered_ious,
            hist,
            reordered_ap_class,
            class_names=class_names)

      if iteration % config.empty_cache_freq == 0:
        # Clear cache
        torch.cuda.empty_cache()

  global_time = global_timer.toc(False)

  reordered_ious = dataset.reorder_result(ious)
  reordered_ap_class = dataset.reorder_result(ap_class)
  class_names = dataset.get_classnames()
  print_info(
      iteration,
      max_iter_unique,
      data_time,
      iter_time,
      has_gt,
      losses,
      scores,
      reordered_ious,
      hist,
      reordered_ap_class,
      class_names=class_names)

  if config.test_original_pointcloud:
    logging.info('===> Start testing on original pointcloud space.')
    dataset.test_pointcloud(save_pred_dir)

  logging.info("Finished test. Elapsed time: {:.4f}".format(global_time))

  return losses.avg, scores.avg, np.nanmean(ap_class), np.nanmean(per_class_iu(hist)) * 100, pred_examples, gt_examples
