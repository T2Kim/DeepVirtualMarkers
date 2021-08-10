import logging
import os.path as osp

import torch
from torch import nn
import torch.nn.functional as F
import random
import progressbar
import numpy as np

# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

from lib.test import test
from lib.utils import checkpoint, precision_at_one, \
    Timer, AverageMeter, get_prediction, get_torch_device
from lib.solvers import initialize_optimizer, initialize_scheduler

from MinkowskiEngine import SparseTensor


def validate(model, renderer, val_data_loader, writer, curr_iter, config, transform_data_fn):
  # v_loss, v_score, v_mAP, v_mIoU, pred_examples = test(model, renderer, val_data_loader, config, transform_data_fn)
  v_loss, v_score, v_mAP, v_mIoU, pred_examples, gt_examples = test(model, renderer, val_data_loader, config, transform_data_fn)
  writer.add_scalar('validation/mIoU', v_mIoU, curr_iter)
  writer.add_scalar('validation/loss', v_loss, curr_iter)
  writer.add_scalar('validation/precision_at_1', v_score, curr_iter)

  for i, pcd in enumerate(pred_examples):
    if i > 3:
      break
    n = pcd.shape[0]
    t_v = torch.from_numpy(pcd[:,:3].reshape((1, n, 3)))
    t_c = torch.from_numpy(pcd[:,3:].reshape((1, n, 3)))
    writer.add_mesh(str(i).zfill(2) + '/pcds_', t_v, t_c, global_step=curr_iter)
  for i, pcd in enumerate(gt_examples):
    if i > 3:
      break
    n = pcd.shape[0]
    t_v = torch.from_numpy(pcd[:,:3].reshape((1, n, 3)))
    t_c = torch.from_numpy(pcd[:,3:].reshape((1, n, 3)))
    writer.add_mesh(str(i).zfill(2) + '/gt_', t_v, t_c, global_step=curr_iter)

  return v_mIoU

def MLCE_loss(pred, target):
  N = pred.shape[0]
  eps=1e-11
  MLCE = -target * pred
  MLCE = torch.sum(MLCE, 1)
  MLCE += torch.log(torch.sum(torch.exp(pred), 1))
  # MLCE_valid = MLCE[torch.nonzero(MLCE)]
  # loss = MLCE_valid.mean()
  loss = MLCE.mean()
  return loss


def train(model, renderer, data_loader, val_data_loader, config, transform_data_fn=None):
  device = get_torch_device(config.is_cuda)
  # Set up the train flag for batch normalization
  model.train()

  # Configuration
  writer = SummaryWriter(log_dir=config.log_dir)
  data_timer, iter_timer = Timer(), Timer()
  data_time_avg, iter_time_avg = AverageMeter(), AverageMeter()
  losses, scores = AverageMeter(), AverageMeter()

  optimizer = initialize_optimizer(model.parameters(), config)
  scheduler = initialize_scheduler(optimizer, config)
  # Train the network
  logging.info('===> Start training')
  best_val_miou, best_val_iter, curr_iter, epoch, is_training = 0, 0, 1, 1, True

  if config.resume:
    checkpoint_fn = config.resume + '/weights.pth'
    if osp.isfile(checkpoint_fn):
      logging.info("=> loading checkpoint '{}'".format(checkpoint_fn))
      state = torch.load(checkpoint_fn)
      curr_iter = state['iteration'] + 1
      epoch = state['epoch']
      model.load_state_dict(state['state_dict'])
      if config.resume_optimizer:
        scheduler = initialize_scheduler(optimizer, config, last_step=curr_iter)
        optimizer.load_state_dict(state['optimizer'])
      if 'best_val' in state:
        best_val_miou = state['best_val']
        best_val_iter = state['best_val_iter']
      logging.info("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_fn, state['epoch']))
    else:
      raise ValueError("=> no checkpoint found at '{}'".format(checkpoint_fn))

  data_iter = data_loader.__iter__()

  # data to cache
  data_loader.dataset.dataBag.setcapacity(data_loader.dataset.render_epoch_num)
  obj_names = renderer.getObjNames()

  while is_training:
    
    # re-rendering
    logging.info('===> New obj Rendering : Train')
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
    
    for iteration in range(len(data_loader) // config.iter_size):
      optimizer.zero_grad()
      data_time, batch_loss = 0, 0
      iter_timer.tic()

      for sub_iter in range(config.iter_size):
        # Get training data
        data_timer.tic()
        coords, input, target = data_iter.next()

        # For some networks, making the network invariant to even, odd coords is important. Random translation
        coords[:, 1:] += (torch.rand(3) * 100).type_as(coords)

        # Preprocess input
        color = input[:, :3].int()
        if config.normalize_color:
          input[:, :3] = input[:, :3] / 255. - 0.5
        sinput = SparseTensor(input, coords).to(device)

        data_time += data_timer.toc(False)

        # Feed forward
        inputs = (sinput,) if config.wrapper_type == 'None' else (sinput, coords, color)
        soutput_raw = model(*inputs)
        target = target.float().to(device)
        soutput = soutput_raw.F
        
        loss = MLCE_loss(soutput, target)

        # Compute and accumulate gradient
        loss /= config.iter_size
        batch_loss += loss.item()
        loss.backward()

      # clipping
      torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

      # Update number of steps
      optimizer.step()
      scheduler.step()

      data_time_avg.update(data_time)
      iter_time_avg.update(iter_timer.toc(False))

      pred = get_prediction(data_loader.dataset, soutput, target)
      score = 100 - torch.norm(F.softmax(soutput, dim=1) - target).item()
      losses.update(batch_loss, target.size(0))
      scores.update(score, target.size(0))

      if curr_iter >= config.max_iter:
        is_training = False
        break

      if curr_iter % config.stat_freq == 0 or curr_iter == 1:
        lrs = ', '.join(['{:.3e}'.format(x) for x in scheduler.get_lr()])
        debug_str = "===> Epoch[{}]({}/{}): Loss {:.4f}\tLR: {}\t".format(
            epoch, curr_iter,
            len(data_loader) // config.iter_size, losses.avg, lrs)
        debug_str += "Score {:.3f}\tData time: {:.4f}, Total iter time: {:.4f}".format(
            scores.avg, data_time_avg.avg, iter_time_avg.avg)
        logging.info(debug_str)
        # Reset timers
        data_time_avg.reset()
        iter_time_avg.reset()
        # Write logs
        writer.add_scalar('training/loss', losses.avg, curr_iter)
        writer.add_scalar('training/precision_at_1', scores.avg, curr_iter)
        writer.add_scalar('training/learning_rate', scheduler.get_lr()[0], curr_iter)
        losses.reset()
        scores.reset()

      # Save current status, save before val to prevent occational mem overflow
      if curr_iter % config.save_freq == 0:
        checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter)

      # Validation
      if curr_iter % config.val_freq == 0:
        config.save_pred_dir_iter = config.save_pred_dir + "/" + str(curr_iter).zfill(8)
        val_miou = validate(model, renderer, val_data_loader, writer, curr_iter, config, transform_data_fn)
        if val_miou > best_val_miou:
          best_val_miou = val_miou
          best_val_iter = curr_iter
          checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter,
                     "best_val")
        logging.info("Current best mIoU: {:.3f} at iter {}".format(best_val_miou, best_val_iter))

        # Recover back
        model.train()

      # End of iteration
      curr_iter += 1

    epoch += 1

  # Explicit memory cleanup
  if hasattr(data_iter, 'cleanup'):
    data_iter.cleanup()

  # Save the final model
  checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter)
  val_miou = validate(model, val_data_loader, writer, curr_iter, config, transform_data_fn)
  if val_miou > best_val_miou:
    best_val_miou = val_miou
    best_val_iter = curr_iter
    checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter, "best_val")
  logging.info("Current best mIoU: {:.3f} at iter {}".format(best_val_miou, best_val_iter))
