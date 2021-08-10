import logging
import os
import sys
from pathlib import Path

import numpy as np
from scipy import spatial

from lib.dataset import VoxelizationRenderedDataset, VoxelizationDataset, DatasetPhase, str2datasetphase_type
from lib.pc_utils import read_plyfile, save_point_cloud
from lib.utils import read_txt, fast_hist, per_class_iu

# import igl

COLOR_MAP = []
palette = [[0,0,1], [0,1,0], [1,0,0], [0,1,1], [1,0,1], [1,1,0],\
    [1,0,0.5], [0,1,0.5], [1,0.5,0], [0.5, 0,1],[0,0.5,1], [0.5,1,0],   [0.7,0.7,0.7], \
    [0.7,0.3,0], [0.3, 0,0.7], [0,0.3,0.7], [0.3,0.7,0], [0.7,0,0.3], [0,0.7,0.3], [0.1,0.1,0.1]]
LABELS = []
scale = 5
decimator = 3
for l in range(99):
  LABELS.append(l)
  s = l % 2 + 1
  r = l % len(palette)
  COLOR_MAP.append((100 + s * 50) * np.array(palette[r]))

  # hashnum = int(l) + 1
  # out_color_x = ((hashnum % decimator) + 1) / scale * 255
  # hashnum = hashnum//decimator
  # out_color_y = ((hashnum % decimator) + 1) / scale * 255
  # hashnum = hashnum//decimator
  # out_color_z = ((hashnum % decimator) + 1) / scale * 255
  # COLOR_MAP.append([out_color_x, out_color_y, out_color_z])
LABELS.append(-1)
COLOR_MAP.append([128, 128, 128])
COLOR_MAP = np.array(COLOR_MAP)
CLASS_LABELS = map(str, LABELS)

class VIMnetVoxelizationDataset(VoxelizationRenderedDataset):

  # Voxelization arguments
  CLIP_BOUND = None
  TEST_CLIP_BOUND = None
  VOXEL_SIZE = 0.01

  # Augmentation arguments
  ROTATION_AUGMENTATION_BOUND = ((0, 0), (0, 0), (0, 0))
  TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.005, 0.005), (-0.005, 0.005), (-0.005, 0.005))
  ELASTIC_DISTORT_PARAMS = ((0.01, 0.02), (0.02, 0.04))

  ROTATION_AXIS = 'z'
  LOCFEAT_IDX = 2
  NUM_LABELS = 99
  IGNORE_LABELS = ()
  IS_FULL_POINTCLOUD_EVAL = True

  # AUGMENT_COORDS_TO_FEATS = True
  # NUM_IN_CHANNEL = 6

  PHASE_SAMPLE = {
      DatasetPhase.Train: (1800, (0.0, 1.0)),
      DatasetPhase.Val: (200, (0.9, 0.95)),
      DatasetPhase.TrainVal: (2000, (0.0, 0.95)),
      DatasetPhase.Test: (100, (0.95, 1.0))
  }

  def __init__(self,
               config,
               prevoxel_transform=None,
               input_transform=None,
               target_transform=None,
               augment_data=True,
               elastic_distortion=False,
               cache=False,
               phase=DatasetPhase.Train):
    if isinstance(phase, str):
      phase = str2datasetphase_type(phase)
    # Use cropped rooms for train/val
    self.render_epoch_num = self.PHASE_SAMPLE[phase][0]
    self.render_frame_ratio = self.PHASE_SAMPLE[phase][1]
    if phase not in [DatasetPhase.Train, DatasetPhase.TrainVal]:
      self.CLIP_BOUND = self.TEST_CLIP_BOUND
    # dummy
    data_paths = [""] * self.render_epoch_num
    data_root = ""
    logging.info('Set Sample per one epoch {}: {}'.format(self.__class__.__name__, self.render_epoch_num))
    super().__init__(
        data_paths,
        data_root=data_root,
        prevoxel_transform=prevoxel_transform,
        input_transform=input_transform,
        target_transform=target_transform,
        ignore_label=config.ignore_label,
        return_transformation=config.return_transformation,
        augment_data=augment_data,
        elastic_distortion=elastic_distortion,
        config=config)

  def get_output_id(self, iteration):
    return '_'.join(Path(self.data_paths[iteration]).stem.split('_')[:2])

  def _augment_locfeat(self, pointcloud):
    # Assuming that pointcloud is xyzrgb(...), append location feat.
    pointcloud = np.hstack(
        (pointcloud[:, :6], 100 * np.expand_dims(pointcloud[:, self.LOCFEAT_IDX], 1),
         pointcloud[:, 6:]))
    return pointcloud

  def test_pointcloud(self, pred_dir):
    # TODO
    return
    print('Running full pointcloud evaluation.')
    eval_path = os.path.join(pred_dir, 'fulleval')
    os.makedirs(eval_path, exist_ok=True)
    # Join room by their area and room id.
    # Test independently for each room.
    sys.setrecursionlimit(100000)  # Increase recursion limit for k-d tree.
    for i, data_path in enumerate(self.data_paths):
      room_id = self.get_output_id(i)
      pred = np.load(os.path.join(pred_dir, 'pred_%06d_%03d.npy' % (i, 0)))

      # save voxelized pointcloud predictions
      # igl.write_off(os.path.join(pred_dir, 'pred_%06d_%03d.off'), pred[:, :3], [], pred[:, 3:])

