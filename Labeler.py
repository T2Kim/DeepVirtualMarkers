import numpy as np
from urllib.request import urlretrieve
import torch
import torch.nn.functional as F
import MinkowskiEngine as ME
try:
  import open3d as o3d
except ImportError:
  raise ImportError('Please install open3d with `pip install open3d`.')

import sys
sys.path.append("../")
sys.path.append("./")

from models import load_model
from models.res16unet import Res16UNet34C
# from lib.renderer import *
from lib.utils import precision_at_one
import json
import progressbar
import igl
import cv2
import os
import copy
from sklearn.neighbors import KDTree
from scipy import spatial

from matplotlib import cm

def pcd_sparse_tensor(coords, voxel_size=0.01):
    # Create a batch, this process is done in a data loader during training in parallel.
    batch = [pcd_voxel(coords, voxel_size)]
    coordinates_, featrues_ = list(zip(*batch))
    coordinates, features = ME.utils.sparse_collate(coordinates_, featrues_)    
    # Normalize features and create a sparse tensor
    return coordinates, (features - 0.5).float()

def pcd_voxel(coords, voxel_size):
    feats = np.zeros_like(coords, dtype=np.float32)
    quantized_coords = np.floor(coords / voxel_size)
    inds = ME.utils.sparse_quantize(quantized_coords, return_index=True)
    return quantized_coords[inds], feats[inds]

class Labeler:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        # Define a model and load the weights   
        NetClass = load_model(config.model)
        self.model = NetClass(3, config.marker, config).to(self.device)
        model_dict = torch.load(config.weights)
        self.model.load_state_dict(model_dict['state_dict'])
        self.model.eval()
        self.voxel_size = config.voxel_size
        self.marker = config.marker
        self._init_color_map()
        self.cmap_name = "default"
        self.image_intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

    def getLabel(self, filename, is_mesh = True, y_filp = True, additional_aff = np.eye(3)):
        if y_filp:
            additional_aff = np.dot(additional_aff, np.array([[1.,0,0],[0,-1,0],[0,0,-1]]))

        if is_mesh:
            v, f = igl.read_triangle_mesh(filename)
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(v)
            mesh.triangles = o3d.utility.Vector3iVector(f)
            mesh.compute_vertex_normals()
            tmp_pcd = mesh.sample_points_uniformly(100000)
            sample_points = np.dot(np.asarray(tmp_pcd.points), additional_aff)
            faces = f
            points = v
        else:
            d_arr = cv2.imread(filename, cv2.IMREAD_UNCHANGED).astype(np.float32)
            d_arr = cv2.bilateralFilter(d_arr, 10, 50, 1)
            d_arr /= 1000.0

            depth_raw = o3d.geometry.Image(d_arr)
            pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_raw, self.image_intrinsic)
            if y_filp:
                pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=40))
            pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
            sample_points = np.dot(np.asarray(pcd.points), additional_aff)
            points = np.asarray(pcd.points)

        pred, up_coord = self._inference(sample_points)
        up_coord_colors = self._gen_soft_color(pred)

        tar_m_pcd = o3d.geometry.PointCloud()
        tar_m_pcd.points = o3d.utility.Vector3dVector(np.dot((up_coord * self.voxel_size), np.linalg.inv(additional_aff)))
        tar_m_pcd_tree = o3d.geometry.KDTreeFlann(tar_m_pcd)

        mesh_p_bag = {}
        n_points = points.shape[0]
        mesh_p_bag["pred"] = np.zeros((n_points, self.config.marker))
        mesh_p_bag["color"] = np.zeros((n_points, 3))
        mesh_p_bag["coord"] = points
        if is_mesh:
            mesh_p_bag["indices"] = faces

        print("KNN Search")
        bar = progressbar.ProgressBar(maxval=n_points,widgets=[' [', progressbar.Timer(), '] ', progressbar.Bar(), ' (', progressbar.ETA(), ') ',]).start()
        
        for i, p in enumerate(points):
            
            near_indices = tar_m_pcd_tree.search_hybrid_vector_3d(p, self.voxel_size * 3.4, 20)
            if near_indices[0] < 1:
                near_indices = tar_m_pcd_tree.search_knn_vector_3d(p, 1)

            if (np.min(near_indices[2]) == 0):
                min_idx = np.argmin(near_indices[2])
                tmp_pred = pred[near_indices[1][min_idx], :]
                tmp_color = up_coord_colors[near_indices[1][min_idx], :]
            else:
                dist_weights = np.reciprocal(near_indices[2])
                dist_weights.reshape((1, -1))
                tmp_pred = np.dot(dist_weights, pred[near_indices[1]]) / np.sum(dist_weights)
                tmp_color = np.dot(dist_weights, up_coord_colors[near_indices[1]]) / np.sum(dist_weights)
            
            mesh_p_bag["pred"][i, :]  = tmp_pred.reshape((1, -1))
            mesh_p_bag["color"][i, :] = tmp_color.reshape((1, -1))
            
            bar.update(i)
        bar.finish()


        if is_mesh:
            mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_p_bag["color"])
            mesh_p_bag["o3d_vim"] = [mesh]
        else:
            pcd.colors = o3d.utility.Vector3dVector(mesh_p_bag["color"])
            mesh_p_bag["o3d_vim"] = [pcd]

        # generate Virtual Marker sphere
        pred_max_idx = np.argsort(mesh_p_bag["pred"], axis=0)[-1, :]
        arr_w = list(range(mesh_p_bag["pred"].shape[1]))        
        pred_max_coords = mesh_p_bag["coord"][pred_max_idx, :]
        pred_max_w = mesh_p_bag["pred"][pred_max_idx, arr_w]
        
        for p, w in zip(pred_max_coords, pred_max_w):
            if(w > 0.3):
                mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                mesh_sphere.compute_vertex_normals()
                mesh_sphere.vertices = o3d.utility.Vector3dVector(np.asarray(mesh_sphere.vertices) + np.array(p))
                mesh_sphere.paint_uniform_color([0.1, 0.1, 0.1])
                mesh_p_bag["o3d_vim"].append(mesh_sphere)
            else:
                mesh_sphere = o3d.geometry.TriangleMesh()
                mesh_p_bag["o3d_vim"].append(mesh_sphere)

        return mesh_p_bag

    def getLabelVoxel(self, filename, is_mesh = True, y_filp = True, additional_aff = np.eye(3)):
        if y_filp:
            additional_aff = np.dot(additional_aff, np.array([[1.,0,0],[0,-1,0],[0,0,-1]]))

        if is_mesh:
            v, f = igl.read_triangle_mesh(filename)
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(v)
            mesh.triangles = o3d.utility.Vector3iVector(f)
            mesh.compute_vertex_normals()
            tmp_pcd = mesh.sample_points_uniformly(400000)
            sample_points = np.dot(np.asarray(tmp_pcd.points), additional_aff)
            points = np.asarray(mesh.vertices)
        else:
            d_arr = cv2.imread(filename, cv2.IMREAD_UNCHANGED).astype(np.float32)
            d_arr = cv2.bilateralFilter(d_arr, 10, 50, 1)
            d_arr /= 1000.0

            depth_raw = o3d.geometry.Image(d_arr)
            pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_raw, self.image_intrinsic)
            if y_filp:
                pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=40))
            pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
            sample_points = np.dot(np.asarray(pcd.points), additional_aff)
            points = np.asarray(pcd.points)

        pred, up_coord = self._inference(sample_points)

        tar_m_pcd = o3d.geometry.PointCloud()
        points = np.dot((up_coord * self.voxel_size), np.linalg.inv(additional_aff))
        tar_m_pcd.points = o3d.utility.Vector3dVector(points)

        mesh_p_bag = {}
        n_points = points.shape[0]
        mesh_p_bag["pred"] = pred
        mesh_p_bag["color"] = self._gen_soft_color(pred)
        mesh_p_bag["coord"] = points

        tar_m_pcd.colors = o3d.utility.Vector3dVector(mesh_p_bag["color"])
        tar_m_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=40))
        mesh_p_bag["o3d_vim"] = [tar_m_pcd]

        # generate Virtual Marker sphere
        pred_max_idx = np.argsort(mesh_p_bag["pred"], axis=0)[-1, :]
        arr_w = list(range(mesh_p_bag["pred"].shape[1]))        
        pred_max_coords = mesh_p_bag["coord"][pred_max_idx, :]
        pred_max_w = mesh_p_bag["pred"][pred_max_idx, arr_w]
        # pred_max_coords = pred_max_coords[np.where(pred_max_w > 0.3), :].reshape((-1,3))
        
        for p, w in zip(pred_max_coords, pred_max_w):
            if(w > 0.3):
                mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                mesh_sphere.compute_vertex_normals()
                mesh_sphere.vertices = o3d.utility.Vector3dVector(np.asarray(mesh_sphere.vertices) + np.array(p))
                mesh_sphere.paint_uniform_color([0.1, 0.1, 0.1])
                mesh_p_bag["o3d_vim"].append(mesh_sphere)
            else:
                mesh_sphere = o3d.geometry.TriangleMesh()
                mesh_p_bag["o3d_vim"].append(mesh_sphere)

        return mesh_p_bag

    def setIntrinsic(self, d_intrinsic):
        self.image_intrinsic.set_intrinsics(d_intrinsic['width'],
                                          d_intrinsic['height'],
                                              d_intrinsic['fx'],
                                              d_intrinsic['fy'],
                                              d_intrinsic['cx'],
                                              d_intrinsic['cy'])

    def _inference(self, coords):
        with torch.no_grad():
            coordinates, features = pcd_sparse_tensor(coords, voxel_size=self.voxel_size)
            sinput = ME.SparseTensor(features, coords=coordinates).to(self.device)
            soutput = self.model(sinput)
        pred = F.softmax(soutput.F[:, :self.marker], dim=1)
        pred = pred.cpu().numpy()
        cell = soutput.C.numpy()[:, 1:]
        return pred, cell

    def _gen_soft_color(self, upsampled_pred):
        v_color = np.zeros((len(upsampled_pred), 3), dtype=np.float32)

        for i in range(upsampled_pred.shape[1]):
            v_color += np.dot((upsampled_pred[:, i])[:, np.newaxis],
                            np.array(self.COLOR_MAPS[self.cmap_name][i], dtype=np.float32).reshape(1, 3))

        return v_color / 255

    def _init_color_map(self):
        self.COLOR_MAPS = {}

        ## default
        cc_map = cm.get_cmap('Paired')
        COLOR_MAP = np.zeros((100, 3))
        part = [[],[],[],[],[],[]]
        part[0]= np.array([12,6,7,8,9,10,11,0,1,2,3,4,5, 13,14,15,16])[::-1]                              # head
        part[1]=np.concatenate((np.arange(17,25),np.arange(45,69))) # body

        part[2]= np.arange(25,35) # right arm
        part[3]= np.arange(35,45) # left arm 

        part[4]= np.arange(69,84) # right leg
        part[5]= np.arange(84,99) # left leg

        n_part = 6
        for i, p in enumerate(part):
            n_seg = len(p)
            for j, m in enumerate(p):
                COLOR_MAP[m, :] = np.array([k * 255 for k in list(cc_map(i/n_part + (1/n_part) * (j/n_seg))[:3])])
        self.COLOR_MAPS["default"] = copy.deepcopy(COLOR_MAP)

        ## legacy
        COLOR_MAP = []

        palette = [[0,0,1], [0,1,0], [1,0,0], [0,1,1], [1,0,1], [1,1,0],\
            [1,0,0.5], [0,1,0.5], [1,0.5,0], [0.5, 0,1],[0,0.5,1], [0.5,1,0],   [0.7,0.7,0.7], \
            [0.7,0.3,0], [0.3, 0,0.7], [0,0.3,0.7], [0.3,0.7,0], [0.7,0,0.3], [0,0.7,0.3]]
        scale = 5
        decimator = 3
        for l in range(110):
            s = l % 2 + 1
            r = l % len(palette)
            COLOR_MAP.append((100 + s * 50) * np.array(palette[r]))

        COLOR_MAP.append([0, 0, 0])
        COLOR_MAP = np.array(COLOR_MAP)
        self.COLOR_MAPS["legacy"] = copy.deepcopy(COLOR_MAP)

