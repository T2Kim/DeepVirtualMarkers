import argparse
from glob import glob
from Labeler import *

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# NOTE: Please ensure that your model has proper scale! (additional_aff <- here))
# getLabel -> dict object
#             ["o3d_vim"]: colored model ([0]), marker sphere ([1~99])
#             ["coord"]: position (n_vertices X 3)
#             ["color"]: color (n_vertices X 3)
#             ["indices"]: (if mesh) triangles (n_faces X 3)
#             ["pred"]: label (n_vertices X 99)

#       getLabel -> KNN (to original points)
#       getLabelVoxel -> no interpolation

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Res16UNet34C', help='Model name')
parser.add_argument('--weights', type=str, default='./data/weights.pth')
parser.add_argument('--marker', type=int, default=99)
parser.add_argument('--bn_momentum', type=float, default=0.05)
parser.add_argument('--voxel_size', type=float, default=0.01)
parser.add_argument('--conv1_kernel_size', type=int, default=3)

g_play = True
def play_onoff():
    global g_play
    g_play = not g_play

if __name__ == '__main__':
    config = parser.parse_args()
    labeler = Labeler(config)
    
    # Mesh
    ## renderpeople (Free rigged model)
    print("---------------------------------------------------")
    print("---------------------------------------------------")
    print("----------------ESC to next example----------------")
    print("---------------------------------------------------")
    print("---------------------------------------------------\n\n")

    print("-------------------Mesh example-------------------")
    print("-------------------Renderpeople-------------------")
    filename = "./data/rp_manuel_animated_001_dancing_F607.obj"
    results = labeler.getLabel(filename, is_mesh = True, y_filp=False, additional_aff = np.eye(3))
    o3d.visualization.draw_geometries(results["o3d_vim"])

    print("-------------------Different color maps-------------------")
    labeler.cmap_name = "legacy" # (varying colors for each markers)
    filename = "./data/rp_manuel_animated_001_dancing_F450.obj"
    results = labeler.getLabel(filename, is_mesh = True, y_filp=False, additional_aff = np.eye(3))
    o3d.visualization.draw_geometries(results["o3d_vim"])
    labeler.cmap_name = "default"

    ## SCAPE
    print("-------------------SCAPE-------------------")
    filename = "./data/mesh031.obj"
    results = labeler.getLabel(filename, is_mesh = True, y_filp=False, additional_aff = np.eye(3))
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(results["coord"])
    mesh.triangles = o3d.utility.Vector3iVector(results["indices"])
    mesh.vertex_colors = o3d.utility.Vector3dVector(results["color"])
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])

    ## FAUST
    print("-------------------FAUST-------------------")
    filename = "./data/test_scan_011.ply"
    results = labeler.getLabel(filename, is_mesh = True, y_filp=False, additional_aff = np.eye(3))
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(results["coord"])
    mesh.triangles = o3d.utility.Vector3iVector(results["indices"])
    mesh.vertex_colors = o3d.utility.Vector3dVector(results["color"])
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])

    # Image
    print("-------------------Image example-------------------")
    print("------------------Single depth map-----------------")
    ## Single depth map
    filename = "./data/000178.png"
    results = labeler.getLabel(filename, is_mesh = False, y_filp=True, additional_aff = np.eye(3))
    o3d.visualization.draw_geometries(results["o3d_vim"])

    ## Stream
    print("-------------------Depth stream-------------------")
    print("--------------Space bar: play on/off--------------")
    conf = json.load(open("./lib/render_conf.json", 'rt', encoding='UTF8'))
    data_case = conf['KinectAzure']
    d_intrinsic = data_case['parameter']['depth_intrinsic']
    labeler.setIntrinsic(d_intrinsic)

    filenames = glob("./data/stream/*.png")
    filenames.sort()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.register_key_callback(256, lambda vis: exit()) #ESC
    vis.register_key_callback(32, lambda vis: play_onoff()) #Space bar -> play on / off

    results = labeler.getLabelVoxel(filenames[0], is_mesh = False, y_filp=True)
    pcd_vec = results["o3d_vim"]
    vis.add_geometry(pcd_vec[0])
    frame_idx = 0
    while True:
        frame_idx %= len(filenames)
        if g_play:
            print("frame index : ", frame_idx)
            results = labeler.getLabelVoxel(filenames[frame_idx], is_mesh = False, y_filp=True)
            pcd_vec[0].points = results["o3d_vim"][0].points
            pcd_vec[0].colors = results["o3d_vim"][0].colors
            pcd_vec[0].normals = results["o3d_vim"][0].normals
            vis.update_geometry(pcd_vec[0])
            frame_idx += 1
        vis.poll_events()
        vis.update_renderer()

