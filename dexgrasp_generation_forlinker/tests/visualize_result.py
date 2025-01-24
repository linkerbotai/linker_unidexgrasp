import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.realpath('.'))

import argparse
import numpy as np
import trimesh as tm
import torch
import pytorch3d.transforms
import plotly.graph_objects as go
from datasets.shadow_hand_builder import ShadowHandBuilder


joint_names = [
    'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
    'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
    'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
    'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
    'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, default='eval_mypc')
    parser.add_argument('--num', type=int, default=0)
    parser.add_argument('--canonical_frame', type=int, default=0)
    args = parser.parse_args()
    
    # load data
    result = torch.load(os.path.join(args.exp_dir, 'result.pt'), map_location='cpu')
    print(result)
    print(len(result))#40 1 1
    result = result[0]
    print(len(result['obj_pc']))#50 1 4
    # print(type(result))
    # print(len(result))
    # print(result[0])
    # print(type(result[1]))
    # result.pt大概应该是这样的格式
#   {

#     'object_code': ['object1', 'object2', 'object3', ...],
#     'scale': [1.0, 1.2, 1.5, ...],
#     'canon_obj_pc': [
#         torch.tensor([[x1, y1, z1], [x2, y2, z2], ...]),
#         torch.tensor([[x1, y1, z1], [x2, y2, z2], ...]),
#         ...
#     ],
#     'obj_pc': [
#         torch.tensor([[x1, y1, z1], [x2, y2, z2], ...]),
#         torch.tensor([[x1, y1, z1], [x2, y2, z2], ...]),
#         ...
#     ],
#     'sampled_rotation': [
#         torch.tensor([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]]),
#         torch.tensor([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]]),
#         ...
#     ],
#     'translation': [
#         torch.tensor([tx1, ty1, tz1]),
#         torch.tensor([tx2, ty2, tz2]),
#         ...
#     ],
#     'hand_qpos': [
#         torch.tensor([q1, q2, q3, ...]),
#         torch.tensor([q1, q2, q3, ...]),
#         ...
#     ],
#     'tta_hand_pose': [
#         torch.tensor([tx, ty, tz, rx, ry, rz, q1, q2, q3, ...]),
#         torch.tensor([tx, ty, tz, rx, ry, rz, q1, q2, q3, ...]),
#         ...
#     ]
# }
    
    # hand model
    builder = ShadowHandBuilder()
    
    # object mesh
    object_code = result['object_code'][args.num]
    object_scale = result['scale'][args.num].item()
    object_mesh = tm.load(os.path.join('/media/moning/Newsmy/from_obj_to_pc/DFCData/meshes', object_code, 'coacd/decomposed.obj')).apply_scale(object_scale)
    object_pc = result['canon_obj_pc' if args.canonical_frame else 'obj_pc'][args.num].numpy()
    object_pc_plotly = go.Scatter3d(
        x=object_pc[:, 0],
        y=object_pc[:, 1],
        z=object_pc[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color='lightgreen',
        )
    )
    
    # hand mesh
    hand_translation = result['sampled_rotation'][args.num].numpy().T @ result['translation'][args.num].numpy() if args.canonical_frame else result['translation'][args.num].numpy()
    hand_rotation = np.eye(3) if args.canonical_frame else result['sampled_rotation'][args.num].numpy()
    hand_qpos = result['hand_qpos'][args.num].numpy()
    hand_mesh = builder.get_hand_mesh(
        rotation_mat=hand_rotation,
        world_translation=hand_translation,
        qpos=hand_qpos,
    )
    hand_mesh = tm.Trimesh(
        vertices=hand_mesh.verts_list()[0].numpy(),
        faces=hand_mesh.faces_list()[0].numpy()
    )
    hand_mesh_plotly = go.Mesh3d(
        x=hand_mesh.vertices[:, 0],
        y=hand_mesh.vertices[:, 1],
        z=hand_mesh.vertices[:, 2],
        i=hand_mesh.faces[:, 0],
        j=hand_mesh.faces[:, 1],
        k=hand_mesh.faces[:, 2],
        color='lightblue',
        opacity=0.5,
    )#opacity是什么意思：透明度。
    tta_hand_translation = result['sampled_rotation'][args.num].numpy().T @ result['tta_hand_pose'][args.num][:3].numpy() if args.canonical_frame else result['tta_hand_pose'][args.num][:3].numpy()
    tta_hand_rotation = np.eye(3) if args.canonical_frame else pytorch3d.transforms.axis_angle_to_matrix(result['tta_hand_pose'][args.num][3:6]).numpy()
    tta_hand_qpos = result['tta_hand_pose'][args.num][6:].numpy()
    tta_hand_mesh = builder.get_hand_mesh(
        rotation_mat=tta_hand_rotation,
        world_translation=tta_hand_translation,
        qpos=tta_hand_qpos,
    )
    tta_hand_mesh = tm.Trimesh(
        vertices=tta_hand_mesh.verts_list()[0].numpy(),
        faces=tta_hand_mesh.faces_list()[0].numpy()
    )
    tta_hand_mesh_plotly = go.Mesh3d(
        x=tta_hand_mesh.vertices[:, 0],
        y=tta_hand_mesh.vertices[:, 1],
        z=tta_hand_mesh.vertices[:, 2],
        i=tta_hand_mesh.faces[:, 0],
        j=tta_hand_mesh.faces[:, 1],
        k=tta_hand_mesh.faces[:, 2],
        color='lightblue',
        opacity=1,
    )
    
    # visualize
    fig = go.Figure([object_pc_plotly, hand_mesh_plotly, tta_hand_mesh_plotly])
    fig.update_layout(scene_aspectmode='data')
    fig.show()
