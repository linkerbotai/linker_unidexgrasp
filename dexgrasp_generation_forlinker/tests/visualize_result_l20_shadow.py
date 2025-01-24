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
from datasets.l20_hand_builder import L20HandBuilder

joint_names = [
    'index_finger_joint0', 'index_finger_joint1', 'index_finger_joint2', 'index_finger_joint3',
    'little_finger_joint0', 'little_finger_joint1', 'little_finger_joint2', 'little_finger_joint3',
    'middle_finger_joint0', 'middle_finger_joint1', 'middle_finger_joint2', 'middle_finger_joint3',
    'ring_finger_joint0', 'ring_finger_joint1', 'ring_finger_joint2', 'ring_finger_joint3',
    'thumb_joint0', 'thumb_joint1', 'thumb_joint2', 'thumb_joint3', 'thumb_joint4'
]

def exchange_qpos(joint_angles):
    print(type(joint_angles))
    mapping = {
        'thumb_joint0': 'robot0:THJ4',
        'thumb_joint1': 'robot0:THJ3',
        'thumb_joint2': 'robot0:THJ2',
        'thumb_joint3': 'robot0:THJ1',
        'thumb_joint4': 'robot0:THJ0',
        'index_finger_joint0': 'robot0:FFJ3',
        'index_finger_joint1': 'robot0:FFJ2',
        'index_finger_joint2': 'robot0:FFJ1',
        'index_finger_joint3': 'robot0:FFJ0',
        'middle_finger_joint0': 'robot0:MFJ3',
        'middle_finger_joint1': 'robot0:MFJ2',
        'middle_finger_joint2': 'robot0:MFJ1',
        'middle_finger_joint3': 'robot0:MFJ0',
        'ring_finger_joint0': 'robot0:RFJ3',
        'ring_finger_joint1': 'robot0:RFJ2',
        'ring_finger_joint2': 'robot0:RFJ1',
        'ring_finger_joint3': 'robot0:RFJ0',
        'little_finger_joint0': 'robot0:LFJ3',
        'little_finger_joint1': 'robot0:LFJ2',
        'little_finger_joint2': 'robot0:LFJ1',
        'little_finger_joint3': 'robot0:LFJ0'
    }
    joint_names = [
        'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
        'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
        'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
        'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
        'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
    ]
    new_joint_dict = {}

    # 遍历映射字典，将关节名称和角度对应起来
    for new_name, old_name in mapping.items():
        index = joint_names.index(old_name)
        new_joint_dict[new_name] = joint_angles[index]

    values_array = list(new_joint_dict.values())# 将新字典中的值转换为列表

    negative_list = [-x for x in values_array]

    # 打印结果
    print(negative_list)
    return np.array(negative_list)


def combine_transforms(t1, R, t2):
    """
    将平移+旋转+平移转换为旋转+平移。
    :param t1: 第一次平移向量 (3,)
    :param R: 旋转矩阵 (3, 3)
    :param t2: 第二次平移向量 (3,)
    :return: 等效的旋转矩阵和平移向量
    """
    # 等效平移
    t_total = np.dot(R, t1) + t2
    return R, t_total

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, default='eval_mypc')
    parser.add_argument('--num', type=int, default=2)
    parser.add_argument('--canonical_frame', type=int, default=0)#canonical_frame是什么意思规范框架
    args = parser.parse_args()
    
    # load data
    result = torch.load(os.path.join(args.exp_dir, 'result.pt'), map_location='cpu')
    print(type(result[0]))#遍历打印list的所有key
    result = result[0]
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
    builder = L20HandBuilder()
    builder_shadow = ShadowHandBuilder()
    # object mesh
    object_code = result['object_code'][args.num]
    print(object_code)
    object_scale = result['scale'][args.num].item()
    object_mesh = tm.load(os.path.join('./mini_generation/DFCData/meshes', object_code, 'coacd/decomposed.obj')).apply_scale(object_scale)
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
    hand_rotation_0 = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    hand_rotation_z = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1]
    ])
    R_y_180 = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ])
    theta = np.pi / 6  # 30 degrees in radians
    rotation_matrix_y = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    hand_translation_0 = np.array([0, 0, 0])
    hand_translation_l20 = np.array([0, -0.006, -0.063])


    # hand mesh shadow
    tta_hand_translation_shadow = result['sampled_rotation'][args.num].numpy().T @ result['tta_hand_pose'][args.num][:3].numpy() if args.canonical_frame else result['tta_hand_pose'][args.num][:3].numpy()
    tta_hand_rotation_shadow = np.eye(3) if args.canonical_frame else pytorch3d.transforms.axis_angle_to_matrix(result['tta_hand_pose'][args.num][3:6]).numpy()
    print(tta_hand_translation_shadow)
    print(type(tta_hand_translation_shadow))
    # tta_hand_translation_shadow = hand_translation_0
    # tta_hand_rotation_shadow  = hand_rotation_0
    tta_hand_qpos_shadow = result['tta_hand_pose'][args.num][6:].numpy()
    tta_hand_mesh_shadow = builder_shadow.get_hand_mesh(
        rotation_mat=tta_hand_rotation_shadow,
        world_translation=tta_hand_translation_shadow,
        qpos=tta_hand_qpos_shadow,
    )
    tta_hand_mesh_shadow = tm.Trimesh(
        vertices=tta_hand_mesh_shadow.verts_list()[0].numpy(),
        faces=tta_hand_mesh_shadow.faces_list()[0].numpy()
    )
    tta_hand_mesh_plotly_shadow = go.Mesh3d(
        x=tta_hand_mesh_shadow.vertices[:, 0],
        y=tta_hand_mesh_shadow.vertices[:, 1],
        z=tta_hand_mesh_shadow.vertices[:, 2],
        i=tta_hand_mesh_shadow.faces[:, 0],
        j=tta_hand_mesh_shadow.faces[:, 1],
        k=tta_hand_mesh_shadow.faces[:, 2],
        color='lightblue',
        opacity=1,
    )



    tta_hand_translation = result['sampled_rotation'][args.num].numpy().T @ result['tta_hand_pose'][args.num][:3].numpy() if args.canonical_frame else result['tta_hand_pose'][args.num][:3].numpy()
    tta_hand_rotation = np.eye(3) if args.canonical_frame else pytorch3d.transforms.axis_angle_to_matrix(result['tta_hand_pose'][args.num][3:6]).numpy()
    tta_hand_qpos = result['tta_hand_pose'][args.num][6:].numpy()
    #姿态映射代码
    tta_hand_qpos = exchange_qpos(tta_hand_qpos)
    tta_hand_rotation = np.dot(tta_hand_rotation, hand_rotation_z)
    tta_hand_rotation, tta_hand_translation = combine_transforms(hand_translation_l20, tta_hand_rotation, tta_hand_translation)

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
    fig = go.Figure([object_pc_plotly, tta_hand_mesh_plotly, tta_hand_mesh_plotly_shadow])
    fig.update_layout(scene_aspectmode='data')
    fig.show()
