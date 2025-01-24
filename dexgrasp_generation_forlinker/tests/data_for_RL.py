import torch
import numpy as np
import os
import pytorch3d.transforms
from scipy.spatial.transform import Rotation as R

def rotation_matrix_to_euler(rotation_mat, order='xyz', degrees=False):
    """
    将旋转矩阵转换为欧拉角表示
    :param rotation_mat: 3x3 旋转矩阵
    :param order: 欧拉角的旋转顺序（如 'xyz', 'zyx' 等）
    :param degrees: 是否以角度为单位返回（默认 False，返回弧度）
    :return: [roll, pitch, yaw] 欧拉角表示
    """
    # 确保输入是 numpy 数组
    rotation = R.from_matrix(rotation_mat)
    # 转换为欧拉角
    euler_angles = rotation.as_euler(order, degrees=degrees)
    return euler_angles

# def exchange_qpos(joint_angles):
#     print(type(joint_angles))
#     mapping = {
#         'thumb_joint0': 'robot0:THJ4',
#         'thumb_joint1': 'robot0:THJ3',
#         'thumb_joint2': 'robot0:THJ2',
#         'thumb_joint3': 'robot0:THJ1',
#         'thumb_joint4': 'robot0:THJ0',
#         'index_finger_joint0': 'robot0:FFJ3',
#         'index_finger_joint1': 'robot0:FFJ2',
#         'index_finger_joint2': 'robot0:FFJ1',
#         'index_finger_joint3': 'robot0:FFJ0',
#         'middle_finger_joint0': 'robot0:MFJ3',
#         'middle_finger_joint1': 'robot0:MFJ2',
#         'middle_finger_joint2': 'robot0:MFJ1',
#         'middle_finger_joint3': 'robot0:MFJ0',
#         'ring_finger_joint0': 'robot0:RFJ3',
#         'ring_finger_joint1': 'robot0:RFJ2',
#         'ring_finger_joint2': 'robot0:RFJ1',
#         'ring_finger_joint3': 'robot0:RFJ0',
#         'little_finger_joint0': 'robot0:LFJ3',
#         'little_finger_joint1': 'robot0:LFJ2',
#         'little_finger_joint2': 'robot0:LFJ1',
#         'little_finger_joint3': 'robot0:LFJ0'
#     }
#     joint_names = [
#         'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
#         'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
#         'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
#         'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
#         'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
#     ]
#     new_joint_dict = {}
#
#
#     # 遍历映射字典，将关节名称和角度对应起来
#     for new_name, old_name in mapping.items():
#         index = joint_names.index(old_name)
#         new_joint_dict[new_name] = joint_angles[index]
#
#     # values_array = list(new_joint_dict.values())# 将新字典中的值转换为列表
#     #
#     # negative_list = [-x for x in values_array]
#
#     # 打印结果
#     # print(new_joint_dict)
#     return new_joint_dict
# def exchange_rotation(rotation_mat):
#     R_x_90 = np.array([
#         [1, 0, 0],
#         [0, 0, -1],
#         [0, 1, 0]
#     ])
#     R_y_90 = np.array([
#         [0, 0, 1],
#         [0, 1, 0],
#         [-1, 0, 0]
#     ])
#     R_z_90 = np.array([
#         [0, -1, 0],
#         [1, 0, 0],
#         [0, 0, 1]
#     ])
#     R_prime = R_z_90 @ rotation_mat
#     return R_prime
# def exchange_translation(translation):
#     return np.array([translation[0]-0.02, translation[1]+0.1, translation[2]+0.095])

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
        print('joint_angles', joint_angles)
        print(f"new_name: {new_name}, index: {index}, joint_angles{index}: {joint_angles[index]}")

        new_joint_dict[new_name] = joint_angles[index]

    values_array = list(new_joint_dict.values())# 将新字典中的值转换为列表

    negative_list = [-x for x in values_array]

    # 打印结果
    print(negative_list)
    # return np.array(negative_list)
    return new_joint_dict
def exchange_rotation(rotation_mat):
    R_x_90 = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    R_y_90 = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0]
    ])
    R_z_90 = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    R_prime = R_z_90 @ rotation_mat
    return R_prime
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

def process_and_save_data(pt_file_path, save_dir):
    """
    从.pt文件中提取指定数据并保存为.npy文件。

    :param pt_file_path: .pt文件的路径。
    :param save_dir: 保存.npy文件的目录。
    :param object_code: 保存路径中使用的对象代码。
    """
    hand_rotation_z = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1]
    ])
    hand_translation_l20 = np.array([0, -0.006, -0.063])
    # 加载.pt文件
    data = torch.load(pt_file_path, map_location=torch.device('cpu'))
    result = data[0]
    # print(result)

    for i in range(len(result['object_code'])):
        object_code = result['object_code'][i]
        tta_hand_translation =  result['tta_hand_pose'][i][:3].numpy()
        tta_hand_rotation = pytorch3d.transforms.axis_angle_to_matrix(result['tta_hand_pose'][i][3:6]).numpy()
        tta_hand_qpos = result['tta_hand_pose'][i][6:].numpy()
        # print(result['tta_hand_pose'])
        # print(len(tta_hand_qpos))
        tta_hand_qpos = exchange_qpos(tta_hand_qpos)
        tta_hand_rotation = np.dot(tta_hand_rotation, hand_rotation_z)
        tta_hand_rotation, tta_hand_translation = combine_transforms(hand_translation_l20, tta_hand_rotation,tta_hand_translation)
        tta_hand_rotation_euler = rotation_matrix_to_euler(tta_hand_rotation, order='xyz', degrees=False)#转换为欧拉角表示
        # print(tta_hand_rotation)
        arr_0 = tta_hand_qpos

        # arr_0['WRJRx'] = 0.9675633457489656

        arr_0['WRJRx'] = tta_hand_rotation_euler[0]
        arr_0['WRJRy'] = tta_hand_rotation_euler[1]
        arr_0['WRJRz'] = tta_hand_rotation_euler[2]
        arr_0['WRJTx'] = tta_hand_translation[0]
        arr_0['WRJTy'] = tta_hand_translation[1]
        arr_0['WRJTz'] = tta_hand_translation[2]
        print(arr_0)
        # arr_0 = np.array([list(arr_0.values())])

        arr_1 = result['scale'][i]
        # print(object_code, arr_1)
        arr_1 = np.array([arr_1])
        arr_2 = result['plane'][i]
        # arr_2 = np.array([arr_2])
        # 构建保存路径
        save_path = os.path.join(save_dir, object_code)
        os.makedirs(save_path, exist_ok=True)

        np.savez(os.path.join(save_path, '00000.npz'), arr_0, arr_1, arr_2)
        # np.savez('00000.npz', arr_0 = arr_0, arr_1 = arr_1, arr_2 = arr_2)


        print(f"数据已保存到 {save_path}")



# 示例调用
pt_file_path = "./eval/result.pt"  # 替换为你的.pt文件路径
save_dir = "./datasetv4.1/core"  # 替换为保存目录
# object_code = "object_code_example"  # 替换为具体对象代码
process_and_save_data(pt_file_path, save_dir)
