import pytorch_kinematics as pk
from pytorch3d.structures import Meshes, join_meshes_as_batch
import torch
import os

import numpy as np
import trimesh
import ipdb

class L20HandBuilder():
    joint_names = [
        # 'WRJ1', 'WRJ0',
        'index_finger_joint0', 'index_finger_joint1', 'index_finger_joint2','index_finger_joint3',
        'little_finger_joint0', 'little_finger_joint1', 'little_finger_joint2', 'little_finger_joint3',
        'middle_finger_joint0', 'middle_finger_joint1', 'middle_finger_joint2', 'middle_finger_joint3',
        'ring_finger_joint0', 'ring_finger_joint1', 'ring_finger_joint2', 'ring_finger_joint3',
        'thumb_joint0', 'thumb_joint1', 'thumb_joint2', 'thumb_joint3', 'thumb_joint4']
    # joint_names = ["robot0:" + name for name in joint_names]

    mesh_filenames = [  "forearm_electric.obj",
                        "forearm_electric_cvx.obj",
                        "wrist.obj",
                        "palm.obj",
                        "knuckle.obj",
                        "F3.obj",
                        "F2.obj",
                        "F1.obj",
                        "lfmetacarpal.obj",
                        "TH3_z.obj",
                        "TH2_z.obj",
                        "TH1_z.obj"]

    def __init__(self,
                 mesh_dir="data/mjcf/meshes",
                 mjcf_path="data/l20_652/urdf/l20_652.urdf"):
        # ipdb.set_trace()
        # data = open(mjcf_path).read()
        self.chain = pk.build_chain_from_urdf(open(mjcf_path).read()).to(dtype=torch.float)
        # print(self.chain._root)

        self.mesh = {}
        device = 'cpu'

        def build_mesh_recurse(body):
            if(len(body.link.visuals) > 0):
                # print(body.link.visuals)
                link_name = body.link.name
                link_vertices = []
                link_faces = []
                n_link_vertices = 0
                for visual in body.link.visuals:
                    scale = torch.tensor([1, 1, 1], dtype=torch.float, device=device)
                    if visual.geom_type == "box":
                        link_mesh = trimesh.primitives.Box(extents=2 * visual.geom_param)
                    elif visual.geom_type == "capsule":
                        link_mesh = trimesh.primitives.Capsule(radius=visual.geom_param[0], height=visual.geom_param[1]*2).apply_translation((0, 0, -visual.geom_param[1]))
                    elif visual.geom_type == "mesh":
                        link_mesh = trimesh.load_mesh(os.path.join('./data/'+ visual.geom_param.split(":")[1][:-4] + '.STL'), process=False)
                        # if visual.geom_param[1] is not None:
                        #     scale = torch.tensor(visual.geom_param[1], dtype=torch.float, device=device)
                    vertices = torch.tensor(link_mesh.vertices, dtype=torch.float, device=device)
                    faces = torch.tensor(link_mesh.faces, dtype=torch.long, device=device)
                    pos = visual.offset.to(device)
                    vertices = vertices * scale
                    vertices = pos.transform_points(vertices)
                    link_vertices.append(vertices)
                    link_faces.append(faces + n_link_vertices)
                    n_link_vertices += len(vertices)
                link_vertices = torch.cat(link_vertices, dim=0)
                link_faces = torch.cat(link_faces, dim=0)
                self.mesh[link_name] = {'vertices': link_vertices,
                                        'faces': link_faces,
                                        }
            for children in body.children:
                build_mesh_recurse(children)
        build_mesh_recurse(self.chain._root)

    def qpos_to_qpos_dict(self, qpos,
                          hand_qpos_names=None):
        """
        :param qpos: [24]
        WARNING: The order must correspond with the joint_names
        """
        if hand_qpos_names is None:
            hand_qpos_names = ShadowHandBuilder.joint_names
        assert len(qpos) == len(hand_qpos_names)
        return dict(zip(hand_qpos_names, qpos))

    def qpos_dict_to_qpos(self, qpos_dict,
                          hand_qpos_names=None):
        """
        :return: qpos: [24]
        WARNING: The order must correspond with the joint_names
        """
        if hand_qpos_names is None:
            hand_qpos_names = ShadowHandBuilder.joint_names
        return np.array([qpos_dict[name] for name in hand_qpos_names])

    def get_hand_mesh(self,
                      rotation_mat,
                      world_translation,
                      qpos=None,
                      hand_qpos_dict=None,
                      hand_qpos_names=None,
                      without_arm=False):
        """
        Either qpos or qpos_dict should be provided.
        :param qpos: [24] numpy array
        :rotation_mat: [3, 3]
        :world_translation: [3]
        :return:
        """
        if qpos is None:
            if hand_qpos_names is None:
                hand_qpos_names = ShadowHandBuilder.joint_names
            assert hand_qpos_dict is not None, "Both qpos and qpos_dict are None!"
            qpos = np.array([hand_qpos_dict[name] for name in hand_qpos_names], dtype=np.float32)
        # ipdb.set_trace()
        # print(qpos[np.newaxis, :])
        current_status = self.chain.forward_kinematics(qpos[np.newaxis, :])#qpos 转换为形状为 (1, num_joints) 的数组


        meshes = []

        for link_name in self.mesh:
            v = current_status[link_name].transform_points(self.mesh[link_name]['vertices'])
            v = v @ rotation_mat.T + world_translation
            f = self.mesh[link_name]['faces']
            meshes.append(Meshes(verts=[v], faces=[f]))

        if without_arm:
            meshes = join_meshes_as_batch(meshes[1:])  # each link is a "batch"
        else:
            meshes = join_meshes_as_batch(meshes)  # each link is a "batch"
        return Meshes(verts=[meshes.verts_packed().type(torch.float32)],
                      faces=[meshes.faces_packed()])
