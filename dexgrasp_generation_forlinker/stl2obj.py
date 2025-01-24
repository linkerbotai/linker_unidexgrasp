import os
import open3d as o3d

# 设置包含.STL文件的文件夹路径
folder_path = './data/rightHand_9dof/meshes'
# 设置输出文件夹路径
output_folder_path = './data/rightHand_9dof/meshes'

# 检查输出文件夹是否存在，如果不存在则创建
os.makedirs(output_folder_path, exist_ok=True)

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    print(filename)
    # 检查文件扩展名是否为.STL
    if filename.endswith(".STL") or filename.endswith(".stl"):
        # 构造完整的文件路径
        file_path = os.path.join(folder_path, filename)
        # 读取STL文件
        mesh_stl = o3d.io.read_triangle_mesh(file_path)
        # 构造.obj文件的完整输出路径
        output_path = os.path.join(output_folder_path, filename[:-4] + '.obj')
        print(output_path)
        # 导出为OBJ文件
        o3d.io.write_triangle_mesh(output_path, mesh_stl)

print("转换完成。")