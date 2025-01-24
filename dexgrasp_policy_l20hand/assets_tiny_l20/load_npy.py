import numpy as np


file_path = './datasetv4.1/core/camera-1ab3abb5c090d9b68e940c4e64a94e1e/00000.npz'
# 使用 numpy 的 load 函数加载 .npy 文件
data = np.load(file_path, allow_pickle=True)
# print("Arrays in file:", data.files)
print(type(data))
print(type(data.files))
# 如果你想要打印所有数组的内容，可以使用循环
for array_name in data.files:
    print(f"\nContents of '{array_name}':")
    print(data[array_name])
    print(type(data[array_name]))
