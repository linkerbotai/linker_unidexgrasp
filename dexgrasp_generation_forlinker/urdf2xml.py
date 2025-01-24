import xml.etree.ElementTree as ET

# 读取URDF文件
tree = ET.parse('./data/rightHand_9dof/urdf/rightHand_9dof.urdf')
root = tree.getroot()

# 打印URDF的XML结构
print(ET.tostring(root, encoding='unicode'))

# 如果需要，可以将URDF保存为XML格式
tree.write('./data/rightHand_9dof/urdf/rightHand_9dof.xml', encoding='utf-8')