import pandas as pd
import numpy as np

# 用pandas读取CSV文件
csv_file_path = 'data/traffic.csv'
data = pd.read_csv(csv_file_path)

# 将数据转换为NumPy数组
numpy_array = data.to_numpy()

# 保存为.npy文件
npy_file_path = 'data/traffic.npy'
np.save(npy_file_path, numpy_array)
