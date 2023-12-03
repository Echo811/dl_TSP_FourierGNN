import pickle

import sys, os
sys.path.append('..')

# 序列化对象
def save_(obj_dic, path='my_tool/obj/'):
    print(os.getcwd())
    for obj_name in obj_dic:
        obj = obj_dic[obj_name]
        f = open(path+f'{obj_name}.txt', 'wb')
        pickle.dump(obj=obj, file=f)
        print(f'-----------保存{obj_name}对象成功！------------')