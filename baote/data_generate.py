import numpy as np
# 生成数据 影响因素为 年龄，工艺类型，搬运时间，使用的工具类型  结果为工时


import pandas as pd
from faker import Faker
from random import randint, uniform

fake = Faker()
# num_samples = 10000  # 设定生成样本的数目


def train_test_split(data, ratio):
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data


# # 初始化一个数据字典，将每个列作为键，并将生成的数据列表设定为值
# data_dict = {
#     "age": [randint(20, 65) for _ in range(num_samples)],  # 随机在20和65岁之间
#     "process_type": [fake.random_element(elements=("1", "2", "3", "4")) for _ in range(num_samples)],  # 随机选择工艺类型
#     "handling_time": [uniform(0.5, 3.5) for _ in range(num_samples)],  # 随机选择在0.5和3.5小时之间的搬运时间
#     "tool_type": [fake.random_element(elements=("1", "2", "3", "4")) for _ in range(num_samples)],  # 随机选择工具类型
#     "working_hours": [randint(4, 12) for _ in range(num_samples)]  # 随机在4和12小时之间选择工作时间
# }





# 从数据字典中创建数据框
# df = pd.DataFrame(data_dict)


def get_workData(data):
    return 3*data[0]+5*data[1]+data[2]+2*data[3]


def get_data(num_samples=1000):
    data_dict = {
        "age": [randint(20, 65) for _ in range(num_samples)],  # 随机在20和65岁之间
        "process_type": [randint(1, 5) for _ in range(num_samples)],  # 随机选择工艺类型
        "handling_time": [uniform(0.5, 3.5) for _ in range(num_samples)],  # 随机选择在0.5和3.5小时之间的搬运时间
        "tool_type": [randint(1, 5) for _ in range(num_samples)],  # 随机选择工具类型
    }

    data_dict["working_hours"] = [3 * data_dict["age"][i] + 5 * data_dict["process_type"][i] + \
                                  data_dict["handling_time"][i] + 2 * data_dict["tool_type"][i] \
                                  for i in range(num_samples)]
    # 从数据字典中创建数据框
    df = pd.DataFrame(data_dict)

    data_array = np.array(list(data_dict.values()))

    # 为了使每一行表示一个样本（一行包含所有的特性），我们需要转置数组
    data_array = np.transpose(data_array)
    return data_array


data_array = get_data()
a, b = train_test_split(data_array, 0.8)
print(a.shape)
print(a[0])


c=get_workData(a[2][:-1])
print(a[2][:-1])
print(c)