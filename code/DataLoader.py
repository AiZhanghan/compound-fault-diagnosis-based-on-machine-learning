import os
import time
import numpy as np
import pandas as pd


class DataLoader:
    '''数据加载器
    get_time_data: 获取时域数据
    get_frequency_data: 获取频域数据
    load_data: csv to npz, 加载原始时域数据
    '''

    def __init__(self, N=5120,
        path="D:/Workspace/Data/20191113_compound_fault"):
        """
        Args:
            N: int, 样本长度
            path: str, 目标文件夹路径
        """
        self.N = N
        self.path = path

    def get_time_data(self):
        """获取时域数据
        Return:
            data_dic: dict{str: np.array}
            label_dic: dict{str: np.array}
            info: dict{str: int}
        """
        data_dic = dict(np.load(os.path.join(self.path, "data_%s.npz" % self.N)))
        label_dic = dict(np.load(os.path.join(self.path, "label_%s.npz" % self.N)))
        info = self._get_data_info(data_dic, label_dic)
        return data_dic, label_dic, info

    def get_frequency_data(self):
        '''获取频域数据, label_len = 8
        Return:
            data_dic: dict{str: np.array}
            label_dic: dict{str: np.array}
            info_dic: dict{str: int}
        '''
        # 获取时域数据
        data_dic, label_dic, _ = self.get_time_data()
        # FFT
        for key in data_dic:
            _, data_dic[key] = self._fft(data_dic[key])
        info = self._get_data_info(data_dic, label_dic)
        return data_dic, label_dic, info

    def load_data(self, 
        source_path=r"D:/Workspace/Data/20191113_compound_fault/time"):
        '''csv to npz, 加载时域数据
        Args:
            source_path: str, 源数据文件夹路径
        '''
        # 得到source_path目录下所有文件夹并排序
        folders = os.listdir(source_path)
        folders.sort(key=lambda x: int(x.split("_")[0]))
        filenames = ["sensor1.csv", "sensor2.csv", "sensor3.csv"]
        
        columns = [
            'MissingTooth',
            'RootCrack', 
            'Surface', 
            'ChippedTooth',
            'OuterRace',
            'InnerRace',
            'Ball'
            ]
        to_multilabel = {key: value for value, key in enumerate(columns, 1)}
        # data_dic, 包括所有样本
        # data, 单类样本, 用于组建datas
        data_dic = {}
        label_dic = {}

        for folder in folders:
            print("loading %s" % folder, end="\t")
            start_time = time.time()
            # 合并三个传感器为三个通道(batch_size, C, N)
            path = os.path.join(source_path, folder, filenames[0])
            data = pd.read_csv(path).values.reshape(-1, 1, self.N)
            for filename in filenames[1: ]:
                path = os.path.join(source_path, folder, filename)
                sensor = pd.read_csv(path).values.reshape(-1, 1, self.N)
                data = np.concatenate((data, sensor), axis = 1)
            
            label = np.zeros((data.shape[0], 8))
            # 构造Multilabel
            index = [to_multilabel[_class] 
                     for _class in folder.split('_')[-2: ]
                     if _class in to_multilabel]
            for i in range(len(index)):
                label[:, index[i]] = 1
            # Normal
            if not index:
                label[:, 0] = 1
            
            data_dic[folder] = data.copy()
            label_dic[folder] = label.copy()

            print("%.2fs" % (time.time() - start_time))

        np.savez(os.path.join(self.path, "data_%s" % self.N), **data_dic)
        np.savez(os.path.join(self.path, "label_%s" % self.N), **label_dic)
    
    def _get_data_info(self, data_dic, label_dic):
        """获取数据集信息
        Args:
            data_dic: dict{str: np.array}
            label_dic: dict{str: np.array}
        Return:
            info: dict{str: int},
                    single_fault_num
                    compound_fault_num
                    signal_len
                    label_len
        """
        info = {}
        # 单一故障样本数, 包括正常数据
        info["single_fault_num"] = 0
        # 并发故障样本数
        info["compound_fault_num"] = 0
        for key in data_dic:
            if "Normal" in key:
                info["single_fault_num"] += data_dic[key].shape[0]
            else:
                info["compound_fault_num"] += data_dic[key].shape[0]
        # 样本信号长度, 每一通道
        info["signal_len"] = data_dic["1_Normal_Normal"].shape[-1]
        # 样本标签长度
        info["label_len"] = label_dic["1_Normal_Normal"].shape[-1]

        return info

    def _fft(self, data, sample_frequency=5120):
        '''FFT for signal matrix
        Args:
            data: np.array, (batch_size, channel, length)
            sample_frequency: 采样频率
        Return:
            xf: np.array, 对应频率(Hz)
            yf: np.array, 幅值(未预处理)
        '''
        # 采样周期
        T = 1 / sample_frequency
        x = np.linspace(0, self.N * T, self.N)
        # 快速傅里叶变换,取模
        yf = np.abs(np.fft.fft(data)) / ((len(x) / 2))
        # 由于对称性，只取一半区间
        yf = yf[:, :, : self.N // 2]
        xf = np.linspace(0.0, 1.0 / (2.0 * T), self.N // 2)

        return xf, yf


def main():
    """test"""
    data_loader = DataLoader()
    data_loader.load_data()


if __name__ == "__main__":
    main()
