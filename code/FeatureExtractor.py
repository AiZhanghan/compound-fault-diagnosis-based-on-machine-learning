import os
import numpy as np
import pandas as pd

from DataLoader import DataLoader


class FeatureExtractor:
    """特征提取器"""

    def extract_feature(self, datas):
        """从datas(1个风机)的各个传感器(sensor)的振动信号(data)中提取特征
        
        Args:
            datas: dict{sensor: data}
                key: sensor, str, 测点
                value: data, np.array, (sample_num, signal_len)
        
        Return:
            feature: np.array
        """
        features = {}
        for sensor, data in datas.items():
            features[sensor] = self._extract_feature(data)
        feature = pd.concat(features, axis=1)
        return feature.values
            
    def _extract_feature(self, data):
        """提取某个测点信号(data)的feature

        Args:
            data: json
        
        Return:
            feature: pd.DataFrame
        """
        df = pd.DataFrame(data)
        time_feature = df.apply(self._time_feature, axis=1)
        frequent_feature = df.apply(self._frequency_feature, axis=1)
        
        feature = pd.concat([time_feature, frequent_feature], axis=1)
        
        return frequent_feature

    def _time_feature(self, x):
        """提取x的时域统计特征

        Args:
            x: pd.Series
        
        Return:
            pd.Series
        """
        mean = x.mean()
        sd = x.std()
        root = (np.sum(np.sqrt(np.abs(x))) / len(x)) ** 2
        rms = np.sqrt(np.sum(x ** 2) / len(x))
        peak = np.max(np.abs(x))
        
        skewness = x.skew()
        kurtosis = x.kurt()
        crest = peak / rms
        clearance = peak / root
        shape = rms / (np.sum(np.abs(x)) / len(x))
        
        impluse = peak / (np.sum(np.abs(x)) / len(x))
    
        feature = pd.Series([mean, sd, root, rms, peak, 
                             skewness, kurtosis, crest, clearance, shape, 
                             impluse], 
                            index = ['mean', 'sd', 'root', 'rms', 'peak', 
                                     'skewness', 'kurtosis', 'crest', 
                                     'clearance', 'shape', 'impluse'])
        
        return feature
    
    def _frequency_feature(self, y):
        """提取y的频域统计特征

        Args:
            y: pd.Series
        
        Return:
            pd.Series
        """
        
        # 采样点数
        N = len(y)
        sample_frequency = 5120
        # 传感器采样周期
        T = 1 / sample_frequency 
        #快速傅里叶变换,取模
        yf = 2.0 / N * np.abs(np.fft.fft(y))
        #由于对称性，只取一半区间
        yf = yf[: N // 2]  
        xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
        # 频域常用统计特征
        K = len(yf)
        # spectrum line
        s = yf 
        # frequency value
        f = xf 
        
        p1 = np.sum(s) / K
        p2 = np.sum((s - p1) ** 2) / (K - 1)
        p3 = np.sum((s - p1) ** 3) / (K * (np.sqrt(p2) ** 3))
        p4 = np.sum((s - p1) ** 4) / (K * p2 ** 2)
        p5 = np.sum(f * s) / np.sum(s)
        p6 = np.sqrt(np.sum((f - p5) ** 2 * s) / K)
        p7 = np.sqrt(np.sum(f ** 2 * s) / np.sum(s))
        p8 = np.sqrt(np.sum(f ** 4 * s) / np.sum(f ** 2 * s))
        p9 = np.sum(f ** 2 * s) / np.sqrt(np.sum(s) * np.sum(f ** 4 * s))
        p10 = p6 / p5
        p11 = np.sum((f - p5) ** 3 * s) / (K * p6 ** 3)
        p12 = np.sum((f - p5) ** 4 * s) / (K * p6 ** 4)
        p13 = np.sum(np.sqrt(np.abs(f - p5)) * s) / (K * np.sqrt(p6))
        p14 = np.sqrt(np.sum((f - p5) ** 2 * s) / np.sum(s))
        
        feature = pd.Series([p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, 
                             p12, p13, p14],
                            index = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7',
                                     'p8', 'p9', 'p10', 'p11', 'p12', 'p13',
                                     'p14'])
        
        return feature


def split_sensor(data):
    """根据传感器把3d array转换成2d array
    
    Args:
        data: np.array
    
    Return:
        dict{sensor: np.array}
            sens
    """
    dic = {}
    for i in range(data.shape[1]):
        dic["sensor" + str(i)] = data[:, i, :]
    return dic


def main():
    path = r"./feature"
    data_dic, _, _ = DataLoader().get_time_data()
    
    feature = {}
    label = {}

    for key, value in data_dic.items():
        feature[key] = FeatureExtractor().extract_feature(split_sensor(value))
        label[key] = np.ones([value.shape[0], 1]) * \
                      (int(key.split("_")[0]) - 1)

    np.savez(os.path.join(path, "frequent_feature.npz"), **feature)
    # np.savez(os.path.join(path, "time_label.npz"), **label)


if __name__ == "__main__":
    main()