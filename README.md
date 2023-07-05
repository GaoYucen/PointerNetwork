# Ptr-net

Pytorch implementation of Pointer Network.



Structure:

- code:
  - Data_Generator.py: 生成optimal tsp方案
    - func tsp_opt(): 针对一个instance生成optimal tsp，结果是形为(length, solution)的tuple
    - class TSPDataset(): 生成一组包含点集Points和解集Solutions的序列，形为{'Points_list': [np.array], 'Solutions': [np.array]}的dict
    - 调用TSPDataset生成训练和测试数据集，可存储与读取数据集
  - PointerNet.py: 配置Ptr-net模型的Encoder-Decoder结构
  - config.py: 利用argparse配置超参数
  - Train.py: 训练模型
    - 利用CrossEntropyLoss训练模型
  - Test.py: 测试模型
    - 对测试集测试模型性能
- data:
  - train.pkl: 以dict形式存储的规模为1000的训练集
  - test.pkl：测试集
- param:
  - param.pkl: 保存模型参数



Environment:
- python 3.8.16
- torch 1.12.1
- numpy 1.23.5
- tqdm 4.64.1
