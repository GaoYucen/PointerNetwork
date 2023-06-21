# Ptr-net

Structure:
- code:
  - Data_Generator.py: 生成optimal tsp方案
    - func tsp_opt(): 针对一个instance生成optimal tsp
    - class TSPDataset(): 生成一组包含点和Solutions的序列
  - config.py: 利用argparse配置超参数
  - Train.py: 训练模型
    - 调用TSPDataset生成数据集，可存储与读取数据集
    - 利用CrossEntropyLoss训练模型
    - 测试模型性能
- data:
  - train.pkl: 以dict形式存储的规模为1000的训练集
  - test.pkl：测试集
 
环境配置:
python 3.8
torch 1.12.1
