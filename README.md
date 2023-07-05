# Ptr-net

Pytorch implementation of Pointer Network.



Structure:

- code:
  - Data_Generator.py: 调用TSPDataset生成训练和测试数据集
    - func tsp_opt(): 针对一个instance生成optimal tsp，结果是形为(length, solution)的tuple
    - class TSPDataset(): 生成一组包含点集Points和解集Solutions的序列，形为{'Points_list': [np.array], 'Solutions': [np.array]}的dict
  - PointerNet.py: 配置Ptr-net模型的Encoder-Decoder结构
  - config.py: 利用argparse配置超参数
  - Train_test.py: 训练和测试模型
    - 利用CrossEntropyLoss训练模型
    - 对测试集测试模型性能
    - 参数设置
      - --test_flag: 定义是否是Test Mode
      - --sys: 定义是否是mac系统，以便使用MPS【但因为torch2.0后才支持mps，需要进一步修改code】
- data:
  - train.npy: 以dict形式存储的规模为1000的训练集
  - test.npy：测试集
- param:
  - param_5_100.pkl: 保存模型参数, 分别表示num_points和num_epochs



Environment:
- python 3.8.16
- torch 1.2.1
- numpy 1.23.5
- tqdm 4.64.1



To learn:

- Meaning of the following code

```python
o = o.contiguous().view(-1, o.size()[-1])
target_batch = target_batch.view(-1)
```



To modify:

- length calculation for the test dataset
- Warnings



More discussions:

- Setting of *num_workers*
