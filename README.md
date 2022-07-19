# README模板

## 1. 简介

- 本次使用的模型为Kernel Ridge Regression即使用核技巧的岭回归（L2正则线性回归），它的学习形式和SVR（support vector regression）相同，但是两者的损失函数不同。

- 论文标题: Data-driven prediction of battery cycle life before capacity degradation 

- 复现所用到的评价指标为MAPE, 即 mean absolute percentage error. 在 Primary test 数据集(已移除Cycle Life 为148的异常样本)上, MAPE=8.07%, 在 Secondary test 数据集上, MAPE=9.14%, 综合MAPE=8.60%.

## 2. 数据集和复现精度

- 给出本repo中用到的数据集的基本信息，例如数据集大小与数据集格式。格式如下：
  
  - 数据集大小：
  
  batch1：2017-05-12_batchdata_updated_struct_errorcorrect.mat cell数：numBatch1 = 41
  
  batch2：2017-06-30_batchdata_updated_struct_errorcorrect.mat cell数：numBatch1 = 43
  
  batch3：2018-04-12_batchdata_updated_struct_errorcorrect.mat cell数：numBatch1 = 40
  
  将三个batch的数据合并后划分为train、primary_test、secondary_test三个数据集，划分方式如下
  
  ```
  # train由(batch1+batch2)中索引编号为单数的cell组成数据集 [1,3,5,7,9,11,...,81,83]
  train_idx = np.arange(1, (numBatch1 + numBatch2), 2)
  
  # primary_test由(batch1+batch2)中索引编号为双数的cell组成数据集 [0,2,4,6,8,10,...,80,82]
  primary_test_idx = np.arange(0, (numBatch1 + numBatch2), 2)
  
  # secondary_test由batch3中的所有cell组成 [84,85,86,87,88,...,123]
  secondary_test_idx = np.arange((numBatch1 + numBatch2), 124)
  ```
  
  另外, 根据论文中的描述, 在给出MAPE时移除了一个寿命过短的cell.
  
  > One battery in the test set reaches 80% state-of-health rapidly and does not
  > match other observed patterns. Therefore, the parenthetical primary test results correspond to the exclusion of this battery
  
  因此, 在也进行了相应的操作.
  
  ```
  primary_test_idx = primary_test_idx.tolist()
  primary_test_idx.remove(42)
  ```
  
  - 数据格式：关于数据集格式的说明

- 基于上述数据集，给出论文中精度、参考代码的精度、本repo复现的精度、数据集名称、模型大小，以表格的形式给出。如果超参数有差别，可以在表格中新增一列备注一下。

|                | 论文精度  | 参考代码精度 | 本repo复现精度 |
| -------------- |:-----:|:------:|:---------:|
| Train          | 5.6%  | 17.2%  | 6.26%     |
| Primary test   | 7.5%  | 15.4%  | 8.07%     |
| Secondary test | 10.7% | 16.0%  | 9.14%     |
| 综合MAPE       | 9.1%  | 15.7%   | 8.60%     |
## 3. 准备数据与环境

### 3.1 准备环境

```
pip install -r requirements.txt
```

### 3.2 准备数据

- 简单介绍下对数据进行了哪些操作，例如数据预处理、train&test数据集选择等。

## 4. 开始使用

### 4.1 模型训练

```
python tools/train.py --config_path ./config/competition.json
```

```
Loading pkl from disk ...
Loading batches ...
Done loading batches
Start building features ...
Done building features
[13:50:54] WARNING: ../src/learner.cc:627: 
Parameters: { "silent" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


Regression Error (Train): 6.262108953796329%
```

### 4.2 模型验证

```
python tools/eval.py --config_path ./config/competition.json
```

```
Loading pkl from disk ...
Loading batches ...
Done loading batches
Start building features ...
Done building features
Regression Error (validation (primary) test): 8.068955692662822%
Regression Error batch 3 (test (secondary)): 9.140820268353899%
```

- 在这里简单说明一下验证（eval.py）的命令，需要提供原始数据等内容，并在文档中体现输出结果。

### 4.3 项目主文件

- 在这里简单说明一下项目主文件（main.py）的命令，main.py中可执行全流程（train+eval）过程。

```
python main.py --config_path ./config/competition.json
```

```
Namespace(AveragingModels={'KernelRidge': {'alpha': 0.98, 'kernel': 'polynomial', 'degree': 6, 'coef0': 3}, 'ElasticNet': {'random_state': 4, 'alpha': 0.005, 'l1_ratio': 0.9}, 'XGBRegressor': {'booster': 'gbtree', 'colsample_bytree': 0.8, 'gamma': 0.1, 'learning_rate': 0.02, 'max_depth': 5, 'n_estimators': 500, 'min_child_weight': 0.8, 'reg_alpha': 0, 'reg_lambda': 1, 'subsample': 0.8, 'silent': 1, 'random_state': 4, 'nthread': 2}}, config_path='./config/model_merge.json', log_target=False, matFilename1='/dataset/2017-05-12_batchdata_updated_struct_errorcorrect.mat', matFilename2='/dataset/2017-06-30_batchdata_updated_struct_errorcorrect.mat', matFilename3='/dataset/2018-04-12_batchdata_updated_struct_errorcorrect.mat')
Loading pkl from disk ...
Loading batches ...
Done loading batches
Start building features ...
Done building features
[13:43:29] WARNING: ../src/learner.cc:627: 
Parameters: { "silent" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


Regression Error (Train): 6.262108953796329%
Regression Error (validation (primary) test): 8.068955692662822%
Regression Error batch 3 (test (secondary)): 9.140820268353899%
```



## 5. 代码结构与简要说明

### 5.1 代码结构

- 列出代码目录结构

```undefined
./repo_template               # 项目文件夹名称，可以修改为自己的文件夹名称
|-- config                    # 配置类文件夹
|   ├── competition.json      # 项目配置信息文件
|-- dataset                   # 数据集类文件夹
|   ├── dataset.py            # 数据集代码文件
|-- log                       # 日志类文件夹
|   ├── train.log             # 训练日志文件
|-- model                     # 模型类文件夹
|   ├── full_regression.pkl   # 训练好的模型文件
|-- preprocess                # 预处理类文件夹
|   ├── preprocess.py         # 数据预处理代码文件
|-- tools                     # 工具类文件夹
|   ├── train.py              # 训练代码文件
|   ├── eval.py               # 验证代码文件
|   ├── averaging_model.py    # 模型融合
|-- main.py                   # 项目主文件
|-- README.md                 # 中文用户手册
|-- LICENSE                   # LICENSE文件
```

### 5.2 代码简要说明

- 说明代码文件中的类以及主要函数功能

```undefined
# 示例
./dataset.py               # 数据集代码文件
|-- class Dataset          # 数据集类
|   ├── get_feature        # 类主函数，返回可用于训练的数据集
|   ├── train_val_split    # 划分train&val数据集
```

## 6. LICENSE

- 本项目的发布受[Apache 2.0 license](https://github.com/thinkenergy/vloong-nature-energy/blob/master/LICENSE)许可认证。

## 7. 参考链接与文献

- **[vloong-nature-energy/repo_template at master thinkenergy/vloong-nature-energy](https://github.com/thinkenergy/vloong-nature-energy/tree/master/repo_template)**

- **[Data-driven prediction of battery cycle life before capacity degradation](https://doi.org/10.1038/s41560-019-0356-8)**