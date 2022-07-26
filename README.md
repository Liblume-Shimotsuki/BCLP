## 1. 简介

- 本次使用的模型为五个机器学习模型: SVR, ElasticNet, KernelRidge, XGBRegressor, GradientBoostingRegressor 以及一个深度学习模型的平均融合模型. 

- 论文标题: Data-driven prediction of battery cycle life before capacity degradation 

- 复现所用到的评价指标为MAPE, 即 mean absolute percentage error. 在 Primary test 数据集(已移除Cycle Life 为148的异常样本)上, MAPE=7.14%, 在 Secondary test 数据集上, MAPE=7.75%, 综合MAPE=7.45%.

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

<!-- - 基于上述数据集，给出论文中精度、参考代码的精度、本repo复现的精度、数据集名称、模型大小，以表格的形式给出。如果超参数有差别，可以在表格中新增一列备注一下。 -->

|                | 论文精度  | 参考代码精度 | 本repo复现精度 |
| -------------- |:-----:|:------:|:---------:|
| Train          | 5.6%  | 17.2%  | 5.83%     |
| Primary test   | 7.5%  | 15.4%  | 7.14%     |
| Secondary test | 10.7% | 16.0%  | 7.75%     |
| 综合MAPE         | 9.1%  | 15.7%  | 7.45%     |

## 3. 准备数据与环境

### 3.1 准备环境

```
pip install -r requirements.txt
```

### 3.2 准备数据

- 分析各个变量之间的相关系数,以及使用递归消除特征法对特征进行选取后, 最终保留了7个变量进行电池寿命的预测.
- 预处理采取MinMaxScaler对特征进行归一化.
- 对于深度学习模型, 将标签 cycle_life 也进行了归一化, 网络输出的预测结果反归一化后最为最终的预测数据. 

## 4. 开始使用

### 4.1 模型训练

只使用机器学习模型: 

```
python tools/train.py --config_path ./config/model_merge.json
```

```
Loading pkl from disk ...
Loading batches ...
Done loading batches
Start building features ...
Done building features
Regression Error (Train): 6.023036495326699%
```

使用机器学习模型结合深度学习模型:

```
python tools/train.py --config_path ./config/model_merge_nn.json
```

```
Loading pkl from disk ...
Loading batches ...
Done loading batches
Start building features ...
Done building features
2022-07-26 15:33:03.564762: I tensorflow/core/common_runtime/process_util.cc:146] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
2022-07-26 15:33:03.654814: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
Regression Error (Train): 5.835425833485559%
```

### 4.2 模型验证

只使用机器学习模型:

```
python tools/train.py --config_path ./config/model_merge.json
```

```
Loading pkl from disk ...
Loading batches ...
Done loading batches
Start building features ...
Done building features
Regression Error (validation (primary) test): 7.3246816703629785%
Regression Error batch 3 (test (secondary)): 7.76573372034495%
```

使用机器学习模型结合深度学习模型:

```
python tools/eval.py --config_path ./config/model_merge_nn.json
```

```
Loading pkl from disk ...
Loading batches ...
Done loading batches
Start building features ...
Done building features
2022-07-26 15:33:34.946047: I tensorflow/core/common_runtime/process_util.cc:146] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
2022-07-26 15:33:35.069670: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
Regression Error (validation (primary) test): 7.148588969925659%
Regression Error batch 3 (test (secondary)): 7.7557230037880425%34495%
```

### 4.3 项目主文件

只使用机器学习模型:

```
python main.py --config_path ./config/model_merge.json
```

```
Namespace(config_path='./config/model_merge.json', matFilename1='/dataset/2017-05-12_batchdata_updated_struct_errorcorrect.mat', matFilename2='/dataset/2017-06-30_batchdata_updated_struct_errorcorrect.mat', matFilename3='/dataset/2018-04-12_batchdata_updated_struct_errorcorrect.mat', model_cfg=[{'model_name': 'SVR', 'log_target': True, 'SVR': {'kernel': 'poly', 'degree': 3, 'coef0': 3, 'C': 0.02}}, {'model_name': 'ElasticNet', 'log_target': False, 'ElasticNet': {'random_state': 4, 'alpha': 0.005, 'l1_ratio': 0.9}}, {'model_name': 'KernelRidge', 'log_target': True, 'KernelRidge': {'kernel': 'polynomial', 'degree': 5, 'coef0': 3, 'alpha': 0.88}}, {'model_name': 'XGBRegressor', 'log_target': False, 'XGBRegressor': {'booster': 'gbtree', 'colsample_bytree': 0.8, 'gamma': 0.1, 'learning_rate': 0.02, 'max_depth': 3, 'n_estimators': 276, 'min_child_weight': 0.8, 'reg_alpha': 0, 'reg_lambda': 1, 'subsample': 0.8, 'random_state': 4, 'nthread': 2}}, {'model_name': 'GradientBoostingRegressor', 'log_target': True, 'GradientBoostingRegressor': {'n_estimators': 64, 'max_depth': 5, 'min_samples_split': 3, 'random_state': 4}}])
Loading pkl from disk ...
Loading batches ...
Done loading batches
Start building features ...
Done building features
Regression Error (Train): 6.023036495326699%
Regression Error (validation (primary) test): 7.3246816703629785%
Regression Error batch 3 (test (secondary)): 7.76573372034495%
```

使用机器学习模型结合深度学习模型:

```
python main.py --config_path ./config/model_merge_nn.json
```

```
Namespace(config_path='./config/model_merge_nn.json', matFilename1='/dataset/2017-05-12_batchdata_updated_struct_errorcorrect.mat', matFilename2='/dataset/2017-06-30_batchdata_updated_struct_errorcorrect.mat', matFilename3='/dataset/2018-04-12_batchdata_updated_struct_errorcorrect.mat', model_cfg=[{'model_name': 'SVR', 'log_target': True, 'SVR': {'kernel': 'poly', 'degree': 3, 'coef0': 3, 'C': 0.02}}, {'model_name': 'ElasticNet', 'log_target': False, 'ElasticNet': {'random_state': 4, 'alpha': 0.005, 'l1_ratio': 0.9}}, {'model_name': 'KernelRidge', 'log_target': True, 'KernelRidge': {'kernel': 'polynomial', 'degree': 5, 'coef0': 3, 'alpha': 0.88}}, {'model_name': 'XGBRegressor', 'log_target': False, 'XGBRegressor': {'booster': 'gbtree', 'colsample_bytree': 0.8, 'gamma': 0.1, 'learning_rate': 0.02, 'max_depth': 3, 'n_estimators': 276, 'min_child_weight': 0.8, 'reg_alpha': 0, 'reg_lambda': 1, 'subsample': 0.8, 'random_state': 4, 'nthread': 2}}, {'model_name': 'GradientBoostingRegressor', 'log_target': True, 'GradientBoostingRegressor': {'n_estimators': 64, 'max_depth': 5, 'min_samples_split': 3, 'random_state': 4}}, {'model_name': 'KerasRegressor', 'epochs': 700, 'batch_size': 16}])
Loading pkl from disk ...
Loading batches ...
Done loading batches
Start building features ...
Done building features
2022-07-26 15:32:31.817371: I tensorflow/core/common_runtime/process_util.cc:146] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
2022-07-26 15:32:31.999492: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
Regression Error (Train): 5.835425833485559%
Regression Error (validation (primary) test): 7.148589220252951%
Regression Error batch 3 (test (secondary)): 7.755723195887669%
```

## 5. 代码结构与简要说明

### 5.1 代码结构

- 代码目录结构

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
|   ├── averaging_model.py    # 定义模型文件
|-- main.py                   # 项目主文件
|-- README.md                 # 中文用户手册
|-- LICENSE                   # LICENSE文件
```

### 5.2 代码简要说明

```undefined
./dataset/dataset.py       # 数据集代码文件
|-- class Dataset          # 数据集类
|   ├── get_feature        # 类主函数，返回可用于训练的数据集
|   ├── train_val_split    # 划分train&val数据集
|   ├── data_normalize     # 对特征进行归一化
|   ├── get_label_scaler   # 获得预测标签的归一化器

./tools/train.py           # 模型训练
|-- class Train            # 训练
|   ├── manual_seed        # 设置随机种子
|   ├── regression         # 回归函数

./tools/eval.py            # 模型验证
|-- class Eval             # 验证类
|   ├── evaluation         # 验证函数

./tools/averaging_model.py # 模型类
|-- class AveragingModels  # 模型融合类
|   ├── fit                # 拟合数据
|   ├── predict            # 预测数据
|   ├── save               # 保存模型
|   ├── load               # 加载模型
|-- class OptionalModel    # 可设置对标签是否进行log变换
|   ├── fit                # 拟合数据
|   ├── predict            # 预测数据
|-- class OptionalNnModels # 深度神经网络模型类
|   ├── fit                # 拟合数据
|   ├── predict            # 预测数据
|-- build_nn               # 构建深度神经网络模型

```

## 6. LICENSE

- 本项目的发布受[Apache 2.0 license](https://github.com/thinkenergy/vloong-nature-energy/blob/master/LICENSE)许可认证。

## 7. 参考链接与文献

- **[vloong-nature-energy/repo_template at master thinkenergy/vloong-nature-energy](https://github.com/thinkenergy/vloong-nature-energy/tree/master/repo_template)**

- **[Data-driven prediction of battery cycle life before capacity degradation](https://doi.org/10.1038/s41560-019-0356-8)**
