# REPO 提交规范

项目和代码的规范和可读性对于项目开发至关重要，可以提升开发效率。本文给出开发者在新建开发者生态项目时的repo目录示例，以供参考。

本文示例项目在文件夹[repo_template](https://github.com/thinkenergy/vloong-nature-energy/blob/master/repo_template)下，您可以将这个文件夹中的内容拷贝出去，放在自己的项目文件夹下，并编写对应的代码与文档。

注意： 该模板中仅给出了必要的代码结构，剩余部分，如模型组网、损失函数、数据处理等，与参考repo中的结构尽量保持一致，便于复现即可。



## 1. 目录结构

建议的目录结构如下：

```Plaintext
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
|-- main.py                   # 项目主文件
|-- README.md                 # 中文用户手册
|-- LICENSE                   # LICENSE文件
```

- config：项目配置类文件夹，包含项目配置信息等文件

- dataset：数据集类文件夹，包含构建数据集等代码文件

- log：日志类文件夹，包含训练日志等文件

- model：模型类文件夹，包含训练好的模型文件

- preprocess：预处理类文件夹，包含数据预处理等代码文件

- tools： 工具类文件夹，包含训练、验证等代码文件

- main.py：项目主文件，可进行项目全流程（训练+验证）执行过程

- README.md： 中文版当前模型的使用说明，规范参考 README 内容要求

- LICENSE： LICENSE文件

## 2. 功能实现

模型需要提供的功能包含：

- 训练：可以在GPU单机单卡的环境下执行训练

- 验证：可以在GPU单机单卡的环境下执行验证

- 项目主文件全流程执行：可以进行项目全流程（训练+验证）执行过程，并在GPU单机单卡的环境下执行

- 模型保存：保存已完成训练的模型，并且模型是可以根据参赛队伍提供的代码重新生成的，不能存在任何通过黑箱得到的模型，否则将认定为作弊行为

## 3. 命名规范和使用规范

- 文件和文件夹命名中，尽量使用下划线`_`代表空格，不要使用`-`。

- 模型定义过程中，需要有一个统一的变量（parameter）命名管理手段，如尽量手动声明每个变量的名字并支持名称可变，禁止将名称定义为一个常数（如"embedding"），避免在复用代码阶段出现各种诡异的问题。

- 重要文件，变量的名称定义过程中需要能够通过名字表明含义，禁止使用含混不清的名称，如net.py, aaa.py等。

- 在代码中定义path时，需要使用os.path.join完成，禁止使用string加的方式，导致模型对windows环境缺乏支持。



## 4. 注释和License

对于代码中重要的部分，需要加入注释介绍功能，帮助用户快速熟悉代码结构，包括但不仅限于：

- Dataset、DataLoader的定义。

- 整个模型定义，包括input，运算过程，loss等内容。

- init，save，load，等io部分

- 运行中间的关键状态，如print loss，save model等。

如：

```undefined
# encoding=utf8
# Copyright (c) 2022 Circue Authors. All Rights Reserved

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import pandas as pd
from torch.utils.data import Dataset


class BatteryDataset(Dataset):
    """
    Battery数据集定义
    """
    def __init__(self,data_path):
        """
        初始化
        :param data_path: 数据路径 str
        """
        super().__init__()
        self.install_data = pd.read_csv(data_path, encoding='utf8', sep=',') # 加载原始数据
        
    def __len__(self):
        """
        返回整个数据集大小
        """
        return len(self.install_data)
    
    def __getitem__(self, idx):
        """
        根据索引idx返回dataset[idx]
        :param idx: 数据索引 int
        """
        input_data = pd.DataFrame([])
        label = pd.DataFrame([])
        
        input_data = self.install_data.iloc[idx, 1] # 获取input数据
        label = self.install_data.iloc[idx, 0] # 获取数据集label
        return label, input_data
```

对于整个模型代码，都需要在文件头内加入licenses，readme中加入licenses标识。

文件头内LICENSE格式如下：

```Plaintext
# encoding=utf8
# Copyright (c) 2022 Circue Authors. All Rights Reserved

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```



## 5.数据集划分说明

NatureEnergy数据共有三个数据集：

**batch1**：2017-05-12_batchdata_updated_struct_errorcorrect.mat        **cell数**：numBatch1 = 41

**batch2**：2017-06-30_batchdata_updated_struct_errorcorrect.mat        **cell数**：numBatch1 = 43

**batch3**：2018-04-12_batchdata_updated_struct_errorcorrect.mat        **cell数**：numBatch1 = 40

需要将三个batch的数据合并后划分为**train**、**primary_test**、**secondary_test**三个数据集，数据集**必须**按照以下要求划分：

```undefined
# train由(batch1+batch2)中索引编号为单数的cell组成数据集 [1,3,5,7,9,11,...,81,83]
train_idx = np.arange(1, (numBatch1 + numBatch2), 2)

# primary_test由(batch1+batch2)中索引编号为双数的cell组成数据集 [0,2,4,6,8,10,...,80,82]
primary_test_idx = np.arange(0, (numBatch1 + numBatch2), 2)

# secondary_test由batch3中的所有cell组成 [84,85,86,87,88,...,123]
secondary_test_idx = np.arange((numBatch1 + numBatch2), 124)
```



## 6. 其他问题

- 代码封装得当，易读性好，不用一些随意的变量/类/函数命名

- 注释清晰，不仅说明做了什么，也要说明为什么这么做

- 随机控制，需要尽量固定含有随机因素模块的随机种子，保证模型可以正常复现

- 数据读取路径需写在项目配置信息文件competition.json中，并可进行调用。



## 7. README 内容&格式说明

模型的readme共分为以下几个部分，可以参考模板见：[README模板](https://github.com/thinkenergy/vloong-nature-energy/blob/master/README模板.md) 

```Plaintext
# 模型名称
## 1. 简介
## 2. 数据集和复现精度
## 3. 准备数据与环境
## 4. 开始使用
## 5. 代码结构与简要说明
## 6. LINENCE
## 7. 参考链接与文献
```



## 8. train.log 日志文件格式说明

参赛队伍若使用神经网络模型则需提供训练日志(train.log)文件，训练日志中需要包含当前的epoch和iter数量，当前loss、当前数据集精度

```undefined
from loguru import logger

logger.add("./train.log",
                   rotation="500MB",
                   encoding="utf-8",
                   enqueue=True,
                   retention="10 days")
                   
for i in range(100):    # 以总共训练100轮为例
    logger.info(f"epoch{i+1}/100:")
    logger.info("train_loss:")
    logger.info("train_mape:")
```