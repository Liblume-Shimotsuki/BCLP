# README模板

## 1. 简介

- 简单的介绍模型，以及模型的主要架构或主要功能，可以在简介的下方直接贴上图片，展示模型结构。

- 论文标题: Data-driven prediction of battery cycle life before capacity degradation 

- 复现所用到的评价指标以及所达到的精度。



## 2. 数据集和复现精度

- 给出本repo中用到的数据集的基本信息，例如数据集大小与数据集格式。格式如下：
  - 数据集大小：关于数据集大小的描述，如类别，数量等等
  - 数据格式：关于数据集格式的说明

- 基于上述数据集，给出论文中精度、参考代码的精度、本repo复现的精度、数据集名称、模型大小，以表格的形式给出。如果超参数有差别，可以在表格中新增一列备注一下。



## 3. 准备数据与环境

### 3.1 准备环境

- 建议将代码中用到的非python原生的库，都写在requirements.txt中，直接使用`pip install -r requirements.txt`安装依赖即可。

### 3.2 准备数据

- 简单介绍下对数据进行了哪些操作，例如数据预处理、train&test数据集选择等。



## 4. 开始使用

### 4.1 模型训练

- 简单说明一下训练（train.py）的命令，建议附一些简短的训练日志。

- 可以简要介绍下可配置的超参数以及配置方法。

### 4.2 模型验证

- 在这里简单说明一下验证（eval.py）的命令，需要提供原始数据等内容，并在文档中体现输出结果。

### 4.3 项目主文件
- 在这里简单说明一下项目主文件（main.py）的命令，main.py中可执行全流程（train+eval）过程。

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

  