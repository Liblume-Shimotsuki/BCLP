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


import pandas as pd
import sys
import os
import json
import warnings
import joblib
import json
from sklearn.metrics import mean_absolute_percentage_error
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.getcwd()))
from dataset.dataset import Dataset


warnings.filterwarnings('ignore')

class Eval:
    """
    验证类
    """
    def __init__(self, args, model=None):
        """
        初始化
        :param args: 初始化信息
        """
        self.model = model
        self.args = args


    def evaluation(self, datasets, regression_type, log_target=True, load_model = False):
        """
        验证函数
        :param datasets: 数据集
        :param regression_type: 回归模型类型
        :param log_target: 是否进行log变换
        """
        # get three sets
        x_val, y_val = datasets.get("val")
        x_test, y_test = datasets.get("test")
        if load_model:
            regr = joblib.load(f"../model/{regression_type}_regression.pkl")
        else:
            regr = self.model

        # predict values/cycle life for all three sets
        pred_val = regr.predict(x_val)
        pred_test = regr.predict(x_test)

        if log_target:
            # scale up the preedictions
            pred_val, pred_test = np.exp(pred_val), np.exp(pred_test)

        # mean percentage error (same as paper)
        error_val = mean_absolute_percentage_error(y_val, pred_val) * 100
        error_test = mean_absolute_percentage_error(y_test, pred_test) * 100

        print(f"Regression Error (validation (primary) test): {error_val}%")
        print(f"Regression Error batch 3 (test (secondary)): {error_test}%")

    def run_evaluation(self):
        """
        验证回归模型主参数
        """
        print("Full")
        features = Dataset(self.args, regression_type="full").get_feature()
        self.evaluation(features,
                        regression_type="full",
                        log_target=self.args.log_target)

        print("\nDischarge")
        features = Dataset(self.args, regression_type="discharge").get_feature()
        self.evaluation(features,
                        regression_type="discharge",
                        log_target=self.args.log_target)

        print("\nVariance")
        features = Dataset(self.args, regression_type="variance").get_feature()
        self.evaluation(features,
                        regression_type="variance",
                        log_target=self.args.log_target)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train Example')
    parser.add_argument('--config_path', type=str,
                        default='../config/competition.json')
    args = parser.parse_args()

    with open(args.config_path, 'r') as file:
        p_args = argparse.Namespace()
        p_args.__dict__.update(json.load(file))
        args = parser.parse_args(namespace=p_args)

    # 创建Eval实例，run_evaluation()开始评估
    Eval(args).run_evaluation()
