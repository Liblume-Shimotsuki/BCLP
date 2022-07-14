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


import sys
import os
import json
import warnings
import joblib
import argparse
import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.getcwd()))

from dataset.dataset import Dataset
from tools.train import Train
from tools.eval import Eval
warnings.filterwarnings('ignore')

class Main:
    """
    算法主程序类
    """
    def __init__(self, args):
        """
        初始化
        :param args: 初始化信息
        """
        self.args = args

    def run(self):
        """
        运行算法主程序
        """
        # Full
        print("Full")
        features_full = Dataset(self.args, regression_type="full").get_feature()
        mode_full = Train(self.args)
        mode_full.regression(features_full, alpha_train=self.args.alpha_train,
                        regression_type="full",
                        l1_ratio=self.args.l1_ratio, log_target=self.args.log_target, model="elastic")

        Eval(self.args, model = mode_full.model).evaluation(features_full,
                        regression_type="full",
                        log_target=self.args.log_target)

        # Discharge
        print("\nDischarge")
        features_discharge = Dataset(self.args, regression_type="discharge").get_feature()
        mode_discharge = Train(self.args)
        mode_discharge.regression(features_discharge, alpha_train=self.args.alpha_train,
                                    regression_type="discharge",
                                    l1_ratio=self.args.l1_ratio, log_target=self.args.log_target, model="elastic")

        Eval(self.args, model = mode_discharge.model).evaluation(features_discharge,
                                   regression_type="discharge",
                                   log_target=self.args.log_target)

        # Variance
        print("\nVariance")
        features_variance = Dataset(self.args, regression_type="variance").get_feature()
        mode_variance = Train(self.args)
        mode_variance.regression(features_variance, alpha_train=self.args.alpha_train,
                                    regression_type="variance",
                                    l1_ratio=self.args.l1_ratio, log_target=self.args.log_target, model="elastic")

        Eval(self.args, model=mode_variance.model).evaluation(features_variance,
                                   regression_type="variance",
                                   log_target=self.args.log_target)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train Example')
    parser.add_argument('--config_path', type=str,
                        default='./config/competition.json')
    args = parser.parse_args()

    with open(args.config_path, 'r') as file:
        p_args = argparse.Namespace()
        p_args.__dict__.update(json.load(file))
        args = parser.parse_args(namespace=p_args)

    Main(args).run()
