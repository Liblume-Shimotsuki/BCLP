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


import h5py
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class Preprocess:
    """
    数据预处理类
    """
    def __init__(self, args):
        """
        初始化
        :param args: 初始化信息
        """
        self.args = args

    def data_preprocess(self):
        """
        数据预处理主函数，返回完成预处理数据集
        """
        f = h5py.File(self.args.matFilename1)
        batch = f['batch']
        num_cells = batch['summary'].shape[0]
        bat_dict = {}
        for i in range(num_cells):
            cl = f[batch['cycle_life'][i, 0]].value
            policy = f[batch['policy_readable'][i, 0]].value.tobytes()[::2].decode()
            summary_IR = np.hstack(f[batch['summary'][i, 0]]['IR'][0, :].tolist())
            summary_QC = np.hstack(f[batch['summary'][i, 0]]['QCharge'][0, :].tolist())
            summary_QD = np.hstack(f[batch['summary'][i, 0]]['QDischarge'][0, :].tolist())
            summary_TA = np.hstack(f[batch['summary'][i, 0]]['Tavg'][0, :].tolist())
            summary_TM = np.hstack(f[batch['summary'][i, 0]]['Tmin'][0, :].tolist())
            summary_TX = np.hstack(f[batch['summary'][i, 0]]['Tmax'][0, :].tolist())
            summary_CT = np.hstack(f[batch['summary'][i, 0]]['chargetime'][0, :].tolist())
            summary_CY = np.hstack(f[batch['summary'][i, 0]]['cycle'][0, :].tolist())
            summary = {'IR': summary_IR, 'QC': summary_QC, 'QD': summary_QD, 'Tavg':
                summary_TA, 'Tmin': summary_TM, 'Tmax': summary_TX, 'chargetime': summary_CT,
                       'cycle': summary_CY}
            cycles = f[batch['cycles'][i, 0]]
            cycle_dict = {}
            for j in range(cycles['I'].shape[0]):
                I = np.hstack((f[cycles['I'][j, 0]].value))
                Qc = np.hstack((f[cycles['Qc'][j, 0]].value))
                Qd = np.hstack((f[cycles['Qd'][j, 0]].value))
                Qdlin = np.hstack((f[cycles['Qdlin'][j, 0]].value))
                T = np.hstack((f[cycles['T'][j, 0]].value))
                Tdlin = np.hstack((f[cycles['Tdlin'][j, 0]].value))
                V = np.hstack((f[cycles['V'][j, 0]].value))
                dQdV = np.hstack((f[cycles['discharge_dQdV'][j, 0]].value))
                t = np.hstack((f[cycles['t'][j, 0]].value))
                cd = {'I': I, 'Qc': Qc, 'Qd': Qd, 'Qdlin': Qdlin, 'T': T, 'Tdlin': Tdlin, 'V': V, 'dQdV': dQdV, 't': t}
                cycle_dict[str(j)] = cd
            cell_dict = {'cycle_life': cl, 'charge_policy': policy, 'summary': summary, 'cycles': cycle_dict}
            key = 'b1c' + str(i)
            bat_dict[key] = cell_dict

        bat_dict1 = bat_dict

        f = h5py.File(self.args.matFilename2)

        batch = f['batch']

        num_cells = batch['summary'].shape[0]
        bat_dict = {}
        for i in range(num_cells):
            cl = f[batch['cycle_life'][i, 0]].value
            policy = f[batch['policy_readable'][i, 0]].value.tobytes()[::2].decode()
            summary_IR = np.hstack(f[batch['summary'][i, 0]]['IR'][0, :].tolist())
            summary_QC = np.hstack(f[batch['summary'][i, 0]]['QCharge'][0, :].tolist())
            summary_QD = np.hstack(f[batch['summary'][i, 0]]['QDischarge'][0, :].tolist())
            summary_TA = np.hstack(f[batch['summary'][i, 0]]['Tavg'][0, :].tolist())
            summary_TM = np.hstack(f[batch['summary'][i, 0]]['Tmin'][0, :].tolist())
            summary_TX = np.hstack(f[batch['summary'][i, 0]]['Tmax'][0, :].tolist())
            summary_CT = np.hstack(f[batch['summary'][i, 0]]['chargetime'][0, :].tolist())
            summary_CY = np.hstack(f[batch['summary'][i, 0]]['cycle'][0, :].tolist())
            summary = {'IR': summary_IR, 'QC': summary_QC, 'QD': summary_QD, 'Tavg':
                summary_TA, 'Tmin': summary_TM, 'Tmax': summary_TX, 'chargetime': summary_CT,
                       'cycle': summary_CY}
            cycles = f[batch['cycles'][i, 0]]
            cycle_dict = {}
            for j in range(cycles['I'].shape[0]):
                I = np.hstack((f[cycles['I'][j, 0]].value))
                Qc = np.hstack((f[cycles['Qc'][j, 0]].value))
                Qd = np.hstack((f[cycles['Qd'][j, 0]].value))
                Qdlin = np.hstack((f[cycles['Qdlin'][j, 0]].value))
                T = np.hstack((f[cycles['T'][j, 0]].value))
                Tdlin = np.hstack((f[cycles['Tdlin'][j, 0]].value))
                V = np.hstack((f[cycles['V'][j, 0]].value))
                dQdV = np.hstack((f[cycles['discharge_dQdV'][j, 0]].value))
                t = np.hstack((f[cycles['t'][j, 0]].value))
                cd = {'I': I, 'Qc': Qc, 'Qd': Qd, 'Qdlin': Qdlin, 'T': T, 'Tdlin': Tdlin, 'V': V, 'dQdV': dQdV, 't': t}
                cycle_dict[str(j)] = cd

            cell_dict = {'cycle_life': cl, 'charge_policy': policy, 'summary': summary, 'cycles': cycle_dict}
            key = 'b2c' + str(i)
            bat_dict[key] = cell_dict

        bat_dict2 = bat_dict

        f = h5py.File(self.args.matFilename3)

        batch = f['batch']

        num_cells = batch['summary'].shape[0]
        bat_dict = {}
        for i in range(num_cells):
            cl = f[batch['cycle_life'][i, 0]].value
            policy = f[batch['policy_readable'][i, 0]].value.tobytes()[::2].decode()
            summary_IR = np.hstack(f[batch['summary'][i, 0]]['IR'][0, :].tolist())
            summary_QC = np.hstack(f[batch['summary'][i, 0]]['QCharge'][0, :].tolist())
            summary_QD = np.hstack(f[batch['summary'][i, 0]]['QDischarge'][0, :].tolist())
            summary_TA = np.hstack(f[batch['summary'][i, 0]]['Tavg'][0, :].tolist())
            summary_TM = np.hstack(f[batch['summary'][i, 0]]['Tmin'][0, :].tolist())
            summary_TX = np.hstack(f[batch['summary'][i, 0]]['Tmax'][0, :].tolist())
            summary_CT = np.hstack(f[batch['summary'][i, 0]]['chargetime'][0, :].tolist())
            summary_CY = np.hstack(f[batch['summary'][i, 0]]['cycle'][0, :].tolist())
            summary = {'IR': summary_IR, 'QC': summary_QC, 'QD': summary_QD, 'Tavg':
                summary_TA, 'Tmin': summary_TM, 'Tmax': summary_TX, 'chargetime': summary_CT,
                       'cycle': summary_CY}
            cycles = f[batch['cycles'][i, 0]]
            cycle_dict = {}
            for j in range(cycles['I'].shape[0]):
                I = np.hstack((f[cycles['I'][j, 0]].value))
                Qc = np.hstack((f[cycles['Qc'][j, 0]].value))
                Qd = np.hstack((f[cycles['Qd'][j, 0]].value))
                Qdlin = np.hstack((f[cycles['Qdlin'][j, 0]].value))
                T = np.hstack((f[cycles['T'][j, 0]].value))
                Tdlin = np.hstack((f[cycles['Tdlin'][j, 0]].value))
                V = np.hstack((f[cycles['V'][j, 0]].value))
                dQdV = np.hstack((f[cycles['discharge_dQdV'][j, 0]].value))
                t = np.hstack((f[cycles['t'][j, 0]].value))
                cd = {'I': I, 'Qc': Qc, 'Qd': Qd, 'Qdlin': Qdlin, 'T': T, 'Tdlin': Tdlin, 'V': V, 'dQdV': dQdV, 't': t}
                cycle_dict[str(j)] = cd

            cell_dict = {'cycle_life': cl, 'charge_policy': policy, 'summary': summary, 'cycles': cycle_dict}
            key = 'b3c' + str(i)
            bat_dict[key] = cell_dict

        bat_dict3 = bat_dict

        return bat_dict1, bat_dict2, bat_dict3


