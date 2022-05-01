import os
import pickle
import sys
import warnings
from datetime import datetime
import random

import argparse
import avalanche
from avalanche.models import avalanche_forward
from avalanche.training.plugins.evaluation import default_logger
import numpy as np

import torch
import torch.nn as nn

from avalanche.benchmarks.generators import tensor_scenario
from avalanche.evaluation.metrics import (
    ExperienceForgetting,
    StreamConfusionMatrix,
    accuracy_metrics,
    cpu_usage_metrics,
    disk_usage_metrics,
    loss_metrics,
    timing_metrics,
)
from avalanche.logging import InteractiveLogger, TensorboardLogger, TextLogger

from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.strategies import EWC, GEM,AGEM,BaseStrategy
from avalanche.training.plugins import AGEMPlugin
from avalanche.training.plugins.strategy_plugin import StrategyPlugin

from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss,Module
from torch.optim import Adam
from torch.optim import Optimizer, SGD

from torch.utils.data import DataLoader
from typing import Optional, Sequence, Union, List

warnings.filterwarnings("ignore")

now = datetime.now()
cur_time = now.strftime("%d-%m-%Y::%H:%M:%S")

class cbrsPLUGIN(AGEMPlugin):

    def __init__(self, patterns_per_experience: int, sample_size: int):
        """
        :param patterns_per_experience: number of patterns per experience in the
            memory.
        :param sample_size: number of patterns in memory sample when computing
            reference gradient.
        """

        super().__init__(patterns_per_experience, sample_size)

        self.full_class = set()
        self.class_cnt = {}
        self.buf_mem_max_size = 1000

    @torch.no_grad()
    def update_memory(self, dataset):
        pass

    def before_training_iteration(self, strategy, **kwargs):
        if len(self.buffers) > 0:
            strategy.model.train()
            strategy.optimizer.zero_grad()

            r_samples = self.get_buffer_random_samples(strategy.train_mb_size)

            if len(r_samples) == 0:
                replay_loss = 0
            else:
                xref = []
                yref = []
                tref = []
                for rsamp in r_samples:
                    xref.append(rsamp[0])
                    yref.append(rsamp[1])
                    tref.append(rsamp[-1])
                #print(type(xref))
                #print("xref : ", xref)
                #print(type(yref))
                #print("yref : ", yref)
                #print(type(tref))
                #print("tref : ", tref)
                xref = torch.stack(xref)
                yref = torch.Tensor(yref)
                tref = torch.Tensor(tref)
                xref, yref, tref = xref.to(strategy.device), yref.to(strategy.device), tref.to(strategy.device)

                r_pred = avalanche_forward(strategy.model, xref,tref)
                yref = yref.type(torch.int64)
                #print("yref :",yref)
                #print(type(yref))
                replay_loss = strategy._criterion(r_pred,yref)

            replay_loss.backward()
            self.reference_gradients = [
                    p.grad.view(-1) if p.grad is not None
                    else torch.zeros(p.numel(), device=strategy.device)
                    for n, p in strategy.model.named_parameters()]
            self.reference_gradients = torch.cat(self.reference_gradients)
            strategy.optimizer.zero_grad()


    def update_mem(self,instance,n_c):
        if instance[1] in self.class_cnt.keys():
            self.class_cnt[instance[1]] +=1
        else:
            self.class_cnt[instance[1]] =1

        self.full_class.add(max(self.class_cnt, key=self.class_cnt.get))

        if len(self.buffers) < self.buf_mem_max_size:
            self.buffers.append(instance)
        else :
            if instance[1] not in self.full_class :
                largest_cls = max(self.class_cnt, key=self.class_cnt.get)

                largest_cls_idxs = []

                for idx,buff in enumerate(self.buffers):
                    yval = int(buff[1])
                    #print(yval,largest_cls)
                    if yval == largest_cls:
                        largest_cls_idxs.append(idx)
                if len(largest_cls_idxs) > 0:
                    idx = random.choice(largest_cls_idxs)
                    self.buffers[idx] = instance
            else :
                mC = 0
                for buff in self.buffers :
                    yval = int(buff[1])
                    if int(yval) == int(instance[1]) :
                        mC += 1

                u = random.random()
                if u < mC/n_c :
                    req_cls_idxs = []

                    for idx,buff in enumerate(self.buffers):
                        yval = int(buff[1])
                        if yval == int(instance[1]):
                            req_cls_idxs.append(idx)

                    idx = random.choice(req_cls_idxs)
                    self.buffers[idx] = instance


    def get_buffer_random_samples(self,batch_size):
        if batch_size > len(self.buffers):
            empty_lst = []
            return empty_lst
        random_samples = random.sample(self.buffers,batch_size)
        return random_samples




class CBRS(BaseStrategy):
    """ Average Gradient Episodic Memory (A-GEM) strategy.

    See AGEM plugin for details.
    This strategy does not use task identities.
    """
    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 patterns_per_exp: int, sample_size: int = 64,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger, eval_every=-1):
        """ Init.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param patterns_per_exp: number of patterns per experience in the memory
        :param sample_size: number of patterns in memory sample when computing
            reference gradient.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        """

        cbrs = cbrsPLUGIN(patterns_per_exp, sample_size)
        if plugins is None:
            plugins = [cbrs]
        else:
            plugins.append(cbrs)

        self.nC = set()
        self.nb = 1
        self.stream_instances_encountered = {}
        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)
        #print("Plugins are :",self.plugins)
    def training_epoch(self, **kwargs):
        """ Training epoch.

        :param kwargs:
        :return:
        """
        for self.mbatch in self.dataloader:

            if self._stop_training:
                break

            self._unpack_minibatch()

            #print("type of mbatch : ",type(self.mbatch))
            #print("len of mbatch : ",len(self.mbatch))
            #print("mbatch : \n",self.mbatch)
            for lbl in self.mbatch[1].tolist():
                self.nC.add(lbl)
            a = 1/len(self.nC)

            self._before_training_iteration(**kwargs)

            self.optimizer.zero_grad()
            self.loss = 0

            # Forward
            self._before_forward(**kwargs)
            self.mb_output = self.forward()
            self._after_forward(**kwargs)

            # Loss & Backward
            self.loss += a*self.criterion()

            #for p in self.plugins:
            #    if isinstance(p,cbrsPLUGIN):
            #        r_samples = p.get_buffer_random_samples(self.train_mb_size)

            #if len(r_samples) == 0:
            #    replay_loss = 0
            #else:
            #    r_pred = avalanche_forward(self.model, r_samples[0], r_samples[-1])
            #    replay_loss = self._criterion(r_pred,r_samples[1])

            #self.loss += replay_loss*(1-a)

            self._before_backward(**kwargs)
            self.loss.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer.step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)

            for i in range(len(self.mbatch[1])) :
                index = int(self.mbatch[1][i])
                if index not in self.stream_instances_encountered :
                    self.stream_instances_encountered[index] = 1
                else :
                    self.stream_instances_encountered[index] += 1
                #print("Stream Instance: ",self.stream_instances_encountered)
                for p in self.plugins:
                    instance = [self.mbatch[0][i],int(self.mbatch[1][i]),int(self.mbatch[2][i])]
                    if isinstance(p,cbrsPLUGIN):
                        #print(self.mbatch[1][i])
                        #print(type(self.mbatch[1][i]))
                        #print("stream_instances_encountred :", self.stream_instances_encountered[self.mbatch[1][i]] )
                        p.update_mem(instance,self.stream_instances_encountered[index])



def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_architecture",
        type=str,
        default="GEM",
        help="The type (EWC|GEM) of the model architecture",
    )
    args = parser.parse_args(argv)
    return args


args = get_args(sys.argv[1:])

# Creating folder to save weights and logs
os.makedirs("weights", exist_ok=True)
os.makedirs("logs", exist_ok=True)

train_data_x = []
train_data_y = []
test_data_x = []
test_data_y = []

cache_label = "./data/train_data_x.pth"

# Normal = 0
# Attack = 1
if os.path.exists(cache_label):

    with open("./data/train_data_x.pth", "rb") as f:
        train_data_x = pickle.load(f)
    with open("./data/train_data_y.pth", "rb") as f:
        train_data_y = pickle.load(f)

    with open("./data/test_data_x.pth", "rb") as f:
        test_data_x = pickle.load(f)

    with open("./data/test_data_y.pth", "rb") as f:
        test_data_y = pickle.load(f)

else:
    with open("./data/ids_18.pth", "rb") as f:
        df = pickle.load(f)

    y = df.pop(df.columns[-1]).to_frame()

    # Replacing nans, infs with 0's
    df["Flow Byts/s"].replace([np.inf, -np.inf, -np.nan, np.nan], 0, inplace=True)
    df["Flow Pkts/s"].replace([np.inf, -np.inf, -np.nan, np.nan], 0, inplace=True)

    # Normalsing Dataset
    for column in df.columns:
        df[column] = (df[column] - df[column].min()) / (
            df[column].max() - df[column].min() + 1e-5
        )

    X_train, X_test, y_train, y_test = train_test_split(
        df, y, stratify=y, test_size=0.33
    )

    del df

    train_dict = {}
    train_label_dict = {}
    test_dict = {}
    test_label_dict = {}

    # Labelling classses as 0 or 1 based on type of class.
    for i in range(y_train.iloc[:, -1].nunique()):
        train_dict["cat" + str(i)] = X_train[y_train.iloc[:, -1] == i]

        temp = y_train[y_train.iloc[:, -1] == i]

        # Class label 0 = Normal class
        if i == 0:
            temp.iloc[:, -1] = 0
        else:
            temp.iloc[:, -1] = 1

        train_label_dict["cat" + str(i)] = temp

    for i in range(y_test.iloc[:, -1].nunique()):
        test_dict["cat" + str(i)] = X_test[y_test.iloc[:, -1] == i]

        temp = y_test[y_test.iloc[:, -1] == i]

        if i == 0:
            temp.iloc[:, -1] = 0
        else:
            temp.iloc[:, -1] = 1
        test_label_dict["cat" + str(i)] = temp

    train_data_x = list(torch.Tensor(train_dict[key].to_numpy()) for key in train_dict)
    train_data_y = list(
        torch.Tensor(train_label_dict[key].to_numpy().flatten()) for key in train_label_dict
    )
    test_data_x = list(torch.Tensor(test_dict[key].to_numpy()) for key in test_dict)
    test_data_y = list(
        torch.Tensor(test_label_dict[key].to_numpy().flatten()) for key in test_label_dict
    )

    with open("./data/train_data_x.pth", "wb") as f:
        pickle.dump(train_data_x, f)

    with open("./data/train_data_y.pth", "wb") as f:
        pickle.dump(train_data_y, f)

    with open("./data/test_data_x.pth", "wb") as f:
        pickle.dump(test_data_x, f)

    with open("./data/test_data_y.pth", "wb") as f:
        pickle.dump(test_data_y, f)


def task_ordering(perm):
    """Divides Data into tasks based on the given permutation order

    Parameters
    ----------
    perm : dict
        Dictionary containing task id and the classes present in it.

    Returns
    -------
    tuple
        Final dataset divided into tasks
    """
    final_train_data_x = []
    final_train_data_y = []
    final_test_data_x = []
    final_test_data_y = []

    for key, values in perm.items():
        temp_train_data_x = torch.Tensor([])
        temp_train_data_y = torch.Tensor([])
        temp_test_data_x = torch.Tensor([])
        temp_test_data_y = torch.Tensor([])

        for value in values:
            temp_train_data_x = torch.cat([temp_train_data_x, train_data_x[value]])
            temp_train_data_y = torch.cat([temp_train_data_y, train_data_y[value]])
            temp_test_data_x = torch.cat([temp_test_data_x, test_data_x[value]])
            temp_test_data_y = torch.cat([temp_test_data_y, test_data_y[value]])

        final_train_data_x.append(temp_train_data_x)
        final_train_data_y.append(temp_train_data_y)
        final_test_data_x.append(temp_test_data_x)
        final_test_data_y.append(temp_test_data_y)

    final_train_data_y = [x.long() for x in final_train_data_y]
    final_test_data_y = [x.long() for x in final_test_data_y]
    return final_train_data_x, final_train_data_y, final_test_data_x, final_test_data_y


class Conv(nn.Module):
    def __init__(self, num_classes=2, height=7, width=10, hidden_size=100):
        super().__init__()

        self.conv2d = nn.Conv2d(1, 14, kernel_size=3)

        self.feature1 = nn.Sequential(
            nn.Linear(14 * 5 * 8, hidden_size), nn.ReLU(inplace=True), nn.Dropout()
        )
        self.classifier = nn.Linear(hidden_size, num_classes)
        self._height = height
        self._width = width

    def forward(self, x):
        x = x.view(x.size(0), 1, self._height, self._width)
        # (128,7,10)
        x = self.conv2d(x)
        x = x.view(-1, 14 * 5 * 8)
        x = self.feature1(x)
        x = self.classifier(x)
        return x


# Reshaping the input
train_data_x = [x.view(x.shape[0], 7, 10) for x in train_data_x]
test_data_x = [x.view(x.shape[0], 7, 10) for x in test_data_x]

# Model Creation
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Conv()
model.to(device)


# Model architecture
arch = args.model_architecture.upper()

# Task Orders
perm1 = {
    "1": [0, 1, 2],
    "2": [3, 4, 5],
    "3": [6, 7, 8],
    "4": [9, 10, 11],
    "5": [12, 13, 14],
}
perm2 = {
    "1": [2, 4, 8],
    "2": [5, 9, 0],
    "3": [12, 1, 7],
    "4": [3, 14, 13],
    "5": [10, 11, 6],
}
perm3 = {
    "1": [14, 9, 12],
    "2": [4, 3, 5],
    "3": [0, 11, 8],
    "4": [7, 6, 10],
    "5": [2, 13, 1],
}
perm4 = {
    "1": [1, 7, 4],
    "2": [3, 12, 2],
    "3": [10, 6, 11],
    "4": [13, 8, 0],
    "5": [9, 14, 5],
}
perm5 = {
    "1": [10, 13, 14],
    "2": [3, 5, 6],
    "3": [9, 4, 2],
    "4": [1, 12, 8],
    "5": [7, 11, 0],
}

task_order_list = [perm1, perm2, perm3, perm4, perm5]


for task_order in range(len(task_order_list)):
    print("Current task order processing ", task_order + 1)
    dataset = task_ordering(task_order_list[task_order])

    generic_scenario = tensor_scenario(
        train_data_x=dataset[0],
        train_data_y=dataset[1],
        test_data_x=dataset[2],
        test_data_y=dataset[3],
        task_labels=[
            0 for key in task_order_list[task_order].keys()
        ],  # shouldn't provide task ID for inference
    )

    # log to Tensorboard
    tb_logger = TensorboardLogger(
        f"./tb_data/{cur_time}_CNN2D_ClassInc_0_in_task{task_order+1}/"
    )

    # log to text file
    text_logger = TextLogger(
        open(f"./logs/{cur_time}_CNN2D_ClassInc_0_in_task{task_order+1}.txt", "w+")
    )

    # print to stdout
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True, epoch_running=True),
        ExperienceForgetting(),
        cpu_usage_metrics(experience=True),
        StreamConfusionMatrix(num_classes=2, save_image=False),
        disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=[interactive_logger, text_logger, tb_logger],
    )

    # if arch == "GEM":
    #     cl_strategy = GEM(
    #         model,
    #         optimizer=Adam(model.parameters()),
    #         patterns_per_exp=4400,
    #         criterion=CrossEntropyLoss(),
    #         train_mb_size=128,
    #         train_epochs=50,
    #         eval_mb_size=128,
    #         evaluator=eval_plugin,
    #         device=device,
    #     )
    # else:
    #     cl_strategy = EWC(
    #         model,
    #         optimizer=Adam(model.parameters()),
    #         ewc_lambda=0.001,
    #         criterion=CrossEntropyLoss(),
    #         train_mb_size=128,
    #         train_epochs=50,
    #         eval_mb_size=128,
    #         evaluator=eval_plugin,
    #         device=device,
    #     )

    cl_strategy = CBRS(
            model,
            optimizer=Adam(model.parameters()),
            patterns_per_exp=4400,
            criterion=CrossEntropyLoss(),
            train_mb_size=128,  # b - batch size
            train_epochs=1,
            eval_mb_size=128,
            evaluator=eval_plugin,
            device=device,
        )

    # TRAINING LOOP
    print("Starting experiment...")

    os.makedirs(
        os.path.join("weights", f"CNN2D_ClassInc_0inTask{task_order+1}"), exist_ok=True
    )

    results = []

    for task_number, experience in enumerate(generic_scenario.train_stream):
        print("Start of experience: ", experience.current_experience)
        #print("Current Classes: ", experience.classes_in_this_experience)

        # # train returns a dictionary which contains all the metric values
        res = cl_strategy.train(experience)

        print("Training completed")
        torch.save(
            model.state_dict(),
            "./weights/CNN2D_ClassInc_0inTask{}/After_training_Task_{}".format(
                task_order + 1, task_number + 1
            ),
        )
        print("Model saved!")
        print("Computing accuracy on the whole test set")
        # test also returns a dictionary which contains all the metric values
        results.append(cl_strategy.eval(generic_scenario.test_stream))