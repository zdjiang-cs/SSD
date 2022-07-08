import os
import sys
import time
import argparse
import socket
import pickle
import asyncio
import concurrent.futures
import json
import random
from pulp import *
import math
import numpy as np
import torch

from training_module.config import *
from communication_module.comm_utils import *
from training_module import datasets, models, utils

parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--model_type', type=str, default='CNN')
parser.add_argument('--dataset_type', type=str, default='FashionMNIST')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--data_pattern', type=float, default=0)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--algorithm', type=str, default='proposed')
parser.add_argument('--mode', type=str, default='adaptive')
parser.add_argument('--topology', type=str, default='ring')
parser.add_argument('--ratio', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--step_size', type=float, default=1.0)
parser.add_argument('--decay_rate', type=float, default=0.98)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--local_updates', type=int, default=50)
parser.add_argument('--time_budget', type=float, default=50000)
parser.add_argument('--confidence_threshold', type=float, default=0.95)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

SERVER_IP = "YOUR_SERVER_IP"

def main():
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    # init config
    common_config = CommonConfig()
    common_config.master_listen_port_base += random.randint(0, 6000)
    common_config.model_type = args.model_type
    common_config.dataset_type = args.dataset_type
    common_config.batch_size = args.batch_size
    common_config.ratio = args.ratio
    common_config.epoch = args.epoch
    common_config.lr = args.lr
    common_config.decay_rate = args.decay_rate
    common_config.weight_decay = args.weight_decay
    common_config.confidence_threshold = args.confidence_threshold

    # init p2p topology
    worker_num = 20
    adjacency_matrix = np.ones((worker_num, worker_num), dtype=np.int)

    for worker_idx in range(worker_num):
        adjacency_matrix[worker_idx][worker_idx] = 0

    p2p_port = np.zeros_like(adjacency_matrix)
    curr_port = common_config.p2p_listen_port_base + random.randint(0, 5000)
    for idx_row in range(len(adjacency_matrix)):
        for idx_col in range(len(adjacency_matrix[0])):
            if adjacency_matrix[idx_row][idx_col] != 0:
                curr_port += 2
                p2p_port[idx_row][idx_col] = curr_port

    # create workers
    for worker_idx in range(worker_num):
        custom = dict()
        custom["neighbor_info"] = dict()
        common_config.worker_list.append(
            Worker(config=ClientConfig(idx=worker_idx,
                                       client_ip=CLIENT_IP[worker_idx],
                                       master_ip=SERVER_IP,
                                       master_port=common_config.master_listen_port_base+worker_idx,
                                       custom=custom),
                   common_config=common_config, 
                   location='remote'
                   )
        )
    
    # init workers' config
    for worker_idx in range(worker_num):
        for neighbor_idx, link in enumerate(adjacency_matrix[worker_idx]):
            if link == 1:
                neighbor_config = common_config.worker_list[neighbor_idx].config
                neighbor_ip = neighbor_config.client_ip

                # neighbor ip, send_port, listen_port
                common_config.worker_list[worker_idx].config.custom["neighbor_info"][neighbor_idx] = \
                        (neighbor_ip, p2p_port[worker_idx][neighbor_idx], p2p_port[neighbor_idx][worker_idx])

    train_dataset, test_dataset = datasets.load_datasets(common_config.dataset_type)
    label_loader = datasets.create_dataloaders(train_dataset, batch_size=common_config.batch_size)
    
    print("label dataset:")
    utils.count_dataset(label_loader)

    # Create model instance
    global_model = models.create_model_instance(common_config.dataset_type, common_config.model_type)

    init_para = torch.nn.utils.parameters_to_vector(global_model.parameters())
    model_size = init_para.nelement() * 4 / 1024 / 1024
    print("Model Size: {} MB".format(model_size))

    # partition dataset
    data_ratio = [0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.015, 0.015, 0.015, 0.015]
    train_data_partition, test_data_partition = partition_data(common_config.dataset_type, data_ratio, args.data_pattern, worker_num)

    for worker_idx, worker in enumerate(common_config.worker_list):
        worker.config.para = init_para
        worker.config.custom["label_train_data_idxes"] = train_data_partition.use(worker_idx)
        worker.config.custom["unlabel_train_data_idxes"] = train_data_partition.use(worker_idx+worker_num)
        worker.config.custom["test_data_idxes"] = test_data_partition.use(worker_idx)

    # connect socket and send init config
    communication_parallel(common_config.worker_list, action="init")
    
    training_recorder = TrainingRecorder(common_config.worker_list, common_config.recorder)
    consensus_distance = np.ones((worker_num, worker_num)) * 1e-6

    if args.local_updates > 0:
        local_steps = args.local_updates
    else:
        local_steps = int(np.ceil(5000 / 32))
    print("local steps: {}".format(local_steps))

    avg_train_loss = 0.0
    total_traffic = 0
    total_time = 0
    total_mean_comp_time = 0
    total_mean_comm_time = 0


    acc_arr = [0.1]*worker_num
    model_norm_dis_arr = list()
    model_norm_arr = list()

    k = np.ones((worker_num, worker_num))
    topology = [[0 for i in range(worker_num)] for j in range(worker_num)]
    d = [[0.0001 for i in range(worker_num)] for j in range(worker_num)]

    for epoch_num in range(1, 1+common_config.epoch):
        print("\n--**--\nEpoch: {}".format(epoch_num))
        
        topology = fully_topo(worker_num)
        total_transfer_size = 0
        for worker in common_config.worker_list:
            worker_data_size = np.sum(topology[worker.idx]) * model_size
            total_transfer_size += worker_data_size
            neighbors_list = list()
            for neighbor_idx, link in enumerate(topology[worker.idx]):
                if link == 1:
                    neighbors_list.append((neighbor_idx, 1.0 / (np.max([np.sum(topology[worker.idx]), np.sum(topology[neighbor_idx])])+1)))
            send_data_socket((local_steps, neighbors_list), worker.socket)

        communication_parallel(common_config.worker_list, action="get")
        avg_acc = 0.0
        avg_test_loss = 0.0
        avg_train_loss = 0.0
        epoch_time = list()
        epoch_comp_time = list()
        epoch_comm_time = list()
        consensus_distance = list()
        for worker in common_config.worker_list:
            _, worker_time, comp_time, comm_time, traffic, acc, test_loss, train_loss, model_norm, neighbors_consensus_distance, model_norm_dis = worker.train_info[-1]
            common_config.recorder.add_scalar('Accuracy/worker_' + str(worker.idx), acc, epoch_num)
            common_config.recorder.add_scalar('Test_loss/worker_' + str(worker.idx), test_loss, epoch_num)
            common_config.recorder.add_scalar('Time/worker_' + str(worker.idx), worker_time, epoch_num)
            epoch_time.append(worker_time)
            epoch_comp_time.append(comp_time)
            epoch_comm_time.append(comm_time)

            total_traffic += traffic
            avg_acc += acc
            avg_test_loss += test_loss
            avg_train_loss += train_loss
            acc_arr.append(acc)
            model_norm_dis_arr.append(model_norm_dis)
            model_norm_arr.append(model_norm)
        
        avg_acc /= worker_num
        avg_test_loss /= worker_num
        avg_train_loss /= worker_num

        total_mean_comp_time += np.mean(epoch_comp_time)
        total_mean_comm_time += np.mean(epoch_comm_time)
        total_time = max(epoch_time)

        common_config.recorder.add_scalar('Accuracy/time', avg_acc, total_time)
        common_config.recorder.add_scalar('Accuracy/epoch', avg_acc, epoch_num)
        common_config.recorder.add_scalar('Test_loss/time', avg_test_loss, total_time)
        common_config.recorder.add_scalar('Test_loss/epoch', avg_test_loss, epoch_num)
        common_config.recorder.add_scalar('Train_loss/time', avg_train_loss, total_time)
        common_config.recorder.add_scalar('Train_loss/epoch', avg_train_loss, epoch_num)
        common_config.recorder.add_scalar('Time/epoch', total_time, epoch_num)
        common_config.recorder.add_scalar('Traffic/epoch', total_traffic, epoch_num)

    # close socket
    for worker in common_config.worker_list:
        worker.socket.shutdown(2)
        worker.socket.close()

class TrainingRecorder(object):
    def __init__(self, worker_list, recorder, beta=0.95):
        self.worker_list = worker_list
        self.worker_num = len(worker_list)
        self.beta = beta
        self.moving_consensus_distance = np.ones((self.worker_num, self.worker_num)) * 1e-6
        self.avg_update_norm = 0
        self.round = 0
        self.epoch = 0
        self.recorder = recorder
        self.total_time = 0

        for i in range(self.worker_num):
            self.moving_consensus_distance[i][i] = 0

    def get_train_info(self):
        self.round += 1
        communication_parallel(self.worker_list, action="get")
        avg_train_loss = 0.0
        round_consensus_distance = np.ones_like(self.moving_consensus_distance) * -1
        round_update_norm = np.zeros(self.worker_num)
        for worker in self.worker_list:
            train_loss, neighbors_consensus_distance, local_update_norm = worker.train_info[-1]
            for neighbor_idx in neighbors_consensus_distance.keys():
                round_consensus_distance[worker.idx][neighbor_idx] = neighbors_consensus_distance[neighbor_idx]
            round_update_norm[worker.idx] = local_update_norm
            avg_train_loss += train_loss
        if self.round == 1:
            self.avg_update_norm = np.average(round_update_norm)
        else:
            self.avg_update_norm = self.beta * self.avg_update_norm + (1 - self.beta) * np.average(round_update_norm)

        for worker_idx in range(self.worker_num):
            for neighbor_idx in range(worker_idx+1, self.worker_num):
                round_consensus_distance[worker_idx][neighbor_idx] = (round_consensus_distance[worker_idx][neighbor_idx]
                                                                + round_consensus_distance[neighbor_idx][worker_idx]) / 2
                round_consensus_distance[neighbor_idx][worker_idx] = round_consensus_distance[worker_idx][neighbor_idx]
        
        backup_distance = round_consensus_distance.copy()
        for k in range(self.worker_num):
            for worker_idx in range(self.worker_num):
                for neighbor_idx in range(worker_idx+1, self.worker_num):
                    if round_consensus_distance[worker_idx][k] >= 0 and round_consensus_distance[neighbor_idx][k] >=0:
                        tmp = round_consensus_distance[worker_idx][k] + round_consensus_distance[neighbor_idx][k]
                        if round_consensus_distance[worker_idx][neighbor_idx] < 0:
                            round_consensus_distance[worker_idx][neighbor_idx] = tmp
                        else:    
                            round_consensus_distance[worker_idx][neighbor_idx] = np.min([round_consensus_distance[worker_idx][neighbor_idx], tmp])
                        round_consensus_distance[neighbor_idx][worker_idx] = round_consensus_distance[worker_idx][neighbor_idx]
        
        for worker_idx in range(self.worker_num):
            round_consensus_distance[worker_idx][worker_idx] = 0

        if self.round == 1:
            self.moving_consensus_distance = round_consensus_distance
        else:
            self.moving_consensus_distance = self.beta * self.moving_consensus_distance + (1 - self.beta) * round_consensus_distance

        for worker_idx in range(self.worker_num):
            for neighbor_idx in range(worker_idx+1, self.worker_num):
                if backup_distance[worker_idx][neighbor_idx] >= 0:
                    self.moving_consensus_distance[worker_idx][neighbor_idx] = backup_distance[worker_idx][neighbor_idx]
                    self.moving_consensus_distance[neighbor_idx][worker_idx] = backup_distance[worker_idx][neighbor_idx]


        avg_train_loss = avg_train_loss / self.worker_num
        self.recorder.add_scalar('Train_loss/train', avg_train_loss, self.round)
        self.recorder.add_scalar('Distance', np.average(self.moving_consensus_distance), self.round)

        print("communication round {}, train loss: {}".format(self.round, avg_train_loss))

        return self.moving_consensus_distance, self.avg_update_norm

    def get_test_info(self):
        self.epoch += 1
        communication_parallel(self.worker_list, action="get")
        avg_acc = 0.0
        avg_test_loss = 0.0
        epoch_time = 0.0
        for worker in self.worker_list:
            _, worker_time, acc, loss, train_loss = worker.train_info[-1]
            self.recorder.add_scalar('Accuracy/worker_' + str(worker.idx), acc, self.round)
            self.recorder.add_scalar('Test_loss/worker_' + str(worker.idx), loss, self.round)
            self.recorder.add_scalar('Time/worker_' + str(worker.idx), worker_time, self.epoch)

            avg_acc += acc
            avg_test_loss += loss
            epoch_time = max(epoch_time, worker_time)
        
        avg_acc /= self.worker_num
        avg_test_loss /= self.worker_num
        self.total_time += epoch_time
        self.recorder.add_scalar('Time/total', epoch_time, self.epoch)
        self.recorder.add_scalar('Accuracy/average', avg_acc, self.epoch)
        self.recorder.add_scalar('Test_loss/average', avg_test_loss, self.epoch)
        self.recorder.add_scalar('Accuracy/round_average', avg_acc, self.round)
        self.recorder.add_scalar('Test_loss/round_average', avg_test_loss, self.round)
        print("Epoch: {}, time: {}, average accuracy: {}, average test loss: {}, average train loss: {}".format(self.epoch, self.total_time, avg_acc, avg_test_loss, train_loss))


def fully_topo(worker_num):
    topology = np.ones((worker_num, worker_num), dtype=np.int)

    for worker_idx in range(worker_num):
        topology[worker_idx][worker_idx] = 0
    
    return topology

def communication_parallel(worker_list, action, data=None):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(worker_list),)
        tasks = []
        for worker in worker_list:
            if action == "init":
                tasks.append(loop.run_in_executor(executor, worker.send_init_config))
            elif action == "get":
                tasks.append(loop.run_in_executor(executor, worker.get_config))
            elif action == "send":
                tasks.append(loop.run_in_executor(executor, worker.send_data, data))
        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()
    except:
        sys.exit(0)

def non_iid_partition(ratio, worker_num=10):
    partition_sizes = np.ones((10, worker_num)) * ((1 - ratio) / (worker_num-1))

    for worker_idx in range(worker_num):
        partition_sizes[worker_idx][worker_idx] = ratio

    return partition_sizes

def partition_data(dataset_type, data_ratio, data_pattern, worker_num):
    train_dataset, test_dataset = datasets.load_datasets(dataset_type)

    if dataset_type == "FashionMNIST":
        test_partition_sizes = np.ones((10, worker_num)) * (1 / worker_num)
        partition_sizes = np.ones((10, worker_num*2))
        if data_pattern == 0:
            for i in range(10):
                for j in range(worker_num):
                    partition_sizes[i][j] = data_ratio[j]
                    partition_sizes[i][j+worker_num] = 0.05-data_ratio[j]

    print("partition_sizes:\n", partition_sizes)

    train_data_partition = datasets.LabelwisePartitioner(train_dataset, partition_sizes=partition_sizes)
    test_data_partition = datasets.LabelwisePartitioner(test_dataset, partition_sizes=test_partition_sizes)
    
    return train_data_partition, test_data_partition


def cal_param(model_norm_dis_arr, model_norm_arr, acc_arr, k, t, node_num, d, topology):
    
    neighbor_score = np.zeros((node_num, node_num))
    alpha = np.zeros((node_num, node_num))
    beta = np.zeros((node_num, node_num))

    for i in range(node_num):
        for j in range(node_num):
            if t == 1 :
                temp_d = d[i][j]
            elif topology[i][j] == 1:
                temp_d = model_norm_dis_arr[i][j] / model_norm_arr[i]
            else:
                temp_d = d[i][j]
            
            alpha[i][j] = acc_arr[j] / 0.1
            beta[i][j] = math.sqrt( math.log(t) / k[i][j] )
            neighbor_score[i][j] = alpha[i][j] + beta[i][j]
            d[i][j] = temp_d
            
            if i==j:
                neighbor_score[i][j] = 0
    
    print("alpha: \n", alpha)
    print("beta: \n", beta)
    print("neighbor_score: \n", neighbor_score)
    return neighbor_score

def select_algorithm(neighbor_score, resource_limit, node_num, k):

    prob = LpProblem('MaximizeScore', LpMaximize)
    maximize_object = 0
    rates = list()
    index = np.zeros((node_num,node_num))
    for i in range(node_num):
        for j in range(node_num):
            index[i][j] = i*node_num+j
            if j>=i:
                rates.append(LpVariable("prob_"+str(i)+"_"+str(j), 0, 1, cat='Continuous'))
            else:
                rates.append(rates[int(index[j][i])])
            maximize_object += neighbor_score[i][j]*rates[int(index[i][j])]

    prob += maximize_object

    for i in range(node_num):
        resource_sum = 0
        for j in range(node_num):
            resource_sum += rates[int(index[i][j])]
        prob += resource_sum <= resource_limit[i]
 
    prob.solve(PULP_CBC_CMD(msg=0))

    topology = np.zeros((node_num, node_num), dtype=int)
    for i in range(node_num):
        for j in range(node_num):
            if value(rates[int(index[i][j])]) > 0:
                topology[i][j] = 1
                k[i][j] += 1

    return topology
     

if __name__ == "__main__":
    main()
