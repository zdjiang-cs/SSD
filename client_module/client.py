import os
import time
import argparse
import asyncio
import concurrent.futures

import numpy as np
import torch
import torch.optim as optim
import copy
import torch.nn.functional as F

from config import ClientConfig
from client_comm_utils import *
from training_utils import train, test
import utils
import datasets, models

parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--idx', type=str, default="0",
                    help='index of worker')
parser.add_argument('--master_ip', type=str, default="master_ip",
                    help='IP address for controller or ps')
parser.add_argument('--master_port', type=int, default=58000, metavar='N',
                    help='')
parser.add_argument('--dataset_type', type=str, default='FashionMNIST')
parser.add_argument('--model_type', type=str, default='CNN')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--min_lr', type=float, default=0.001)
parser.add_argument('--ratio', type=float, default=1.0)
parser.add_argument('--step_size', type=float, default=1.0)
parser.add_argument('--decay_rate', type=float, default=0.98)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--use_cuda', action="store_false", default=True)
parser.add_argument('--visible_cuda', type=str, default='-1')
parser.add_argument('--algorithm', type=str, default='proposed')
parser.add_argument('--confidence_threshold', type=float, default=0.95)


args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")

def main():
    client_config = ClientConfig(
        idx=args.idx,
        master_ip=args.master_ip,
        master_port=args.master_port
    )
    # receive config
    master_socket = connect_send_socket(args.master_ip, args.master_port)
    config_received = get_data_socket(master_socket)
    for k, v in config_received.__dict__.items():
        setattr(client_config, k, v)

    for arg in vars(args):
        print(arg, ":", getattr(args, arg))

    # create model
    local_model = models.create_model_instance(args.dataset_type, args.model_type)
    # local_model.load_state_dict(client_config.para)
    torch.nn.utils.vector_to_parameters(client_config.para, local_model.parameters())
    local_model.to(device)
    para_nums = torch.nn.utils.parameters_to_vector(local_model.parameters()).nelement()

    # create dataset
    print("label data len : {}\n".format(len(client_config.custom["label_train_data_idxes"])))
    print("unlabel data len : {}\n".format(len(client_config.custom["unlabel_train_data_idxes"])))
    train_dataset, s_train_dataset, test_dataset = datasets.load_datasets(args.dataset_type)
    label_loader = datasets.create_dataloaders(train_dataset, batch_size=args.batch_size, selected_idxs=client_config.custom["label_train_data_idxes"])
    unlabel_loader = datasets.create_dataloaders(train_dataset, batch_size=args.batch_size, selected_idxs=client_config.custom["unlabel_train_data_idxes"], shuffle=False)
    s_unlabel_loader = datasets.create_dataloaders(s_train_dataset, batch_size=args.batch_size, selected_idxs=client_config.custom["unlabel_train_data_idxes"], shuffle=False)
    test_loader = datasets.create_dataloaders(test_dataset, batch_size=args.batch_size, selected_idxs=client_config.custom["test_data_idxes"], shuffle=False)
    
    print("label dataset:")
    utils.count_dataset(label_loader)
    print("unlabel dataset:")
    utils.count_dataset(unlabel_loader)
    print("s_unlabel dataset:")
    utils.count_dataset(s_unlabel_loader)
    print("test dataset:")
    utils.count_dataset(test_loader)
    
    # create p2p communication socket
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=20,)
    tasks = []
    for _, (neighbor_ip, send_port, listen_port, _) in client_config.custom["neighbor_info"].items():
        tasks.append(loop.run_in_executor(executor, connect_send_socket, neighbor_ip, send_port))
        tasks.append(loop.run_in_executor(executor, connect_get_socket, client_config.client_ip, listen_port))
    loop.run_until_complete(asyncio.wait(tasks))

    # save socket for later communication
    for task_idx, neighbor_idx in enumerate(client_config.custom["neighbor_info"].keys()):
        client_config.send_socket_dict[neighbor_idx] = tasks[task_idx*2].result()
        client_config.get_socket_dict[neighbor_idx] = tasks[task_idx*2+1].result()
    loop.close()

    epoch_lr = args.lr
    for epoch in range(1, 1+args.epoch):
        traffic = 0.0
        comm_time = 0.0
        comp_time = 0.0
        print("--**--")
        epoch_start_time = time.time()
        if epoch > 1 and epoch % 1 == 0:
            epoch_lr = max((args.decay_rate * epoch_lr, args.min_lr))
        print("epoch-{} lr: {}".format(epoch, epoch_lr))

        local_steps, comm_neighbors = get_data_socket(master_socket)
        start_time = time.time()

        #local train
        optimizer = optim.SGD(local_model.parameters(), lr=epoch_lr)
        train_loss = train(local_model, label_loader, optimizer, local_iters=local_steps, device=device, model_type=args.model_type)
        comp_time = time.time() - start_time
        print("train time: ", comp_time)
        test_loss, acc = test(local_model, test_loader, device, model_type=args.model_type)
        print("epoch: {}, label test loss: {}, test accuracy: {}".format(epoch, test_loss, acc))

        #communicate with neighbors
        rec_msg = dict()
        start_time = time.time()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=20,)
        tasks = []
        for neighbor_idx, _ in comm_neighbors:
            send_msg = [str(args.idx)+"_TO_"+str(neighbor_idx), local_model]
            print("neighbor:",neighbor_idx, "send_traffic:", len(pickle.dumps(send_msg))/(1024*1024))
            traffic += len(pickle.dumps(local_model))/(1024*1024)
            tasks.append(loop.run_in_executor(executor, send_data_socket, send_msg,
                                                client_config.send_socket_dict[neighbor_idx]))
            tasks.append(loop.run_in_executor(executor, get_msg, rec_msg, neighbor_idx,
                                                client_config.get_socket_dict[neighbor_idx]))
        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()
        comm_time = time.time() - start_time
        print("send and get time: ", comm_time)


        # aggregate models
        neighbor_num = 1
        model_norm_arr = [0] * 20
        neighbors_consensus_distance = dict()
        local_para = torch.nn.utils.parameters_to_vector(local_model.parameters()).clone().detach()
        for neighbor_idx, _ in comm_neighbors:
            remote_para = torch.nn.utils.parameters_to_vector(rec_msg[neighbor_idx][1].parameters()).clone().detach()
            model_norm_arr[neighbor_idx] = torch.norm(torch.sub(local_para, remote_para), 2)
            print(rec_msg[neighbor_idx][0])
            local_model = utils.add_model(local_model, rec_msg[neighbor_idx][1]) 
            neighbor_num += 1

        print("model_norm_arr: ", model_norm_arr)

        local_model = utils.scale_model(local_model, neighbor_num)   
        local_model.to(device) 

        local_para = torch.nn.utils.parameters_to_vector(local_model.parameters()).clone().detach()
        model_norm = torch.norm(local_para, 2)

        test_loss, acc = test(local_model, test_loader, device, model_type=args.model_type)
        print("epoch: {}, aggregate test loss: {}, test accuracy: {}".format(epoch, test_loss, acc))

        # produce pseudo labels
        time_1 = time.time()
        prediction_results, t_acc = get_results_val(unlabel_loader, local_model)
        print("confidence_threshold: ", args.confidence_threshold)
        avg_labels, selected_indices = select_unlabelled_data(prediction_results, args.confidence_threshold)
        time_2 = time.time()
        print("Pseudo labeling time ", time_2 - time_1)

        optimizer = optim.SGD(local_model.parameters(), lr=epoch_lr)
        train_on_pseduo(local_model, s_train_dataset, np.array(client_config.custom["unlabel_train_data_idxes"]), selected_indices, avg_labels, optimizer, args.batch_size)
 

        print("***")
        
        start_time = time.time()
        test_loss, acc = test(local_model, test_loader, device, model_type=args.model_type)
        send_data_socket((epoch, time.time() - epoch_start_time, comp_time, comm_time, traffic, acc, test_loss, train_loss, model_norm, neighbors_consensus_distance, model_norm_arr), master_socket)
        print("test time: ", time.time() - start_time)
        print("epoch: {}, train loss: {}, test loss: {}, test accuracy: {}".format(epoch, train_loss, test_loss, acc))
        print("\n\n")


    # close socket
    for _, conn in client_config.send_socket_dict.items():
        conn.shutdown(2)
        conn.close()
    for _, conn in client_config.get_socket_dict.items():
        conn.shutdown(2)
        conn.close()
    master_socket.shutdown(2)
    master_socket.close()

def aggregate_model(local_para, comm_neighbors, client_config, step_size):
    with torch.no_grad():
        para_delta = torch.zeros_like(local_para)
        for neighbor_idx, average_weight in comm_neighbors:
            print("idx: {}, weight: {}".format(neighbor_idx, average_weight))
            indice = client_config.neighbor_indices[neighbor_idx]
            selected_indicator = torch.zeros_like(local_para)
            selected_indicator[indice] = 1.0
            model_delta = (client_config.neighbor_paras[neighbor_idx] - local_para) * selected_indicator
            para_delta += step_size * average_weight * model_delta

            client_config.estimated_consensus_distance[neighbor_idx] = np.power(np.power(torch.norm(model_delta).item(), 2) / len(indice) * model_delta.nelement(), 0.5)
        local_para += para_delta

    return local_para

def get_msg(rec_msg, idx, connection):
    rec_msg[idx] = get_data_socket(connection)

def get_results_val(data_loader, label_model, device=torch.device("cuda")):
    setResults = list()
    label_model.eval()
    data_loader = data_loader.loader

    correct = 0
    with torch.no_grad():
        for _, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            output = label_model(data)
            softmax1 = F.softmax(output, dim=1).cpu().detach().numpy()
            avg_pred = softmax1.copy()
            setResults.extend(softmax1.copy())

            pred = avg_pred.argmax(1)
            correct = correct + np.sum(pred == target.cpu().detach().numpy().reshape(pred.shape))

    t_acc = correct/len(data_loader.dataset)
    print("teachers' acc in val: {}".format(t_acc))
    return np.array(setResults), t_acc

def select_unlabelled_data(prediction_results, p_th):
    avg_labels = prediction_results
    max_label = np.argmax(avg_labels, axis=1)

    selected_indices = list()
    for data_idx in range(len(prediction_results)):
        if avg_labels[data_idx][max_label[data_idx]] >= p_th:
            selected_indices.append(data_idx)

    print("num of selected samples: ", len(selected_indices))
    return np.array(avg_labels)[selected_indices], np.array(selected_indices)

def count_dataset(loader, soft_labels, tf_batch_size):
    counts = np.zeros(len(loader.loader.dataset.classes))
    right = np.zeros(len(loader.loader.dataset.classes))

    st_matrix = np.zeros((len(loader.loader.dataset.classes), len(loader.loader.dataset.classes)))
    for data_idx, (_, target) in enumerate(loader.loader):
        predu = torch.from_numpy(soft_labels[data_idx*tf_batch_size:(data_idx+1)*tf_batch_size])
        predu = predu.argmax(1)
        batch_correct = predu.eq(target.view_as(predu))

        labels = target.view(-1).numpy()
        predu = predu.view(-1).numpy()
        for label_idx, label in enumerate(labels):
            counts[label] += 1
            st_matrix[label][predu[label_idx]] += 1
            if batch_correct[label_idx] == True:
                right[label] += 1
    print(st_matrix.astype(np.int))
    print("class counts: ", counts.astype(np.int))
    print("total data count: ", np.sum(counts))
    print("right class counts: ", right.astype(np.int))
    print("total right data count: ", np.sum(right))

    return np.sum(counts), np.sum(right)

def train_on_pseduo(model, train_dataset, unlabelled_indices, selected_indices, soft_labels, optimizer, tf_batch_size, device=torch.device("cuda")):
    if len(selected_indices) <= 0:
        return
    model.train()
    model = model.to(device)
    selected_shuffle = [i for i in range(len(selected_indices))]
    np.random.shuffle(selected_shuffle)
    selected_indices = selected_indices[selected_shuffle]
    soft_labels = soft_labels[selected_shuffle]
    
    train_loader = datasets.create_dataloaders(train_dataset, batch_size=tf_batch_size, selected_idxs=unlabelled_indices[selected_indices], shuffle=False)
    total_num, right_num = count_dataset(train_loader, soft_labels, tf_batch_size)

    data_loader = train_loader.loader
    samples_num = 0

    train_loss = 0.0
    correct = 0

    correct_s = 0
    
    for data_idx, (data, label) in enumerate(data_loader):

        target = torch.from_numpy(soft_labels[data_idx*tf_batch_size:(data_idx+1)*tf_batch_size])
        data, target, label = data.to(device), target.to(device), label.to(device)
        
        output = model(data)
        # print(len(data), len(target), len(label), len(output))
        optimizer.zero_grad()

        loss = F.cross_entropy(output, target.argmax(1), reduction='mean')
        # loss = F.cross_entropy(output, label)

        loss.backward()
        optimizer.step()

        train_loss += (loss.item() * data.size(0))
        samples_num += data.size(0)

        pred = target.argmax(1, keepdim=True)
        batch_correct = pred.eq(label.view_as(pred)).sum().item()
        correct += batch_correct

        pred_s = output.argmax(1, keepdim=True)
        batch_correct_s = pred.eq(pred_s).sum().item()
        correct_s += batch_correct_s

    if samples_num != 0:
        train_loss /= samples_num
        # print("sample num: ", samples_num)
        test_accuracy = np.float(1.0 * correct / samples_num)
        print("teacher's acc : {}".format(test_accuracy))
        test_accuracy = np.float(1.0 * correct_s / samples_num)
        print("student's training acc : {}".format(test_accuracy))
    
    return train_loss, total_num, right_num, test_accuracy

if __name__ == '__main__':
    main()
