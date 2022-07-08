import os
import time
import math
import numpy as np
import torch
import copy
import datasets, models

def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def killport(port):
    if is_port_in_use(port):
        print("Warning: port " + str(port) + "is in use")
        command = '''kill -9 $(netstat -nlp | grep :''' + str(
            port) + ''' | awk '{print $7}' | awk -F"/" '{ print $1 }')'''
        os.system(command)

def count_dataset(loader):
    counts = np.zeros(10)
    for _, target in loader.loader:
        labels = target.view(-1).numpy()
        for label in labels:
            counts[label] += 1
    print("class counts: ", counts)
    print("total data count: ", np.sum(counts))

def printer(content, fid):
    print(content)
    content = content.rstrip('\n') + '\n'
    fid.write(content)
    fid.flush()


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '{:>3}m {:2.0f}s'.format(m, s)

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def add_model(dst_model, src_model):
    """Add the parameters of two models.

        Args:
            dst_model (torch.nn.Module): the model to which the src_model will be added.
            src_model (torch.nn.Module): the model to be added to dst_model.
        Returns:
            torch.nn.Module: the resulting model of the addition.

        """

    params1 = src_model.named_parameters()
    params2 = dst_model.named_parameters()
    dict_params2 = dict(params2)
    with torch.no_grad():
        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].set_(
                    param1.data + dict_params2[name1].data)
    return dst_model


def scale_model(model, scale):
    """Scale the parameters of a model.

    Args:
        model (torch.nn.Module): the models whose parameters will be scaled.
        scale (float): the scaling factor.
    Returns:
        torch.nn.Module: the module with scaled parameters.

    """
    params = model.named_parameters()
    dict_params = dict(params)
    with torch.no_grad():
        for name, param in dict_params.items():
            dict_params[name].set_(dict_params[name].data / scale)
    return model


def CNN_prune_model(old_model, ratio):
    model_weight_index = list()
    model_bias_index = list()
    model_weight_vector = list()
    model_bias_vector = list()
    conv_r = [ratio, 0]
    fc_r = [ratio, 0]
    x = 0
    y = 0
    new_model = models.MNIST_Net_v1(ratio)
    new_dict_params = new_model.state_dict().copy()
    old_dict_params = old_model.state_dict().copy()
    index = np.arange(1)
    for name in new_dict_params:
        if 'features' in name:
            if 'weight' in name:
                weight_tensor, weight_index_, new_index, weight_vector_ = prune_conv_weight(old_dict_params[name], index, conv_r[x])
                new_dict_params[name] = weight_tensor
                model_weight_index.append(weight_index_)
                model_weight_vector.append(weight_vector_)
                index = new_index 
                x = x+1
            elif 'bias' in name:
                bias_tensor, bias_index_, bias_vector_ = prune_bias(old_dict_params[name], index)
                new_dict_params[name] = bias_tensor
                model_bias_index.append(bias_index_)
                model_bias_vector.append(bias_vector_)

    index = np.arange(1024)
    for name in new_dict_params:
        if 'classifier' in name:
            if 'weight' in name:
                weight_tensor, weight_index_, new_index, weight_vector_ = prune_fc_weight(old_dict_params[name], index, fc_r[y])
                new_dict_params[name] = weight_tensor
                model_weight_index.append(weight_index_)
                model_weight_vector.append(weight_vector_)
                index = new_index 
                y = y+1
            elif 'bias' in name:
                bias_tensor, bias_index_, bias_vector_ = prune_bias(old_dict_params[name], index)
                new_dict_params[name] = bias_tensor
                model_bias_index.append(bias_index_)
                model_bias_vector.append(bias_vector_)

    model = copy.deepcopy(new_model)
    model.load_state_dict(new_dict_params, strict=True)
    # model = model.to(self.device)
    return model, model_weight_index, model_bias_index, model_weight_vector, model_bias_vector


def CNN_restore_model(sub_model, weight_index, bias_index, weight_vector, bias_vector):
    total_consensus_distance = 0
    restored_model = models.MNIST_Net()
    sub_dict_params = sub_model.state_dict().copy()
    restored_dict_params = restored_model.state_dict().copy()
    axis_0 = [32, 64, 512, 10]
    axis_1 = [1, 32, 1024, 512]
    axis_2 = [5, 5]
    axis_3 = [5, 5]
    x = 0
    y = 0
    for name in restored_dict_params:
        if 'features' in name:
            if 'weight' in name:
                restored_dict_params[name], consensus_distance = restore_conv_weight(sub_dict_params[name], 
                                                                            weight_index[x],
                                                                            weight_vector[x], 
                                                                            axis_0[x], 
                                                                            axis_1[x],
                                                                            axis_2[x],
                                                                            axis_3[x])
                x = x+1
                total_consensus_distance += consensus_distance
            if 'bias' in name:
                restored_dict_params[name], consensus_distance = restore_bias(sub_dict_params[name], bias_index[y], bias_vector[y], axis_0[y])
                y = y+1
                total_consensus_distance += consensus_distance

    for name in restored_dict_params:
        if 'classifier' in name:
            if 'weight' in name:
                restored_dict_params[name], consensus_distance = restore_fc_weight(sub_dict_params[name], weight_index[x], weight_vector[x], axis_0[x], axis_1[x])
                x = x+1
                total_consensus_distance += consensus_distance
            if 'bias' in name:
                restored_dict_params[name], consensus_distance = restore_bias(sub_dict_params[name], bias_index[y], bias_vector[y], axis_0[y])
                y = y+1
                total_consensus_distance += consensus_distance

    model = copy.deepcopy(restored_model)
    model.load_state_dict(restored_dict_params, strict=True)
    # model = model.to(self.device)
    return model, np.power(total_consensus_distance, 0.5)


def prune_conv_weight(old_weight, col_index, row_ratio):
    old_weight_array = old_weight.cpu().numpy()
    old_flatten_weight_array = old_weight_array.flatten()
    #对每一行求和
    temp = np.sum(np.abs(old_weight_array), axis=3)
    temp = np.sum(np.abs(temp), axis=2)
    row = np.sum(np.abs(temp), axis=1)
    row_keep_num = int((1-row_ratio)*row.size)
    col_keep_num = col_index.size
    inx = np.argpartition(np.abs(row), -row_keep_num)[-row_keep_num:]
    iny = col_index
    weight_selected_indices = np.zeros((old_weight_array.shape[0],old_weight_array.shape[1],old_weight_array.shape[2],old_weight_array.shape[3])).astype(np.bool)
    for i in range(iny.size):
        weight_selected_indices[inx, iny[i]] = True
    weight_not_selected_indices = ~weight_selected_indices
    weight_array = copy.deepcopy(old_weight_array)
    weight_array[weight_not_selected_indices] = 65536

    flatten_weight_array = weight_array.flatten()
    #标记的索引
    weight_index = np.argwhere(np.abs(flatten_weight_array)<65536).flatten()

    new_weight_array = weight_array[np.abs(weight_array)<65536]
    new_weight_array.resize((row_keep_num, col_keep_num, old_weight_array.shape[2], old_weight_array.shape[3]))
    new_weight_tensor = torch.from_numpy(new_weight_array)
    return new_weight_tensor, weight_index, inx, old_flatten_weight_array

def prune_fc_weight(old_weight, col_index, row_ratio):
    old_weight_array = old_weight.cpu().numpy()
    old_flatten_weight_array = old_weight_array.flatten()
    #对每一行求和
    row = np.sum(np.abs(old_weight_array), axis=1)
    column = np.sum(np.abs(old_weight_array), axis=0)
    row_keep_num = int((1-row_ratio)*row.size)
    col_keep_num = col_index.size
    inx = np.argpartition(np.abs(row), -row_keep_num)[-row_keep_num:]
    iny = col_index
    weight_selected_indices = np.zeros((row.size,column.size)).astype(np.bool)
    for i in range(iny.size):
        weight_selected_indices[inx, iny[i]] = True
    weight_not_selected_indices = ~weight_selected_indices
    weight_array = copy.deepcopy(old_weight_array)
    weight_array[weight_not_selected_indices] = 65536

    flatten_weight_array = weight_array.flatten()
    #标记的索引
    weight_index = np.argwhere(np.abs(flatten_weight_array)<65536).flatten()

    new_weight_array = weight_array[np.abs(weight_array)<65536]
    new_weight_array.resize((row_keep_num, col_keep_num))
    new_weight_tensor = torch.from_numpy(new_weight_array)
    return new_weight_tensor, weight_index, inx, old_flatten_weight_array

def prune_bias(old_bias, bias_index):
    old_bias_array = old_bias.cpu().numpy()
    bias_selected_indices = np.zeros((old_bias_array.size,)).astype(np.bool)
    bias_selected_indices[bias_index] = True
    bias_not_selected_indices = ~bias_selected_indices
    bias_array = old_bias_array.copy()
    bias_array[bias_not_selected_indices] = 65536

    flatten_bias_array = bias_array.flatten()
    #标记的索引
    new_bias_index = np.argwhere(np.abs(flatten_bias_array)<65536).flatten()

    new_bias_array = bias_array[np.abs(bias_array)<65536]
    new_bias_tensor = torch.from_numpy(new_bias_array)
    return new_bias_tensor, new_bias_index, old_bias_array

def restore_conv_weight(sub_weight, w_index, w_vector, size_0, size_1, size_2, size_3):
    consensus_distance = 0
    sub_weight_array = sub_weight.cpu().numpy()
    flatten_sub_weight_array = sub_weight_array.flatten()
    restored_weight_array = w_vector
    # print("flatten_sub_weight_array.size:",flatten_sub_weight_array.size)
    # print("w_index.size:",w_index.size)
    for i in range(flatten_sub_weight_array.size):
        idx = w_index[i]
        consensus_distance += (restored_weight_array[idx] - flatten_sub_weight_array[i])**2
        restored_weight_array[idx] = flatten_sub_weight_array[i]
    consensus_distance = consensus_distance/flatten_sub_weight_array.size*restored_weight_array.size
    restored_weight_array.resize((size_0, size_1, size_2, size_3))
    restored_weight_array = restored_weight_array.astype(np.float32)
    restored_weight_tensor = torch.from_numpy(restored_weight_array)
    return restored_weight_tensor, consensus_distance

def restore_fc_weight(sub_weight, w_index, w_vector, row_size, col_size):
    consensus_distance = 0
    sub_weight_array = sub_weight.cpu().numpy()
    flatten_sub_weight_array = sub_weight_array.flatten()
    restored_weight_array = w_vector
    # print("flatten_sub_weight_array.size:",flatten_sub_weight_array.size)
    # print("w_index.size:",w_index.size)
    for i in range(flatten_sub_weight_array.size):
        idx = w_index[i]
        consensus_distance += (restored_weight_array[idx] - flatten_sub_weight_array[i])**2
        restored_weight_array[idx] = flatten_sub_weight_array[i]
    consensus_distance = consensus_distance/flatten_sub_weight_array.size*restored_weight_array.size
    restored_weight_array.resize((row_size, col_size))
    restored_weight_array = restored_weight_array.astype(np.float32)
    restored_weight_tensor = torch.from_numpy(restored_weight_array)
    return restored_weight_tensor, consensus_distance

def restore_bias(sub_bias, b_index, b_vector, bias_size):
    consensus_distance = 0
    sub_bias_array = sub_bias.cpu().numpy()
    restored_bias_array = b_vector
    for i in range(sub_bias_array.size):
        consensus_distance += (restored_bias_array[b_index[i]] - sub_bias_array[i])**2
        restored_bias_array[b_index[i]] = sub_bias_array[i]
    consensus_distance = consensus_distance/sub_bias_array.size*restored_bias_array.size
    restored_bias_array = restored_bias_array.astype(np.float32)
    restored_bias_tensor = torch.from_numpy(restored_bias_array)
    return restored_bias_tensor, consensus_distance
