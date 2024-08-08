import argparse
import numpy as np
import random
import os
import copy
import torch, gc
import torch.optim as optim

from model.FSTA import FSTA
from model.Optim import ScheduledOptim

# from utils import utils
from utils.utils import *
from utils.FourierAttUtils import EarlyStopping, check_path, set_seed, get_local_time, get_seq_dic, get_rating_matrix

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def prepare_dataloaders(opt, device):
    path = '../data/real/individual_'+opt.pos+'_mtl_reduced/'
    ground_truth = real_data_label(opt.pos)

    all_path = os.listdir(path)
    subjects = len(all_path)
    data = np.empty((subjects, 0, 0))
    i = 0
    for sub_path in all_path:
        position = path + '/' + sub_path
        data_tmp = np.loadtxt(position, skiprows=opt.skiprows, delimiter='\t')
        if i == 0:
            data = np.expand_dims(data_tmp, axis=0)
        else:
            data = np.concatenate((data, np.expand_dims(data_tmp, axis=0)), axis=0)
        i += 1
    print("data:", data.shape)

    # data:[S, T, N], S:number of subjects
    data = torch.FloatTensor(data).to(device)
    opt.nodes_num = data.shape[2]
    label = data
    opt.time_num = data.shape[1]  # T
    dataset = torch.utils.data.TensorDataset(data, label)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size)
    return data_loader, ground_truth

def train_epoch(model, data_loader, optimizer, criterion, opt, device, smoothing):
    ''' Epoch operation in training phase'''
    model.train()
    train_loss = []
    batch_adj = []
    for data_tmp, label_tmp in data_loader:
        optimizer.zero_grad()
        output, adj = model(data_tmp)
        loss = criterion(output, adj, label_tmp)  # output/label_tmp:[B, T, N]
        loss.backward()
        optimizer.step_and_update_lr()

        train_loss.append(loss.item())
        batch_adj.append(adj.cpu().detach().numpy())
    train_loss = np.average(train_loss)
    adj_mean = np.mean(batch_adj, axis=0) # [c, N, N]->[N, N]
    adj_mean = adj_mean.T

    return train_loss, adj_mean

def train(data_loader, ground_truth, device, opt):
    ''' Start training '''
    model = FSTA(opt=opt,
                   time_num=opt.time_num,
                   d_model=opt.d_model,
                   d_inner=opt.d_inner_hid,
                   n_head=opt.n_head,
                   d_k=opt.d_k,
                   d_v=opt.d_v,
                   dropout=opt.dropout).to(device)
    print(f"train params: d_model:{opt.d_model}, d_k/d_v:{opt.d_k}, n_head:{opt.n_head}, alpha_sp:{opt.alpha_sp}, soft_threshold:{opt.soft_threshold}, dropout:{opt.dropout}")
    optimizer = ScheduledOptim(
        optim.Adam([{'params': model.parameters()}], betas=(0.9, 0.98), eps=1e-09),
        opt.lr_mul, opt.d_model, opt.n_warmup_steps)

    criterion = loss_func(alpha_sp=opt.alpha_sp).to(device)

    for epoch_i in range(opt.epoch):
        train_loss, adj = train_epoch(
            model, data_loader, optimizer, criterion, opt, device, smoothing=opt.label_smoothing)
        adj_init = copy.deepcopy(adj)
        adj[np.arange(opt.nodes_num), np.arange(opt.nodes_num)] = 0
        opt.threshold = softThres(adj, opt.soft_threshold)
        adj_binary = change01(adj, threshold=opt.threshold)
        precision, recall, F1, accuracy, SHD = cal_metrics(adj_binary, ground_truth)
        if epoch_i % 100 == 0:
            print(f'epoch:{epoch_i}, loss:{train_loss: .3f}, precision:{precision}, recall:{recall}, F1:{F1}, accuracy:{accuracy}, SHD:{SHD}')
            print("threshold:", opt.threshold)
            print(adj_init)
            print(adj_binary)
        gc.collect()
        torch.cuda.empty_cache()
    return adj, precision, recall, F1, accuracy, SHD

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-pos', type=str, default='left', help='the index of fMRI dataset')
    parser.add_argument('-skiprows', type=int, default=1, help='in np.loadtxt, the num of skiprows')

    parser.add_argument('-epoch', type=int, default=301)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-d_model', type=int, default=16)
    parser.add_argument('-d_inner_hid', type=int, default=64)
    parser.add_argument('-d_k', type=int, default=8)
    parser.add_argument('-d_v', type=int, default=8)
    parser.add_argument('-n_head', type=int, default=2)

    parser.add_argument('-soft_threshold', type=float, default=0.5)
    parser.add_argument('-alpha_sp', type=float, default=0.8)
    parser.add_argument('-warmup','--n_warmup_steps', type=int, default=4000)
    parser.add_argument('-lr_mul', type=float, default=1.2)
    # parser.add_argument('-seed', type=int, default=2023)

    parser.add_argument('-dropout', type=float, default=0.2)
    parser.add_argument('-label_smoothing', action='store_true')

    # FourierAtt param
    parser.add_argument('-time_num', type=int, default=None)

    ### modify(add):
    parser.add_argument("--nodes_num", type=int, default=None)

    # model args
    parser.add_argument("--model_name", default="FMLPRec", type=str)
    parser.add_argument("--num_hidden_layers", default=1, type=int, help="number of filter-enhanced blocks")
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", default=0.5, type=float)
    parser.add_argument("--hidden_dropout_prob", default=0.5, type=float)
    parser.add_argument("--initializer_range", default=0.02, type=float)
    # parser.add_argument("--max_seq_length", default=50, type=int)
    parser.add_argument("--no_filters", action="store_true",
                        help="if no filters, filter layers transform to self-attention")

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", default=0.9, type=float, help="adam first beta value")
    parser.add_argument("--adam_beta2", default=0.999, type=float, help="adam second beta value")
    parser.add_argument("--gpu_id", default="0", type=str, help="gpu_id")
    parser.add_argument("--variance", default=5, type=float)

    opt = parser.parse_args()
    set_seed(opt.seed)
    opt.d_word_vec = opt.d_model

    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = False
        # torch.set_deterministic(True)
        np.random.seed(opt.seed)
        random.seed(opt.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #========= Loading Dataset =========#
    opt.d_model = 16
    opt.n_head = 2
    opt.d_k = 8
    opt.d_v = 8
    opt.alpha_sp = 0.8
    opt.soft_threshold = 0.5
    opt.dropout = 0.2
    opt.epoch = 301
    data_loader, ground_truth = prepare_dataloaders(opt, device)

    runs = 20
    metrics = []
    for i in range(1, runs + 1):
        print('runs:', i)
        adj, precision, recall, F1, accuracy, SHD = train(data_loader, ground_truth, device, opt)
        path = './out/real/' + opt.pos
        if not os.path.exists(path):
            os.makedirs(path)
        np.savetxt(path + "/" + str(i + 1) + ".txt", adj, fmt='%.04f', delimiter='\t')
        metrics.append([precision, recall, F1, accuracy, SHD])
    mu = np.mean(metrics, axis=0)
    std = np.std(metrics, axis=0)
    print(f'mu:{mu}, std:{std}')

if __name__ == '__main__':
    main()