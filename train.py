import argparse
import numpy as np
import time
import torch
import utils
import os
from model import RENet
from sklearn.utils import shuffle
import pickle


def train(args):
    # load data
    num_nodes, num_rels = utils.get_total_number('./data/' + args.dataset, 'stat.txt')
    train_data, train_times = utils.load_quadruples('./data/' + args.dataset, 'train.txt')
    valid_data, valid_times = utils.load_quadruples('./data/' + args.dataset, 'valid.txt')
    total_data, total_times = utils.load_quadruples('./data/' + args.dataset, 'train.txt', 'valid.txt','test.txt')

    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    seed = 999
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    os.makedirs('models', exist_ok=True)
    if args.model == 0:
        model_state_file = 'models/' + args.dataset + 'attn.pth'
    elif args.model == 1:
        model_state_file = 'models/' + args.dataset + 'mean.pth'
    elif args.model == 2:
        model_state_file = 'models/' + args.dataset + 'gcn.pth'
    elif args.model == 3:
        model_state_file = 'models/' + args.dataset + 'rgcn.pth'
        model_graph_file = 'models/' + args.dataset + 'rgcn_graph.pth'

    print("start training...")
    model = RENet(num_nodes,
                    args.n_hidden,
                    num_rels,
                    dropout=args.dropout,
                    model=args.model,
                    seq_len=args.seq_len,
                    num_k=args.num_k) 

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.00001)


    if use_cuda:
        model.cuda()
    if args.model == 3:
        train_sub = '/train_history_sub.txt'
        train_ob = '/train_history_ob.txt'
        valid_sub = '/dev_history_sub.txt'
        valid_ob = '/dev_history_ob.txt'
        with open('./data/' + args.dataset+'/train_graphs.txt', 'rb') as f:
            graph_dict = pickle.load(f)
        model.graph_dict = graph_dict
    else:
        train_sub = '/train_history_sub1.txt'
        train_ob = '/train_history_ob1.txt'
        valid_sub = '/dev_history_sub1.txt'
        valid_ob = '/dev_history_ob1.txt'
    with open('./data/' + args.dataset+train_sub, 'rb') as f:
        s_history_data = pickle.load(f)
    with open('./data/' + args.dataset+train_ob, 'rb') as f:
        o_history_data = pickle.load(f)

    with open('./data/' + args.dataset+valid_sub, 'rb') as f:
        s_history_valid_data = pickle.load(f)
    with open('./data/' + args.dataset+valid_ob, 'rb') as f:
        o_history_valid_data = pickle.load(f)
    valid_data = torch.from_numpy(valid_data)

    if args.model == 3:
        s_history = s_history_data[0]
        s_history_t = s_history_data[1]
        o_history = o_history_data[0]
        o_history_t = o_history_data[1]
        s_history_valid = s_history_valid_data[0]
        s_history_valid_t = s_history_valid_data[1]
        o_history_valid = o_history_valid_data[0]
        o_history_valid_t = o_history_valid_data[1]
    else:
        s_history = s_history_data
        o_history = o_history_data
        s_history_valid = s_history_valid_data
        o_history_valid = o_history_valid_data

    total_data = torch.from_numpy(total_data)
    if use_cuda:
        total_data = total_data.cuda()
    

    epoch = 0
    best_mrr = 0
    while True:
        model.train()
        if epoch == args.max_epochs:
            break
        epoch += 1
        loss_epoch = 0
        t0 = time.time()

        if args.model == 3:
            for batch_data, s_hist, s_hist_t, o_hist, o_hist_t in utils.make_batch2(train_data, s_history, s_history_t,
                                                                                   o_history, o_history_t, args.batch_size):

                batch_data = torch.from_numpy(batch_data)
                if use_cuda:
                    batch_data = batch_data.cuda()
                loss = model.get_loss(batch_data, (s_hist, s_hist_t), (o_hist, o_hist_t), graph_dict)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
                optimizer.step()
                optimizer.zero_grad()
                loss_epoch += loss.item()

        else:
            for batch_data, s_hist, o_hist in utils.make_batch(train_data, s_history, o_history, args.batch_size):
                batch_data = torch.from_numpy(batch_data)
                if use_cuda:
                    batch_data = batch_data.cuda()

                loss = model.get_loss(batch_data, s_hist, o_hist, None)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
                optimizer.step()
                optimizer.zero_grad()
                loss_epoch += loss.item()

        t3 = time.time()
        print("Epoch {:04d} | Loss {:.4f} | time {:.4f}".
              format(epoch, loss_epoch/(len(train_data)/args.batch_size), t3 - t0))

        if epoch % 1 == 0:
            model.eval()
            total_loss = 0
            total_ranks = np.array([])
            model.init_history()
            model.latest_time = valid_data[0][3]

            for i in range(len(valid_data)):
                batch_data = valid_data[i]
                s_hist = s_history_valid[i]
                o_hist = o_history_valid[i]
                if args.model == 3:
                    s_hist_t = s_history_valid_t[i]
                    o_hist_t = o_history_valid_t[i]

                if use_cuda:
                    batch_data = batch_data.cuda()

                with torch.no_grad():
                    if args.model == 3:
                        ranks, loss = model.evaluate_filter(batch_data, (s_hist, s_hist_t), (o_hist, o_hist_t), total_data)
                    else:
                        ranks, loss = model.evaluate_filter(batch_data, s_hist, o_hist, total_data)
                    total_ranks = np.concatenate((total_ranks, ranks))
                    total_loss += loss.item()

            mrr = np.mean(1.0 / total_ranks)
            mr = np.mean(total_ranks)
            hits = []
            for hit in [1, 3, 10]:
                avg_count = np.mean((total_ranks <= hit))
                hits.append(avg_count)
                print("valid Hits (filtered) @ {}: {:.6f}".format(hit, avg_count))
            print("valid MRR (filtered): {:.6f}".format(mrr))
            print("valid MR (filtered): {:.6f}".format(mr))
            print("valid Loss: {:.6f}".format(total_loss / (len(valid_data))))

            if mrr > best_mrr:
                best_mrr = mrr
                if args.model == 3:
                    torch.save({'state_dict': model.state_dict(), 'epoch': epoch,
                            's_hist': model.s_hist_test, 's_cache': model.s_his_cache,
                            'o_hist': model.o_hist_test, 'o_cache': model.o_his_cache,
                            's_hist_t': model.s_hist_test_t, 's_cache_t': model.s_his_cache_t,
                            'o_hist_t': model.o_hist_test_t, 'o_cache_t': model.o_his_cache_t,
                            'latest_time': model.latest_time},
                           model_state_file)
                    with open(model_graph_file, 'wb') as fp:
                        pickle.dump(model.graph_dict, fp)
                else:
                    torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 
                    's_hist': model.s_hist_test, 's_cache': model.s_his_cache, 
                    'o_hist': model.o_hist_test, 'o_cache': model.o_his_cache, 
                    'latest_time': model.latest_time},
                    model_state_file)

    print("training done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RENet')
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=200,
            help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=0,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-3,
            help="learning rate")
    parser.add_argument("-d", "--dataset", type=str, default='ICEWS18',
            help="dataset to use")
    parser.add_argument("--grad-norm", type=float, default=1.0,
    help="norm to clip gradient to")
    parser.add_argument("--max-epochs", type=int, default=20
                        ,
                        help="maximum epochs")
    parser.add_argument("--model", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--num-k", type=int, default=10,
                    help="cuttoff position")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--rnn-layers", type=int, default=1)

    args = parser.parse_args()
    print(args)
    train(args)

