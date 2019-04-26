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
        model_state_file = 'models/'+args.dataset+'attn.pth'
    elif args.model == 1:
        model_state_file = 'models/'+args.dataset+'mean.pth'
    elif args.model == 2:
        model_state_file = 'models/'+args.dataset+'gcn.pth'

    print("start training...")
    model = RENet(num_nodes,
                    args.n_hidden,
                    num_rels,
                    dropout=args.dropout,
                    model=args.model,
                    seq_len=args.seq_len) 

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.00001)


    if use_cuda:
        model.cuda()
    with open('./data/' + args.dataset+'/train_history_sub.txt', 'rb') as f:
        s_history = pickle.load(f)
    with open('./data/' + args.dataset+'/train_history_ob.txt', 'rb') as f:
        o_history = pickle.load(f)

    with open('./data/' + args.dataset+'/dev_history_sub.txt', 'rb') as f:
        s_history_valid = pickle.load(f)
    with open('./data/' + args.dataset+'/dev_history_ob.txt', 'rb') as f:
        o_history_valid = pickle.load(f)
    valid_data = torch.from_numpy(valid_data)


    epoch = 0
    best_mrr = 0
    while True:
        model.train()
        if epoch == args.max_epochs:
            break
        epoch += 1
        loss_epoch = 0
        t0 = time.time()

        train_data, s_history, o_history = shuffle(train_data, s_history, o_history)
        i = 0
        for batch_data, s_hist, o_hist in utils.make_batch(train_data, s_history, o_history, args.batch_size):
            batch_data = torch.from_numpy(batch_data)
            if use_cuda:
                batch_data = batch_data.cuda()

            loss = model.get_loss(batch_data, s_hist, o_hist)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
            optimizer.step()
            optimizer.zero_grad()
            loss_epoch += loss.item()
            i += 1


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

                if use_cuda:
                    batch_data = batch_data.cuda()

                with torch.no_grad():
                    ranks, loss = model.evaluate(batch_data, s_hist, o_hist)
                    total_ranks = np.concatenate((total_ranks, ranks))
                    total_loss += loss.item()

            mrr = np.mean(1.0 / total_ranks)
            mr = np.mean(total_ranks)
            hits = []
            for hit in [1, 3, 10]:
                avg_count = np.mean((total_ranks <= hit))
                hits.append(avg_count)
                print("valid Hits (raw) @ {}: {:.6f}".format(hit, avg_count))
            print("valid MRR (raw): {:.6f}".format(mrr))
            print("valid MR (raw): {:.6f}".format(mr))
            print("valid Loss: {:.6f}".format(total_loss / (len(valid_data))))

            if mrr > best_mrr:
                best_mrr = mrr
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
    parser.add_argument("--gpu", type=int, default=-1,
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
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--rnn-layers", type=int, default=1)

    args = parser.parse_args()
    print(args)
    train(args)

