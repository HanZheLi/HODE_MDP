# coding=utf-8
"""
@author: Yantong Lai
@paper: [24 SIGIR] Disentangled Contrastive Hypergraph Learning for Next POI Recommendation
"""

import argparse
import time
import os
import logging
import yaml
import datetime
import torch.optim as optim
import random

# from dataset import *
from dataset_ours import *
from model_ours import *
from metrics import batch_performance
from utils import *


# clear cache
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


# parse argument
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="NYC", help='NYC/TKY/Gowalla')
parser.add_argument('--seed', default=42, help='Random seed')
parser.add_argument('--distance_threshold', default=2.5, type=float, help='distance threshold 2.5 or 0.25')
parser.add_argument('--num_epochs', type=int, default=20, help='number of epochs') # 对于NYC要设置20  对于TKY设置20 对于SH设置7或20 20的效果更好一点
parser.add_argument('--batch_size', type=int, default=1000, help='input batch size') # NYC和TKY都是200
parser.add_argument('--emb_dim', type=int, default=128, help='embedding size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--decay', type=float, default=5e-4)    # 5e-4
parser.add_argument('--dropout', type=float, default=0.2, help='dropout')    # 0.3
parser.add_argument('--deviceID', type=int, default=0)
parser.add_argument('--lambda_cl', type=float, default=0.01, help='lambda of contrastive loss') # NYC:0.4   TKY: 0.3   SH: 0.01
parser.add_argument('--num_mv_layers', type=int, default=2)
parser.add_argument('--num_geo_layers', type=int, default=2)
parser.add_argument('--num_di_layers', type=int, default=2, help='layer number of directed hypergraph convolutional network')
parser.add_argument('--temperature', type=float, default=0.1)
parser.add_argument('--keep_rate', type=float, default=1, help='ratio of edges to keep')
parser.add_argument('--keep_rate_poi', type=float, default=1, help='ratio of poi-poi directed edges to keep')  # 0.7
parser.add_argument('--lr-scheduler-factor', type=float, default=0.1, help='Learning rate scheduler factor')
parser.add_argument('--save_dir', type=str, default="logs")
parser.add_argument('--t1', type=int, default="t for u2p")
parser.add_argument('--t2', type=int, default="t for geo")
parser.add_argument('--t3', type=int, default="t for p2p")
args = parser.parse_args()

# set random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# set device gpu/cpu
device = torch.device("cuda:{}".format(args.deviceID) if torch.cuda.is_available() else "cpu")

# set save_dir
current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)
current_save_dir = os.path.join(args.save_dir, current_time)

# create current save_dir
os.mkdir(current_save_dir)

# Setup logger
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=os.path.join(current_save_dir, f"log_training.txt"),
                    filemode='w+')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
logging.getLogger('matplotlib.font_manager').disabled = True

# Save run settings
args_filename = args.dataset + '_args.yaml'
with open(os.path.join(current_save_dir, args_filename), 'w') as f:
    yaml.dump(vars(args), f, sort_keys=False)


def main():
    # Parse Arguments
    logging.info("1. Parse Arguments")
    logging.info(args)
    logging.info("device: {}".format(device))
    if args.dataset == "TKY":
        NUM_USERS = 2173
        NUM_POIS = 7038
        PADDING_IDX = NUM_POIS
    elif args.dataset == "NYC":
        NUM_USERS = 834
        NUM_POIS = 3835
        PADDING_IDX = NUM_POIS
    elif args.dataset == "SH":
        NUM_USERS = 10251
        NUM_POIS = 11535
        PADDING_IDX = NUM_POIS

    # Load Dataset
    logging.info("2. Load Dataset")
    train_dataset = POIDataset(data_filename="datasets/{}/train_session_label.pkl".format(args.dataset),
                               pois_coos_filename="datasets/{}/{}_pois_coos_poi_zero.pkl".format(args.dataset, args.dataset),
                               num_users=NUM_USERS,
                               num_pois=NUM_POIS,
                               padding_idx=PADDING_IDX,
                               args=args,
                               device=device)

    valid_dataset = POIDataset(data_filename="datasets/{}/valid_session_label.pkl".format(args.dataset),
                              pois_coos_filename="datasets/{}/{}_pois_coos_poi_zero.pkl".format(args.dataset, args.dataset),
                              num_users=NUM_USERS,
                              num_pois=NUM_POIS,
                              padding_idx=PADDING_IDX,
                              args=args,
                              device=device)

    test_dataset = POIDataset(data_filename="datasets/{}/test_session_label.pkl".format(args.dataset),
                              pois_coos_filename="datasets/{}/{}_pois_coos_poi_zero.pkl".format(args.dataset, args.dataset),
                              num_users=NUM_USERS,
                              num_pois=NUM_POIS,
                              padding_idx=PADDING_IDX,
                              args=args,
                              device=device) # 原本是test_poi_zero.txt  对于SH是test_session_label.pkl

    # 3. Construct DataLoader
    logging.info("3. Construct DataLoader")
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=lambda batch: collate_fn_4sq(batch, padding_value=PADDING_IDX))
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=True,
                                 collate_fn=lambda batch: collate_fn_4sq(batch, padding_value=PADDING_IDX))
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True,
                                 collate_fn=lambda batch: collate_fn_4sq(batch, padding_value=PADDING_IDX))


    # Load Model
    logging.info("4. Load Model")
    model = DCHL(NUM_USERS, NUM_POIS,args, device)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    criterion = nn.CrossEntropyLoss().to(device)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', verbose=True, factor=args.lr_scheduler_factor)

    # Train
    logging.info("5. Start Training")
    Ks_list = [1, 5, 10, 20]
    final_results = {"Rec1": 0.0, "Rec5": 0.0, "Rec10": 0.0, "Rec20": 0.0,
                     "NDCG1": 0.0, "NDCG5": 0.0, "NDCG10": 0.0, "NDCG20": 0.0,
                     }

    monitor_loss = float('inf')
    best_valid_rec5 = 0.0
    saved_model_path = ''
    for epoch in range(args.num_epochs):
        logging.info("================= Epoch {}/{} =================".format(epoch, args.num_epochs))
        start_time = time.time()
        model.train()

        train_loss = 0.0

        # to save recall and ndcg results
        train_recall_array = np.zeros(shape=(len(train_dataloader), len(Ks_list)))
        train_ndcg_array = np.zeros(shape=(len(train_dataloader), len(Ks_list)))
        for idx, batch in enumerate(train_dataloader):
            logging.info("Train. Batch {}/{}".format(idx, len(train_dataloader)))
            optimizer.zero_grad()

            predictions, loss_cl_users, loss_cl_pois, loss_cl_intra = model(train_dataset, batch)
            # predictions, loss_cl_users, loss_cl_pois, loss_cl_HyperEdge = model(train_dataset, batch)

            # calculate loss
            loss_rec = criterion(predictions, batch["label"].to(device))
            # loss = loss_rec
            # loss = loss_rec + args.lambda_cl * (loss_cl_pois + loss_cl_users)
            loss = loss_rec + args.lambda_cl * (loss_cl_pois + loss_cl_intra)
            # loss = loss_rec + args.lambda_cl * (loss_cl_intra)
            # loss = loss_rec + args.lambda_cl * (0.2 * loss_cl_pois + 0.8 * loss_cl_intra)
            # loss = loss_rec + 0.7*loss_cl_pois + 0.3*loss_cl_intra
            logging.info("Train. loss_rec: {:.4f}; loss_cl_pois: {:.4f}; loss_cl_users: {:.4f}; "
                         "loss: {:.4f}".format(loss_rec.item(), loss_cl_pois, loss_cl_users, loss))

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            for k in Ks_list:
                recall, ndcg = batch_performance(predictions.detach().cpu(), batch["label"].detach().cpu(), k)
                col_idx = Ks_list.index(k)
                train_recall_array[idx, col_idx] = recall
                train_ndcg_array[idx, col_idx] = ndcg

        logging.info("Training finishes at this epoch. It takes {} min".format((time.time() - start_time) / 60))
        logging.info("Training loss: {:.4f}".format(train_loss / len(train_dataloader)))
        logging.info("Training Epoch {}/{} results:".format(epoch, args.num_epochs))
        for k in Ks_list:
            col_idx = Ks_list.index(k)
            logging.info("Recall@{}: {:.4f}".format(k, np.mean(train_recall_array[:, col_idx])))
            logging.info("NDCG@{}: {:.4f}".format(k, np.mean(train_ndcg_array[:, col_idx])))
        logging.info("\n")


        # validating
        logging.info("Validating")
        valid_loss = 0.0
        valid_recall_array = np.zeros(shape=(len(valid_dataloader), len(Ks_list)))
        valid_ndcg_array = np.zeros(shape=(len(valid_dataloader), len(Ks_list)))

        model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(valid_dataloader):

                logging.info("Valid. Batch {}/{}".format(idx, len(valid_dataloader)))

                predictions, loss_cl_users, loss_cl_pois, loss_cl_intra = model(valid_dataset, batch)
                # calculate loss
                loss_rec = criterion(predictions, batch["label"].to(device))
                loss = loss_rec + args.lambda_cl * (0.8*loss_cl_pois + 0.2*loss_cl_intra)

                valid_loss += loss.item()

                for k in Ks_list:
                    recall, ndcg = batch_performance(predictions.detach().cpu(), batch["label"].detach().cpu(), k)
                    col_idx = Ks_list.index(k)
                    valid_recall_array[idx, col_idx] = recall
                    valid_ndcg_array[idx, col_idx] = ndcg

        logging.info("Validating finishes")
        logging.info("Validating loss: {}".format(valid_loss / len(valid_dataloader)))
        logging.info("Validating results:")
        for k in Ks_list:
            col_idx = Ks_list.index(k)
            recall = np.mean(valid_recall_array[:, col_idx])
            ndcg = np.mean(valid_ndcg_array[:, col_idx])
            logging.info("Recall@{}: {:.4f}".format(k, recall))
            logging.info("NDCG@{}: {:.4f}".format(k, ndcg))

        # Check monitor loss and monitor score for updating
        monitor_loss = min(monitor_loss, valid_loss)

        # Learning rate schuduler
        lr_scheduler.step(monitor_loss)

        # update best_valid_rec5
        valid_recall5 = np.mean(valid_recall_array[:, 1])
        if valid_recall5 > best_valid_rec5:
            best_valid_rec5 = valid_recall5
            logging.info("Update valid results and save model at epoch{}".format(epoch))

            # define saved_model_path
            saved_model_path = os.path.join(current_save_dir, "{}.pt".format(args.dataset))
            torch.save(model, saved_model_path)


    # testing
    logging.info("Testing")
    test_loss = 0.0
    test_recall_array = np.zeros(shape=(len(test_dataloader), len(Ks_list)))
    test_ndcg_array = np.zeros(shape=(len(test_dataloader), len(Ks_list)))
    best_model = torch.load(saved_model_path)
    with torch.no_grad():
        for idx, batch in enumerate(test_dataloader):

            logging.info("Test. Batch {}/{}".format(idx, len(test_dataloader)))

            predictions, loss_cl_users, loss_cl_pois, loss_cl_intra = best_model(test_dataset, batch)
            # calculate loss
            loss_rec = criterion(predictions, batch["label"].to(device))
            loss = loss_rec + args.lambda_cl * (0.8*loss_cl_pois + 0.2*loss_cl_intra)
            # logging.info("Test. loss_rec: {:.4f}; loss_cl_pois: {:.4f}; loss_cl_users: {:.4f}; "
            #              "loss: {:.4f}".format(loss_rec.item(), loss_cl_pois, loss_cl_users, loss))

            test_loss += loss.item()

            for k in Ks_list:
                recall, ndcg = batch_performance(predictions.detach().cpu(), batch["label"].detach().cpu(), k)
                col_idx = Ks_list.index(k)
                test_recall_array[idx, col_idx] = recall
                test_ndcg_array[idx, col_idx] = ndcg


    # update best result
    for k in Ks_list:
        if k == 1:
            final_results["Rec1"] = max(final_results["Rec1"], np.mean(test_recall_array[:, 0]))
            final_results["NDCG1"] = max(final_results["NDCG1"], np.mean(test_ndcg_array[:, 0]))

        elif k == 5:
            final_results["Rec5"] = max(final_results["Rec5"], np.mean(test_recall_array[:, 1]))
            final_results["NDCG5"] = max(final_results["NDCG5"], np.mean(test_ndcg_array[:, 1]))

        elif k == 10:
            final_results["Rec10"] = max(final_results["Rec10"], np.mean(test_recall_array[:, 2]))
            final_results["NDCG10"] = max(final_results["NDCG10"], np.mean(test_ndcg_array[:, 2]))

        elif k == 20:
            final_results["Rec20"] = max(final_results["Rec20"], np.mean(test_recall_array[:, 3]))
            final_results["NDCG20"] = max(final_results["NDCG20"], np.mean(test_ndcg_array[:, 3]))
    logging.info("==================================\n\n")

    logging.info("6. Final Results")
    formatted_dict = {key: f"{value:.4f}" for key, value in final_results.items()}
    logging.info(formatted_dict)
    logging.info("\n")


if __name__ == '__main__':
    main()

