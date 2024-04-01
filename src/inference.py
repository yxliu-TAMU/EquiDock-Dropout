# -*- coding: utf-8 -*-
#
import os
import time

from src.utils.io import create_dir

os.environ['DGLBACKEND'] = 'pytorch'
from datetime import datetime as dt
from src.utils.train_utils import *
from src.utils.args import *
from src.utils.ot_utils import *
from src.utils.eval import Meter_Unbound_Bound
from src.utils.early_stop import EarlyStopping
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing
import random
import torch
from torch.utils.data import DataLoader
from src.utils.db5_data import Unbound_Bound_Data
from src.model.rigid_docking_model import *
import numpy as np
def G_fn(protein_coords, x, sigma):
    # protein_coords: (n,3) ,  x: (m,3), output: (m,)
    e = torch.exp(- torch.sum((protein_coords.view(1, -1, 3) - x.view(-1,1,3)) ** 2, dim=2) / float(sigma) )  # (m, n)
    return - sigma * torch.log(1e-3 +  e.sum(dim=1) )
### Create log files only when in train mode
def get_dataloader(args):
    num_worker = 0
    train_set = Unbound_Bound_Data(args, if_swap=True, reload_mode='train', load_from_cache=True, data_fraction=args['data_fraction'])
    val_set = Unbound_Bound_Data(args, if_swap=False, reload_mode='val', load_from_cache=True)
    test_set = Unbound_Bound_Data(args, if_swap=False, reload_mode='test', load_from_cache=True)

    train_loader = DataLoader(dataset=train_set,
                              batch_size=args['bs'],
                              shuffle=True,
                              collate_fn=partial(batchify_and_create_hetero_graphs),
                              num_workers=num_worker,
                              )
    val_loader = DataLoader(dataset=val_set,
                            batch_size=args['bs'],
                            collate_fn=partial(batchify_and_create_hetero_graphs),
                            num_workers=num_worker
                            )
    test_loader = DataLoader(dataset=test_set,
                             batch_size=args['bs'],
                             collate_fn=partial(batchify_and_create_hetero_graphs),
                             num_workers=num_worker)
    args['input_edge_feats_dim'] = train_set[0][0].edata['he'].shape[1]

    return train_loader, val_loader, test_loader

def create_model(args):
    assert 'input_edge_feats_dim' in args.keys(), 'get_loader has to be called before create_model.'

    return Rigid_Body_Docking_Net(args=args)

def run_a_generic_epoch(ep_type, args, epoch, model, data_loader, loss_fn_coors, optimizer):
    time.sleep(2)

    meter = Meter_Unbound_Bound()

    avg_loss, total_loss, num_batches = 0., 0., 0

    total_loss_ligand_coors = 0.
    avg_loss_ligand_coors = 0.

    total_loss_receptor_coors = 0.
    avg_loss_receptor_coors = 0.

    total_loss_ot = 0.
    avg_loss_ot = 0.

    total_loss_intersection = 0.
    avg_loss_intersection = 0.


    num_clip = 0
    num_total_possible_clips = 0
    loader = tqdm(data_loader)

    for batch_id, batch_data in enumerate(loader):
        num_batches += 1

        if ep_type == 'train':
            optimizer.zero_grad()

        batch_hetero_graph, \
        bound_ligand_repres_nodes_loc_array_list, bound_receptor_repres_nodes_loc_array_list, \
        pocket_coors_ligand_list, pocket_coors_receptor_list = batch_data

        batch_hetero_graph = batch_hetero_graph.to(args['device'])


        ######## RUN MODEL ##############
        model_ligand_coors_deform_list, \
        model_keypts_ligand_list, model_keypts_receptor_list, \
        _, _,  = model(batch_hetero_graph, epoch=epoch)
        ################################

        # Compute MSE loss for each protein individually, then average over the minibatch.
        batch_ligand_coors_loss = torch.zeros([]).to(args['device'])
        batch_receptor_coors_loss = torch.zeros([]).to(args['device'])
        batch_ot_loss = torch.zeros([]).to(args['device'])
        batch_intersection_loss = torch.zeros([]).to(args['device'])

        assert len(pocket_coors_ligand_list) == len(model_ligand_coors_deform_list)
        assert len(bound_ligand_repres_nodes_loc_array_list) == len(model_ligand_coors_deform_list)

        for i in range(len(model_ligand_coors_deform_list)):
            ## Compute average MSE loss (which is 3 times smaller than average squared RMSD)
            batch_ligand_coors_loss = batch_ligand_coors_loss + loss_fn_coors(model_ligand_coors_deform_list[i],
                                                                              bound_ligand_repres_nodes_loc_array_list[i].to(args['device']))

            # Compute the OT loss for the binding pocket:
            ligand_pocket_coors = pocket_coors_ligand_list[i].to(args['device'])  ##  (N, 3), N = num pocket nodes
            receptor_pocket_coors = pocket_coors_receptor_list[i].to(args['device'])  ##  (N, 3), N = num pocket nodes

            ligand_keypts_coors = model_keypts_ligand_list[i]  ##  (K, 3), K = num keypoints
            receptor_keypts_coors = model_keypts_receptor_list[i]  ##  (K, 3), K = num keypoints

            ## (N, K) cost matrix
            cost_mat_ligand = compute_sq_dist_mat(ligand_pocket_coors, ligand_keypts_coors)
            cost_mat_receptor = compute_sq_dist_mat(receptor_pocket_coors, receptor_keypts_coors)

            ot_dist, _ = compute_ot_emd(cost_mat_ligand + cost_mat_receptor, args['device'])
            batch_ot_loss = batch_ot_loss + ot_dist

            batch_intersection_loss = batch_intersection_loss + compute_body_intersection_loss(
                model_ligand_coors_deform_list[i], bound_receptor_repres_nodes_loc_array_list[i].to(args['device']),
                args['intersection_sigma'], args['intersection_surface_ct'])

            ### Add new stats to the meter
            if ep_type != 'train' or random.random() < 0.1:
                meter.update_rmsd(model_ligand_coors_deform_list[i],
                                  bound_receptor_repres_nodes_loc_array_list[i],
                                  bound_ligand_repres_nodes_loc_array_list[i],
                                  bound_receptor_repres_nodes_loc_array_list[i])


        batch_ligand_coors_loss = batch_ligand_coors_loss / float(len(model_ligand_coors_deform_list))
        batch_receptor_coors_loss = batch_receptor_coors_loss / float(len(model_ligand_coors_deform_list))
        batch_ot_loss = batch_ot_loss / float(len(model_ligand_coors_deform_list))
        batch_intersection_loss = batch_intersection_loss  / float(len(model_ligand_coors_deform_list))

        loss_coors = batch_ligand_coors_loss + batch_receptor_coors_loss

        loss = loss_coors + args['pocket_ot_loss_weight'] * batch_ot_loss + args['intersection_loss_weight'] * batch_intersection_loss

        #########
        if ep_type == 'train':
            loss.backward()

            clip = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args['clip'], norm_type=2)
            if clip > args['clip']:
                # gradient exploded
                # if clip > args['clip'] * 100 and num_batches > 1:
                    # log(f"Gradient Exploded: {clip}")
                    # optimizer.zero_grad()
                num_clip += 1
            num_total_possible_clips += 1

            optimizer.step()

            for name, param in model.named_parameters():
                if param.norm() > 500.:
                    log('    PARAM: ', name, ' --> norm = ', param.norm(), ' --> grad = ', param.grad.norm())
        ###########

        total_loss += loss.detach()
        total_loss_ligand_coors += batch_ligand_coors_loss.detach()
        total_loss_receptor_coors += batch_receptor_coors_loss.detach()
        total_loss_ot += batch_ot_loss.detach()
        total_loss_intersection += batch_intersection_loss.detach()

        if batch_id % args['log_every'] == args['log_every']-1:
            log('batch {:.0f}% || Loss {:.6f}'.format((100. * batch_id) / len(data_loader), loss.item()))


    if num_batches != 0:
        avg_loss = total_loss / num_batches
        avg_loss_ligand_coors = total_loss_ligand_coors / num_batches
        avg_loss_receptor_coors = total_loss_receptor_coors / num_batches
        avg_loss_ot = total_loss_ot / num_batches
        avg_loss_intersection = total_loss_intersection / num_batches

    ligand_rmsd_list, receptor_rmsd_list, complex_rmsd_list = meter.no_summarize()


    #########


    return ligand_rmsd_list, receptor_rmsd_list, complex_rmsd_list
           
def run_an_eval_epoch(args, model, data_loader, loss_fn_coors=None):
    with torch.no_grad():
        model.eval()
        ligand_rmsd_list, receptor_rmsd_list, complex_rmsd_list = \
            run_a_generic_epoch('eval', args=args, epoch=-1, model=model, data_loader=data_loader,
                                loss_fn_coors=loss_fn_coors, optimizer=None)

    return ligand_rmsd_list, receptor_rmsd_list, complex_rmsd_list
def compute_body_intersection_loss(model_ligand_coors_deform, bound_receptor_repres_nodes_loc_array, sigma, surface_ct):
    loss = torch.mean( torch.clamp(surface_ct - G_fn(bound_receptor_repres_nodes_loc_array, model_ligand_coors_deform, sigma), min=0) ) + \
           torch.mean( torch.clamp(surface_ct - G_fn(model_ligand_coors_deform, bound_receptor_repres_nodes_loc_array, sigma), min=0) )
    return loss
def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def get_monte_carlo_predictions(forward_passes,model,test_loader,loss_fn_coors,args,n_samples):
    """ Function to get the monte-carlo samples and uncertainty estimates
    through multiple forward passes

    Parameters
    ----------
    data_loader : object
        data loader object from the data loader module
    forward_passes : int
        number of monte-carlo samples/forward passes
    model : object
        keras model
    n_classes : int
        number of classes in the dataset
    n_samples : int
        number of samples in the test set
    """

    dropout_predictions = np.empty((n_samples, forward_passes))
    softmax = nn.Softmax(dim=1)
    for i in range(forward_passes):
        model.eval()
        enable_dropout(model)
        ligand_rmsd_list, receptor_rmsd_list, complex_rmsd_list = run_an_eval_epoch(args, model, test_loader, loss_fn_coors=nn.MSELoss(reduction='mean'))
        #print(complex_rmsd_list)
        dropout_predictions[:, i] = complex_rmsd_list
        # dropout predictions - shape (forward_passes, n_samples, n_classes)

    # Calculating mean across multiple MCD forward passes 
    #mean = np.mean(dropout_predictions, axis=0)  # shape (n_samples, n_classes)

    # Calculating variance across multiple MCD forward passes 
    #variance = np.var(dropout_predictions, axis=0)  # shape (n_samples, n_classes)

    #epsilon = sys.float_info.min
    # Calculating entropy across multiple MCD forward passes 
    #entropy = -np.sum(mean * np.log(mean + epsilon), axis=-1)  # shape (n_samples,)

    # Calculating mutual information across multiple MCD forward passes 
    #mutual_info = entropy - np.mean(np.sum(-dropout_predictions * np.log(dropout_predictions + epsilon),
                                           #axis=-1), axis=0)  # shape (n_samples,)
    return dropout_predictions
def main(args):
    checkpoint_dir = "./checkpts/EQUIDOCK__dropout_{}_drop_connect_{}_drop_connect_rate_{}_drop_message_{}_drop_message_rate_{}_num_layers_{}".format(args['dropout'], args['drop_connect'], args['drop_connect_rate'], args['drop_message'], args['drop_message_rate'], args['iegmn_n_lays'])
    checkpoint_dir = checkpoint_dir + "/db5_model_best.pth"
    args['cache_path'] = os.path.join(args['cache_path'], 'cv_' + str(args['split']))
    checkpoint = torch.load(checkpoint_dir)
    if args["data"]=="dips":
        checkpoint_dir = "./checkpts/dips_EQUIDOCK__dropout_{}_drop_connect_{}_drop_connect_rate_{}_drop_message_{}_drop_message_rate_{}_num_layers_{}".format(args['dropout'], args['drop_connect'], args['drop_connect_rate'], args['drop_message'], args['drop_message_rate'], args['iegmn_n_lays'])
        checkpoint_dir = checkpoint_dir + "/dips_model_best.pth"
        checkpoint = torch.load(checkpoint_dir)
        #args['cache_path'] = args['cache_path']
    train_loader, val_loader, test_loader = get_dataloader(args)
    model = create_model(args)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(args['device'])
    ligand_rmsd_list, receptor_rmsd_list, complex_rmsd_list = run_an_eval_epoch(args, model, test_loader, loss_fn_coors=nn.MSELoss(reduction='mean'))
    print("average complex rmsd: ",np.mean(complex_rmsd_list),np.median(complex_rmsd_list))
    dropout_predictions = get_monte_carlo_predictions(20,model,test_loader,nn.MSELoss(reduction='mean'),args,965)
    prediction_variance = np.var(dropout_predictions, axis=1)
    average_estimation = np.mean(dropout_predictions, axis=1)
    threshold_list = [1e-10,1e-8,1e-6,1e-4,1e-2,1]
    PAvPU = np.zeros(len(threshold_list))
    i = 0
    for threshold in threshold_list:
        ac = np.sum(np.multiply(average_estimation<10,prediction_variance<threshold))
        iu = np.sum(np.multiply(average_estimation>=10,prediction_variance>threshold))
        print(iu)
        PAvPU[i] = (ac+iu)/25
        i = i+1
    return PAvPU,dropout_predictions
    

if __name__ == "__main__":
    print(args["dropout"])
    print(args['patience'])
    PAvPU,dropout_predictions = main(args)
    np.save("5_dropout_predictions.npy",dropout_predictions)