import argparse
import numpy as np
import torch
import torch.nn

from parse import parser_add_main_args
from data_utils import *
from backbone import *
from diffusion_util import *
from gold_utils import *

parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()

# Intialise Model and data
diff_energy_model, device, criterion, eval_func, logger, ind_data, ood_train, ood_test, \
        denoise_model, denoise_optimizer, main_optimizer, mlp, combined_optimizer, scheduler, betas, \
                 sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, params = initialise_setup(args.lr, args, use_detector = args.use_detector)
fix_seed(args.seed)
train_split = ind_data.splits['train']
val_split = ind_data.splits['valid']
train_y = ind_data.y[train_split]

if isinstance(ood_test, list):
    tests = [ood_train] + ood_test
else:
    tests = [ood_train] + [ood_test]

# Check if using ckpt model
if args.use_saved:
    if args.dataset in ["twitch", "arxiv"]:
        save_path = f'{args.model_path}/{args.dataset}.pth'
    else:
        save_path = f'../model_weights/{args.dataset}_{args.ood_type}.pth'

    print("Loaded Model")
    diff_energy_model.load_state_dict(torch.load(save_path, weights_only=True)['classifier'])
    mlp.load_state_dict(torch.load(save_path, weights_only=True)['detector'])
else:
    # Warm Up Training
    best_val_loss = float("inf")
    for disr in range(0, params["disr_iter"]):
        diff_energy_model.train()
        main_optimizer.zero_grad()
        logits, x_g = diff_energy_model(ind_data, device)
        pred_id = F.log_softmax(logits[train_split], dim=-1).to(device)
        id_sup_loss = criterion(pred_id, train_y.flatten().cuda())

        id_sup_loss.backward()
        main_optimizer.step()

    diff_energy_model.eval()
    logits, gen_data = diff_energy_model(ind_data, device)
    denoise_model.train()
            
    for diff_iter in range(1, params["denoise_iter"]):
        t = torch.randint(0, params["time_steps"], (gen_data.size(0),), device=device).long()
        diffusion_loss, predicted_noise, x_noisy = p_losses(denoise_model, gen_data, t, None, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type="huber")

        denoise_optimizer.zero_grad()
        diffusion_loss.backward(retain_graph=True)
        denoise_optimizer.step()

    # Begin Adversarial training
    for epoch in range(1, args.adv_epochs+1):
        start_training = True
        mlp.train()
        diff_energy_model.train()
        denoise_model.eval()

        samp = torch.randn(x_g.shape, device=device)
        syn_latent = draw_samples(samp, denoise_model, None, params["time_steps"], betas, x_g.shape, device)[-1]
        for main_iter in range(0, params["iter_num"]):
            syn_logits = diff_energy_model.encoder.convs[-1](syn_latent, ind_data.edge_index.to(device)).to(device)

            logits, x_g = diff_energy_model(ind_data, device)
            pred_id = F.log_softmax(logits[train_split], dim=1).to(device)
            id_sup_loss = criterion(pred_id, train_y.flatten().cuda())

            id_energy = - torch.logsumexp(logits, dim=-1)
            syn_energy = - torch.logsumexp(syn_logits, dim=-1)
            if args.prop_gcn:
                id_energy = diff_energy_model.propagation(id_energy, ind_data.edge_index.to(device), args.K, args.alpha)
                syn_energy = diff_energy_model.propagation(syn_energy, ind_data.edge_index.to(device), args.K, args.alpha)

            if args.use_detector:
                input_for_lr = torch.cat((id_energy, syn_energy), -1)
                labels_for_lr = torch.cat((torch.ones(len(id_energy)).cuda(), torch.zeros(len(syn_energy)).cuda()), -1)
                mlp_logits, _ = mlp(input_for_lr.view(-1, 1))

                id_mlp_energy = - torch.logsumexp(mlp_logits[:len(id_energy)], dim=-1)
                syn_mlp_energy = - torch.logsumexp(mlp_logits[len(id_energy):], dim=-1)
                if args.prop_mlp:
                    id_mlp_energy = diff_energy_model.propagation(id_mlp_energy, ind_data.edge_index.to(device), args.K, args.alpha)[train_split]
                    syn_mlp_energy = diff_energy_model.propagation(syn_mlp_energy, ind_data.edge_index.to(device), args.K, args.alpha)[train_split]
                else:
                    id_mlp_energy = id_mlp_energy[train_split]
                    syn_mlp_energy = syn_mlp_energy[train_split]

            id_energy = id_energy[train_split]
            syn_energy = syn_energy[train_split]

            if args.use_detector:
                input_for_lr = torch.cat((id_energy, syn_energy), -1)
                labels_for_lr = torch.cat((torch.ones(len(id_energy)).cuda(), torch.zeros(len(syn_energy)).cuda()), -1)
                mlp_logits, _ = mlp(input_for_lr.view(-1, 1))
                crit = torch.nn.CrossEntropyLoss()
                mlp_sup_loss = crit(mlp_logits, labels_for_lr.long())

            reg_loss = torch.mean(F.relu(id_energy - args.m_in) ** 2) + torch.mean(F.relu(args.m_out - syn_energy) ** 2)
            if args.use_detector:
                mlp_reg_loss = torch.mean(F.relu(id_mlp_energy - id_energy) ** 2) + torch.mean(F.relu(syn_energy - syn_mlp_energy) ** 2)
                loss = id_sup_loss + params["w_reg"] * reg_loss + params["w_lreg"] * mlp_sup_loss + params["w_mlp"] * mlp_reg_loss
            else:
                loss = id_sup_loss + params["w_reg"] * reg_loss
            
            combined_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            combined_optimizer.step()

            diff_energy_model.eval()
            valid_idx = ind_data.splits['valid']
            logits, _ = diff_energy_model(ind_data, device)
            valid_out = F.log_softmax(logits[valid_idx], dim=1)
            valid_loss_energy = criterion(valid_out, ind_data.y[valid_idx].squeeze(1).to(device))
            
        if (epoch != params["epoch_num"]):
            denoise_model.train()
            diff_energy_model.eval()
            logits, gen_data = diff_energy_model(ind_data, device)
            for diff_iter in range(1, params["denoise_iter"]):
                t = torch.randint(0, params["time_steps"], (gen_data.size(0),), device=device).long()
                diffusion_loss, predicted_noise, x_noisy = p_losses(denoise_model, gen_data, t, None, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type="huber")
                
                denoise_optimizer.zero_grad()
                diffusion_loss.backward(retain_graph=True)
                denoise_optimizer.step()

        weights = [params["w_reg"], params["w_lreg"], params["w_mlp"]]
        valid_id, test_id, valid_loss, ood_res, avg_res, best_val_loss = evaluate_performance(diff_energy_model, mlp, ind_data, tests, syn_energy, eval_func, criterion, args, device, weights, \
                    epoch, best_val_loss, prop_gcn = args.prop_gcn, prop_mlp = args.prop_mlp, use_detector = args.use_detector, verbose = False)
        
### Model Evaluation
evaluate_performance(diff_energy_model, mlp, ind_data, tests, None, eval_func, criterion, args, device, None, None, None, \
                    prop_gcn = args.prop_gcn, prop_mlp = args.prop_mlp, use_detector = args.use_detector, verbose = True, test = True)