import torch
from dataset import load_dataset
import numpy as np
import random
from data_utils import *
from gnnsafe import *
from logger import Logger_classify, Logger_detect, save_result
from backbone import *
from diffusion_util import linear_beta_schedule, DenoiseNN


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_energy(model, ind_data, test, logits, logits_ood, args, pos = True, prop = True, device = None):
    T = 1
    if pos:
        T = -1
    
    energy_id = T * torch.logsumexp(logits, dim=-1)
    energy_ood = T * torch.logsumexp(logits_ood, dim=-1)

    if prop:
        energy_id = model.propagation(energy_id, ind_data.edge_index.to(device), args.K, args.alpha)
        energy_ood = model.propagation(energy_ood, test.edge_index.to(device), args.K, args.alpha)

    energy_id = energy_id.cpu().detach().numpy()
    energy_ood = energy_ood.cpu().detach().numpy()
    return energy_id, energy_ood

def initialise(args):
    fix_seed(args.seed)

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    dataset_ind, dataset_ood_tr, dataset_ood_te = load_dataset(args)
    if len(dataset_ind.y.shape) == 1:
        dataset_ind.y = dataset_ind.y.unsqueeze(1)
    if len(dataset_ood_tr.y.shape) == 1:
        dataset_ood_tr.y = dataset_ood_tr.y.unsqueeze(1)
    if isinstance(dataset_ood_te, list):
        for data in dataset_ood_te:
            if len(data.y.shape) == 1:
                data.y = data.y.unsqueeze(1)
    else:
        if len(dataset_ood_te.y.shape) == 1:
            dataset_ood_te.y = dataset_ood_te.y.unsqueeze(1)
            
    ### get splits for all runs ###
    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        pass
    else:
        dataset_ind.splits = rand_splits(dataset_ind.node_idx, train_prop=args.train_prop, valid_prop=args.valid_prop)

    ### print dataset info ###
    c = max(dataset_ind.y.max().item() + 1, dataset_ind.y.shape[1])
    d = dataset_ind.x.shape[1]

    print(f"ind dataset {args.dataset}: all nodes {dataset_ind.num_nodes} | centered nodes {dataset_ind.node_idx.shape[0]} | edges {dataset_ind.edge_index.size(1)} | "
        + f"classes {c} | feats {d}")
    print(f"ood tr dataset {args.dataset}: all nodes {dataset_ood_tr.num_nodes} | centered nodes {dataset_ood_tr.node_idx.shape[0]} | edges {dataset_ood_tr.edge_index.size(1)}")
    if isinstance(dataset_ood_te, list):
        for i, data in enumerate(dataset_ood_te):
            print(f"ood te dataset {i} {args.dataset}: all nodes {data.num_nodes} | centered nodes {data.node_idx.shape[0]} | edges {data.edge_index.size(1)}")
    else:
        print(f"ood te dataset {args.dataset}: all nodes {dataset_ood_te.num_nodes} | centered nodes {dataset_ood_te.node_idx.shape[0]} | edges {dataset_ood_te.edge_index.size(1)}")

    model = GNNSafe(d, c, args).to(device)
    criterion = nn.NLLLoss()

    if args.dataset in ('proteins', 'ppi', 'twitch'): # binary classification
        eval_func = eval_rocauc
    else:
        eval_func = eval_acc
    logger = Logger_detect(args.runs, args)
    model.train()
    return model, device, criterion, eval_func, logger, dataset_ind, dataset_ood_tr, dataset_ood_te

def dataset_params(args):
    params = {"denoise_iter": 600, "time_steps": 600, "denoise_hidden": 256, "denoise_layers": 3, "denoise_lr": 0.001, \
              "mlp_hidden": 512, "mlp_out": 2, "mlp_layers": 3, "mlp_lr": 0.001,  "disr_iter": 100}
    if args.dataset == "twitch":
        temp = {"denoise_iter": 200, "denoise_layers": 2, "disr_iter": 21, "epoch_num": 3, "iter_num": 7, "w_lreg": 1, "w_reg": 0.3, "w_mlp": 1}
        params.update(temp)
    elif args.dataset == "cora":
        if args.ood_type == "structure":
            temp = {"denoise_layers": 2,"epoch_num": 5,"iter_num": 8, "w_lreg": 1,"w_reg": 1,"w_mlp": 1}
            params.update(temp)
        elif args.ood_type == "feature":
            temp = {"denoise_hidden": 512,"denoise_layers": 2,"epoch_num": 2,"iter_num": 15, "w_lreg": 1,"w_reg": 1,"w_mlp": 1}
            params.update(temp)
        elif args.ood_type == "label":
            temp = {"time_steps": 800,"denoise_hidden": 512,"disr_iter": 50,"epoch_num": 5,"iter_num": 5, "w_lreg": 1,"w_reg": 1,"w_mlp": 1}
            params.update(temp)
    elif args.dataset == "amazon-photo":
        if args.ood_type == "structure":
            temp = {"time_steps": 800, "denoise_hidden": 512,"epoch_num": 10,"iter_num": 5, "w_lreg": 1,"w_reg": 0.1,"w_mlp": 0.1}
            params.update(temp)
        elif args.ood_type == "feature":
            temp = {"denoise_iter": 300,"time_steps": 800, "denoise_hidden": 512,"epoch_num": 10,"iter_num": 7, "w_lreg": 1,"w_reg": 0.1,"w_mlp": 0.1}
            params.update(temp)
        else:
            temp = {"denoise_iter": 800,"time_steps": 800,"denoise_hidden": 512,"mlp_hidden": 128,"mlp_layers": 2,"epoch_num": 10,"iter_num": 7, "w_lreg": 1,"w_reg": 0.1,"w_mlp": 0.1}
            params.update(temp)
    elif args.dataset == "coauthor-cs":
        if args.ood_type == "structure":
            temp = {"denoise_iter": 800,"mlp_hidden": 128,"mlp_layers": 2,"epoch_num": 5,"iter_num": 8, "w_lreg": 1,"w_reg": 0.1,"w_mlp": 0.1}
            params.update(temp)
        elif args.ood_type == "feature":
            temp = {"denoise_iter": 800,"mlp_hidden": 128,"mlp_layers": 2,"epoch_num": 4,"iter_num": 6, "w_lreg": 1,"w_reg": 0.1,"w_mlp": 0.1}
            params.update(temp)
        else:
            temp = {"denoise_iter": 800,"mlp_hidden": 128,"mlp_layers": 2,"disr_iter": 55,"epoch_num": 10,"iter_num": 5, "w_lreg": 1,"w_reg": 0.1,"w_mlp": 0.1}
            params.update(temp)
    elif args.dataset == "arxiv":
        temp = {"denoise_iter": 200,"denoise_hidden": 512,"denoise_layers": 2,"epoch_num": 8,"iter_num": 8, "w_lreg": 1,"w_reg": 0.1,"w_mlp": 0.1}
        params.update(temp)
    return params

def initialise_setup(lr, args, use_detector = True):
    params = dataset_params(args)

    betas = linear_beta_schedule(timesteps=params["time_steps"])
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    fix_seed(args.seed)
    diff_energy_model, device, criterion, eval_func, logger, dataset_ind, dataset_ood_tr, dataset_ood_te = initialise(args)
    diff_energy_model.reset_parameters()
    mlp = MLP(in_channels = 1, hidden_channels=params["mlp_hidden"], out_channels=params["mlp_out"], num_layers=params["mlp_layers"], dropout=0).to(device)
    main_optimizer = torch.optim.Adam(diff_energy_model.parameters(), lr=lr)
    denoise_model = DenoiseNN(input_dim=args.hidden_channels, hidden_dim=params["denoise_hidden"], n_layers=params["denoise_layers"], n_cond=0, d_cond=0).to(device)
    denoise_optimizer = torch.optim.Adam(denoise_model.parameters(), lr=params["denoise_lr"])

    if use_detector:
        combined_optimizer = torch.optim.Adam(
                        [{'params': diff_energy_model.parameters()},
                        {'params': mlp.parameters(), 'lr': params["mlp_lr"]}], lr= lr)
    else:
        combined_optimizer = torch.optim.Adam([{'params': diff_energy_model.parameters()}], lr = lr)

    scheduler = torch.optim.lr_scheduler.StepLR(main_optimizer, step_size=500, gamma=0.1)
    return diff_energy_model, device, criterion, eval_func, logger, dataset_ind, dataset_ood_tr, dataset_ood_te, \
        denoise_model, denoise_optimizer, main_optimizer, mlp, combined_optimizer, scheduler, betas, \
            sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, params

def evaluate_performance(model, detector, ind_data, test_data, syn_energy, eval_func, criterion, args, device, weights, \
                         epoch, best_val_loss, prop_gcn = True, prop_mlp = True, use_detector = True, verbose = False, test = False):
    model.eval()
    detector.eval()
    _, valid_id, test_id, valid_loss = evaluate_classify(model, ind_data, eval_func, criterion, args, device, test_only = False, decoder = None)
    
    if not test:
        crit = torch.nn.CrossEntropyLoss()
        valid_idx = ind_data.splits['valid']
        logits, _ = model(ind_data, device)
        valid_out = F.log_softmax(logits[valid_idx], dim=1)
        val_loss_id = criterion(valid_out, ind_data.y[valid_idx].squeeze(1).to(device))
        val_energy = - torch.logsumexp(logits[valid_idx], dim=-1)
        val_reg_loss = torch.mean(F.relu(val_energy - args.m_in) ** 2) + torch.mean(F.relu(args.m_out - syn_energy) ** 2)

        val_energies = torch.cat((syn_energy, val_energy), -1)
        val_labels = torch.cat((torch.zeros(len(syn_energy)).cuda(), torch.ones(len(val_energy)).cuda()), -1)
        val_mlp_logits, _ = detector(val_energies.view(-1, 1))
        val_mlp = crit(val_mlp_logits, val_labels.long())
        val_mlp_energy = - torch.logsumexp(val_mlp_logits, dim=-1)
        val_mlp_reg = torch.mean(F.relu(val_mlp_energy[:len(val_energy)] - val_mlp) ** 2) + torch.mean(F.relu(syn_energy - val_mlp_energy[len(val_energy):]) ** 2)

        val_loss = val_loss_id + weights[0] * val_reg_loss + weights[1] * val_mlp + weights[2] * val_mlp_reg
        valid_loss = val_loss.cpu().detach().numpy()

        if (epoch >= 5) & (valid_loss < best_val_loss):
            if np.isinf(best_val_loss) or (best_val_loss - valid_loss < 5): 
                best_val_loss = valid_loss

    with torch.no_grad():
        logits, _ = model(ind_data, device)

    ood_results = []
    for test in test_data:
        with torch.no_grad():
            logits_ood, _ = model(test, device)
        energy_id, energy_ood = get_energy(model, ind_data, test, logits, logits_ood, args, pos = True, prop = prop_gcn, device = device)
        te_index = ind_data.splits['test']

        if use_detector:
            input_for_lr = np.concatenate((energy_id, energy_ood), -1)
            mlp_pred_logits, _ = detector(torch.Tensor(input_for_lr).to(device).view(-1, 1))

            mlp_id_logits = mlp_pred_logits[:len(energy_id)]
            mlp_logits_ood = mlp_pred_logits[len(energy_id):]

            neg_energy_id, neg_energy_ood= get_energy(model, ind_data, test, mlp_id_logits, mlp_logits_ood, args, pos = False, prop = prop_mlp, device = device)
        neg_energy_ood = neg_energy_ood[test.node_idx]

        res = get_measures(neg_energy_id[te_index], neg_energy_ood)
        ood_results.append(res)

    if verbose:
        ood_scores = np.round(np.array(ood_results[1:]).mean(axis = 0)*100, 2)
        test_acc = np.round(100 * test_id, 2)
        print("AUPR:", ood_scores[0], "AUROC:", ood_scores[1], "FPR95:", ood_scores[2], "ID_acc:", test_acc)

    return valid_id, test_id, valid_loss, ood_results, np.array(ood_results[1:]).mean(axis = 0)*100, best_val_loss

