import time 
import heapq 
import torch 
import torch.nn as nn 
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders 
import locale
locale.getpreferredencoding = lambda: "UTF-8"

from .ablate import AblateGPT 

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    #attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    #return inps, outs, attention_mask, position_ids 
    return inps, outs, position_ids 

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.layers 

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0




def prune_cim(args, model, tokenizer, device=torch.device("cuda:0"), m=None, n=None, p=None, q=None):
    
    # pruning_method = 'column-wise'
    #pruning_method = 'row-wise'
    #pruning_method = 'ou-base'

    pruning_method = args.grouping
    m = args.m
    n = args.n
    p = args.p
    q = args.q
    

    print ("M: ", m)
    print ("N: ", n)
    print ("P: ", p)
    print ("pruning_method: ", pruning_method)


    use_cache = model.config.use_cache 
    model.config.use_cache = False  
    # model.config.use_cache = use_cache 

    print("loading calibdation data")
    dataloader, _ = get_loaders(dataset=args.dataset,nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        #inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
        inps, outs, position_ids = prepare_calibration_input(model, dataloader, device)


    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            #inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)
            inps, outs, position_ids = inps.to(dev), outs.to(dev), position_ids.to(dev)


        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                #print ("wrapped_layers[name]: ", name)
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                # seq_len = inps[j].shape[0]
                # position_ids = torch.arange(seq_len, dtype=torch.long, device=inps[j].device).unsqueeze(0)
                # position_ids = torch.arange(inps[j].shape[0], dtype=torch.long, device=inps[j].device).unsqueeze(0)
                # outs[j] = layer(inps[j].unsqueeze(0))[0]
                # outs[j] = layer(inps[j].unsqueeze(0), position_ids=position_ids, past_key_values=None)[0]
                # outs[j] = layer(inps[j].unsqueeze(0), position_ids=position_ids, attention_mask=None)[0]
                # outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                outs[j] = layer(inps[j].unsqueeze(0), position_ids=position_ids)[0]
        
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            scaler_row = wrapped_layers[name].scaler_row

            # Check if there are NaN or Inf values in scaler_row
            if torch.isnan(scaler_row).any() or torch.isinf(scaler_row).any():
                #print(f"Warning: NaN or Inf detected in layer {name}'s scaler_row.")
                nan_count = torch.isnan(scaler_row).sum().item()
                inf_count = torch.isinf(scaler_row).sum().item()
                #print(f"NaN count: {nan_count}, Inf count: {inf_count}")

                # Replace NaN and Inf values safely
                scaler_row = torch.nan_to_num(scaler_row, nan=0.0, posinf=1e6, neginf=-1e6)
                

            # Add epsilon to avoid sqrt(0) or negative numbers
            eps = 1e-8
            scaler_row = torch.clamp(scaler_row, min=0) + eps

            H = wrapped_layers[name].H  # Hessian H available in the WrappedGPT object

            alpha = 0.0  # Weight for how much to consider scaler_row vs. Hessian, tune this as needed

            print ("Alpha: ", alpha)


            # Continue with the computation, assuming NaNs have been handled
            #W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(scaler_row.reshape((1, -1)))


            #print(f"Weight matrix shape: {subset[name].weight.data.shape}")  # Shape of the weight matrix
            #print(f"Scaler_row shape: {scaler_row.shape}")  # Shape of the scaler_row
            #print(f"Scaler_row reshaped shape: {scaler_row.reshape((1, -1)).shape}")  # Shape of the reshaped scaler_row
            #print(f"H shape: {H.shape}")  # Shape of the H
            #print(f"H_mean shape: {H.mean(dim=1, keepdim=True).shape}")  # Shape of the mean of H
            #print(f"H.diagonal() shape: {H.diagonal().shape}")  # Shape of the mean of H
            #print(f"H.diagonal() reshaped shape: {H.diagonal().reshape(1, -1).shape}")  # Shape of the mean of H

            #W_metric_org = torch.abs(subset[name].weight.data) * torch.sqrt(scaler_row.reshape((1, -1))
            #print(f"W_metric_org shape: {W_metric_org.shape}")  # Shape of the W_metric_org



            

            ## Option 1: Hessian in experiment
            #W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(scaler_row.reshape((1, -1)) + alpha * H.diagonal().reshape((1, -1)))
            

            ## Option 2: Hessian out experiment
            #W_metric = torch.abs(subset[name].weight.data) *(torch.sqrt(scaler_row.reshape((1, -1))) + alpha * H.diagonal().reshape((1, -1)))
            #W_metric = torch.abs(torch.abs(subset[name].weight.data) *(torch.sqrt(scaler_row.reshape((1, -1))) + alpha * H.diagonal().reshape((1, -1))))





            ## Option 3: (Normalize H) -  Updated W_metric calculation using adaptive scaling of the Hessian component
            #epsilon = 1e-8  # Small value to avoid division by zero
            #hessian_diag_normalized = H.diagonal().reshape((1, -1)) / (H.diagonal().std() + epsilon)  # Normalize Hessian diagonal
            #W_metric = torch.abs(subset[name].weight.data) * (torch.sqrt(scaler_row.reshape((1, -1))) + alpha * hessian_diag_normalized)



            ## Option 4: Normalize both the activation-based and Hessian-based components
            epsilon = 1e-8  # Small constant to avoid division by zero
            # Normalize activation-based component (sqrt of scaler_row)
            normalized_activation = torch.sqrt(scaler_row.reshape((1, -1))) / (torch.max(torch.sqrt(scaler_row)) + epsilon)
            # Normalize Hessian diagonal component
            normalized_hessian = H.diagonal().reshape((1, -1)) / (torch.max(H.diagonal()) + epsilon)
            # Updated W_metric calculation using normalized components
            W_metric = torch.abs(subset[name].weight.data) * (normalized_activation + alpha * normalized_hessian)

            
            num_rows, num_cols = W_metric.shape
            #print(f"Original Matrix:\n{W_metric}")
            #print(f"num_rows: {num_rows}, num_cols: {num_cols}")
            
            # Ensure matrix dimensions are divisible by chunk size
            assert num_rows % m == 0 and num_cols % n == 0, "Matrix dimensions must be divisible by chunk size"

            # Unfold the matrix into chunks (blocks of size m x n)
            W_metric_chunks = W_metric.unfold(0, m, m).unfold(1, n, n).contiguous().view(num_rows // m, num_cols // n, m, n)

            # Initialize a mask
            #W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
            W_mask = torch.zeros_like(W_metric, dtype=torch.bool).to(device)

         

            if pruning_method == 'column-wise':
                # Vectorized sum across rows to get column importance
                column_importance = W_metric_chunks.sum(dim=-2)  # shape: (num_blocks_row, num_blocks_col, n)
                num_columns_to_prune = int(n * args.sparsity_ratio)
                print ("Sparsity Ratio: ", args.sparsity_ratio)
                
                # Find the least important columns for each block
                _, prune_indices = torch.topk(column_importance, num_columns_to_prune, dim=-1, largest=False)
                
                # Ensure prune_indices is on the same device
                prune_indices = prune_indices.to(device)
                
                # Apply mask in vectorized form for column pruning
                block_rows = torch.arange(W_metric_chunks.shape[0])[:, None].to(device)  # Move to device
                block_cols = torch.arange(W_metric_chunks.shape[1])[:, None].to(device)  # Move to device
                
                for block_row in block_rows:
                    for block_col in block_cols:
                        W_mask[block_row * m:(block_row + 1) * m, block_col * n + prune_indices[block_row, block_col]] = True


            elif pruning_method == 'row-wise':
                # Vectorized sum across rows to get column importance
                row_importance = W_metric_chunks.sum(dim=-1)  # shape: (num_blocks_row, num_blocks_col, n)
                num_rows_to_prune = int(m * args.sparsity_ratio)
                print("Sparsity Ratio: ", args.sparsity_ratio)

                # Find the least important columns for each block
                _, prune_indices = torch.topk(row_importance, num_rows_to_prune, dim=-1, largest=False)

                # Ensure prune_indices is on the same device
                prune_indices = prune_indices.to(device)

                # Apply mask in vectorized form for column pruning
                block_rows = torch.arange(W_metric_chunks.shape[0])[:, None].to(device)  # Move to device
                block_cols = torch.arange(W_metric_chunks.shape[1])[:, None].to(device)  # Move to device

                for block_row in block_rows:
                    for block_col in block_cols:
                        W_mask[block_row * m:(block_row + 1) * m,
                        block_col * n + prune_indices[block_row, block_col]] = True
                        
            elif pruning_method == 'ou-base':
                print ("Sparsity Ratio: ", args.sparsity_ratio)
                # Process each chunk into subblocks of size p * q
                for block_row in range(W_metric_chunks.shape[0]):
                    for block_col in range(W_metric_chunks.shape[1]):
                        chunk_ = W_metric_chunks[block_row, block_col, :, :]

                        # Unfold chunk into subblocks and calculate subblock importance
                        chunk_subblocks = chunk_.unfold(0, p, p).unfold(1, q, q).contiguous().view(m // p, n // q, p, q)
                        subblock_importances = chunk_subblocks.sum(dim=(2, 3))  # shape: (m//p, n//q)

                        # Prune the least important subblocks
                        num_subblocks_to_prune = int(((m * n) // (p * q)) * args.sparsity_ratio)
                        _, subblock_prune_indices = torch.topk(subblock_importances.flatten(), num_subblocks_to_prune,
                                                               largest=False)

                        # Ensure subblock_prune_indices is on the correct device
                        subblock_prune_indices = subblock_prune_indices.to(device)

                        # Convert 1D indices back to 2D and update the mask
                        subblock_prune_indices = torch.stack([
                            subblock_prune_indices // (n // q),
                            subblock_prune_indices % (n // q)
                        ], dim=1).to(device)  # Move the stacked tensor to device

                        # Efficiently map local subblock indices to global indices and update the mask
                        for subblock_idx in subblock_prune_indices:
                            subblock_row, subblock_col = subblock_idx.tolist()
                            global_row_start = block_row * m + subblock_row * p
                            global_col_start = block_col * n + subblock_col * q

                            # Set corresponding subblock region in the mask to True
                            W_mask[global_row_start:global_row_start + p, global_col_start:global_col_start + q] = True

            
            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                #outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                outs[j] = layer(inps[j].unsqueeze(0), position_ids=position_ids)[0]
        
        inps, outs = outs, inps



    #model.push_to_hub("rakesh2222/model_cim", check_pr=True)
    #tokenizer.push_to_hub("rakesh2222/model_cim",check_pr=True)
    
    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()





def prune_cim2(args, model, tokenizer, device=torch.device("cuda:0"), m=16, n=16, p=None, q=None):
    
    pruning_method = 'column-wise'
    # pruning_method = 'row-wise'
    #pruning_method = 'ou-base'

    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders(dataset=args.dataset,nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        #inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
        inps, outs, position_ids = prepare_calibration_input(model, dataloader, device)


    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            #inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)
            inps, outs, position_ids = inps.to(dev), outs.to(dev), position_ids.to(dev)


        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                #outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                outs[j] = layer(inps[j].unsqueeze(0), position_ids=position_ids)[0]
        
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            scaler_row = wrapped_layers[name].scaler_row

            # Check if there are NaN or Inf values in scaler_row
            if torch.isnan(scaler_row).any() or torch.isinf(scaler_row).any():
                print(f"Warning: NaN or Inf detected in layer {name}'s scaler_row.")
                nan_count = torch.isnan(scaler_row).sum().item()
                inf_count = torch.isinf(scaler_row).sum().item()
                print(f"NaN count: {nan_count}, Inf count: {inf_count}")

                # Replace NaN and Inf values safely
                scaler_row = torch.nan_to_num(scaler_row, nan=0.0, posinf=1e6, neginf=-1e6)
                

            # Add epsilon to avoid sqrt(0) or negative numbers
            eps = 1e-8
            scaler_row = torch.clamp(scaler_row, min=0) + eps

            # Continue with the computation, assuming NaNs have been handled
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(scaler_row.reshape((1, -1)))

            #W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            #W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(torch.ones_like(wrapped_layers[name].scaler_row).reshape((1, -1)))

            
            num_rows, num_cols = W_metric.shape
            #print(f"Original Matrix:\n{W_metric}")
            print(f"num_rows: {num_rows}, num_cols: {num_cols}")
            
            # Ensure matrix dimensions are divisible by chunk size
            assert num_rows % m == 0 and num_cols % n == 0, "Matrix dimensions must be divisible by chunk size"

            # Unfold the matrix into chunks (blocks of size m x n)
            W_metric_chunks = W_metric.unfold(0, m, m).unfold(1, n, n).contiguous().view(num_rows // m, num_cols // n, m, n)

            # Initialize a mask
            #W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
            W_mask = torch.zeros_like(W_metric, dtype=torch.bool).to(device)

         

            if pruning_method == 'column-wise':
                # Vectorized sum across rows to get column importance
                column_importance = W_metric_chunks.sum(dim=-2)  # shape: (num_blocks_row, num_blocks_col, n)
                num_columns_to_prune = int(n * args.sparsity_ratio)
                print ("Sparsity Ratio: ", args.sparsity_ratio)
                
                # Find the least important columns for each block
                _, prune_indices = torch.topk(column_importance, num_columns_to_prune, dim=-1, largest=False)
                
                # Ensure prune_indices is on the same device
                prune_indices = prune_indices.to(device)
                
                # Apply mask in vectorized form for column pruning
                block_rows = torch.arange(W_metric_chunks.shape[0])[:, None].to(device)  # Move to device
                block_cols = torch.arange(W_metric_chunks.shape[1])[:, None].to(device)  # Move to device
                
                for block_row in block_rows:
                    for block_col in block_cols:
                        W_mask[block_row * m:(block_row + 1) * m, block_col * n + prune_indices[block_row, block_col]] = True
                        
            elif pruning_method == 'row-wise':
                # Vectorized sum across columns to get row importance
                row_importance = W_metric_chunks.sum(dim=-1)  # shape: (num_blocks_row, num_blocks_col, m)
                num_rows_to_prune = int(m * args.sparsity_ratio)
                print ("Sparsity Ratio: ", args.sparsity_ratio)
                
                # Find the least important rows for each block
                _, prune_indices = torch.topk(row_importance, num_rows_to_prune, dim=-1, largest=False)
                
                # Ensure prune_indices is on the same device
                prune_indices = prune_indices.to(device)
                
                # Apply mask in vectorized form for row pruning
                block_rows = torch.arange(W_metric_chunks.shape[0])[:, None].to(device)  # Move to device
                block_cols = torch.arange(W_metric_chunks.shape[1])[:, None].to(device)  # Move to device

                for block_row in block_rows:
                    for block_col in block_cols:
                        W_mask[block_row * m + prune_indices[block_row, block_col], block_col * n : block_col * n + n] = True


            elif pruning_method == 'ou-base':
                # Ensure chunk dimensions are divisible by p * q
                assert m % p == 0 and n % q == 0, "Chunk dimensions must be divisible by p * q"

                # Process each chunk into subblocks of size p * q
                for block_row in range(W_metric_chunks.shape[0]):
                    for block_col in range(W_metric_chunks.shape[1]):
                        chunk_ = W_metric_chunks[block_row, block_col, :, :]

                        # Unfold chunk into subblocks and calculate subblock importance
                        chunk_subblocks = chunk_.unfold(0, p, p).unfold(1, q, q).contiguous().view(m // p, n // q, p, q)
                        subblock_importances = chunk_subblocks.sum(dim=(2, 3))  # shape: (m//p, n//q)
                        
                        # Prune the least important subblocks
                        num_subblocks_to_prune = int(((m * n) // (p * q)) * args.sparsity_ratio)
                        _, subblock_prune_indices = torch.topk(subblock_importances.flatten(), num_subblocks_to_prune, largest=False)
                        
                        # Ensure subblock_prune_indices is on the correct device
                        subblock_prune_indices = subblock_prune_indices.to(device)
                        
                        # Convert 1D indices back to 2D and update the mask
                        subblock_prune_indices = torch.stack([
                            subblock_prune_indices // (n // q), 
                            subblock_prune_indices % (n // q)
                        ], dim=1).to(device)  # Move the stacked tensor to device

                        # Efficiently map local subblock indices to global indices and update the mask
                        for subblock_idx in subblock_prune_indices:
                            subblock_row, subblock_col = subblock_idx.tolist()
                            global_row_start = block_row * m + subblock_row * p
                            global_col_start = block_col * n + subblock_col * q

                            # Set corresponding subblock region in the mask to True
                            W_mask[global_row_start:global_row_start + p, global_col_start:global_col_start + q] = True

                    
            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                #outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                outs[j] = layer(inps[j].unsqueeze(0), position_ids=position_ids)[0]
        
        inps, outs = outs, inps

    #model.push_to_hub("rakesh2222/model_cim", check_pr=True)
    #tokenizer.push_to_hub("rakesh2222/model_cim",check_pr=True)
    
    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()



def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders(dataset=args.dataset,nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, position_ids = inps.to(dev), outs.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
         


            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()



@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()



@torch.no_grad()
def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = AblateGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            if args.prune_method == "ablate_wanda_seq":
                prune_mask = gpts[name].get_wanda_mask(args.sparsity_ratio, prune_n, prune_m)
            elif args.prune_method == "ablate_mag_seq":
                prune_mask = gpts[name].get_mag_mask(args.sparsity_ratio, prune_n, prune_m)
            elif "iter" in args.prune_method:
                prune_mask = None 

            gpts[name].fasterprune(args, args.sparsity_ratio, mask=prune_mask, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
