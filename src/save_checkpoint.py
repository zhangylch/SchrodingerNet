import os
import time
import pickle
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
from dataclasses import asdict, is_dataclass

def save_checkpoint(step, params, opt_state, config, ckpt_dir, max_to_keep=5):
    """
    Save checkpoint and automatically prune old ones, keeping only the most recent max_to_keep.
    """
    # 1. Prepare data (All Ranks execute this)
    state_dict = {
        "step": step,
        "params": jax.device_get(params),
        "opt_state": jax.device_get(opt_state),
        "config": jax.device_get(asdict(config))
    }

    # 2. Only Rank 0 writes the file and cleans up
    if jax.process_index() == 0:
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # --- A. Save new file ---
        save_path = os.path.join(ckpt_dir, f"checkpoint_{step}.pkl")
        temp_path = save_path + ".tmp"
        
        try:
            with open(temp_path, "wb") as f:
                pickle.dump(state_dict, f)
            os.rename(temp_path, save_path)
            print(f"[Rank 0] Checkpoint saved: {save_path}")
        except Exception as e:
            print(f"[Rank 0] Failed to save checkpoint: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return # Skip pruning if save failed

        # --- B. Prune old files (Keep Max N) ---
        try:
            all_files = [
                f for f in os.listdir(ckpt_dir) 
                if f.startswith("checkpoint_") and f.endswith(".pkl")
            ]
            
            # Sort by step (ascending)
            all_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
            
            if len(all_files) > max_to_keep:
                files_to_remove = all_files[:-max_to_keep]
                
                for f_name in files_to_remove:
                    f_path = os.path.join(ckpt_dir, f_name)
                    os.remove(f_path)
                    print(f"[Rank 0] Pruned old checkpoint: {f_name}")
                    
        except Exception as e:
            print(f"[Rank 0] Failed to prune old checkpoints: {e}")


def restore_checkpoint(ckpt_dir, params_sharding=None, opt_state_sharding=None, target_dtype=np.float32):
    """
    Load checkpoint (Multi-host safe & dynamic dtype casting):
    1. Rank 0 finds the latest .pkl and broadcasts the step to all nodes.
    2. Wait for NFS sync to prevent dirty reads.
    3. Cast floating arrays to target_dtype (e.g., float32) before device_put.
    """
    # ==========================================
    # Step 1: Rank 0 finds the latest checkpoint step
    # ==========================================
    latest_step = -1
    if jax.process_index() == 0:
        if os.path.exists(ckpt_dir):
            files = [f for f in os.listdir(ckpt_dir) if f.startswith("checkpoint_") and f.endswith(".pkl")]
            if files:
                files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
                latest_step = int(files[-1].split('_')[1].split('.')[0])
    
    # Broadcast the step to prevent SPMD divergence
    step_array = jnp.array(latest_step, dtype=jnp.int32)
    step_array = multihost_utils.broadcast_one_to_all(step_array)
    latest_step = int(step_array.item())

    if latest_step == -1:
        if jax.process_index() == 0:
            print(f"No checkpoint files found in {ckpt_dir}. Starting from scratch.")
        return None

    load_path = os.path.join(ckpt_dir, f"checkpoint_{latest_step}.pkl")
    if jax.process_index() == 0:
        print(f"Loading checkpoint from: {load_path}")

    # ==========================================
    # Step 2: Read file with NFS timeout handling
    # ==========================================
    timeout_seconds = 60
    start_time = time.time()
    while not os.path.exists(load_path):
        if time.time() - start_time > timeout_seconds:
            raise FileNotFoundError(f"[Rank {jax.process_index()}] Timeout waiting for {load_path} to sync on filesystem.")
        time.sleep(1)

    try:
        with open(load_path, "rb") as f:
            state_dict = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"[Rank {jax.process_index()}] Failed to load checkpoint: {e}")

    # ==========================================
    # Step 3: Cast dtypes in CPU memory
    # ==========================================
    if target_dtype is not None:
        def cast_to_dtype(leaf):
            if isinstance(leaf, np.ndarray) and np.issubdtype(leaf.dtype, np.floating):
                return leaf.astype(target_dtype)
            return leaf

        state_dict["params"] = jax.tree_util.tree_map(cast_to_dtype, state_dict["params"])
        state_dict["opt_state"] = jax.tree_util.tree_map(cast_to_dtype, state_dict["opt_state"])
        
        if jax.process_index() == 0:
            print(f"Casted checkpoint floating-point parameters to {target_dtype}")

    # ==========================================
    # Step 4: Push data to GPU
    # ==========================================
    if not params_sharding:
        params = state_dict["params"]
        opt_state = state_dict["opt_state"]
    else:
        params = jax.device_put(state_dict["params"], params_sharding)
        opt_state = state_dict["opt_state"]
    
    restored_step = state_dict["step"]
    restored_config = state_dict["config"]

    # Older checkpoints store pair metadata without the diagonal.
    npar = restored_config["npar"]
    for name in ("expand_indices", "cusp"):
        values = np.asarray(restored_config[name])
        if values.shape != (npar, npar):
            if values.size != npar*(npar-1):
                raise ValueError(f"Unexpected checkpoint {name} shape: {values.shape}")
            pair_values = np.zeros((npar, npar), dtype=values.dtype)
            pair_values[~np.eye(npar, dtype=bool)] = values.reshape(-1)
            restored_config[name] = pair_values

    if jax.process_index() == 0:
        print(f"Successfully restored state at step {restored_step}")

    return restored_step, params, opt_state, restored_config
