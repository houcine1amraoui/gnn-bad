from src.preprocessing.TimeSeriesDataset import TimeSeriesDataset
from torch.utils.data import DataLoader
import numpy as np
import torch
from tqdm import tqdm
import os
import torch

from src.utils.device import get_device
from src.utils.experiment import load_best_checkpoint
from src.utils.get_folders_utils import get_processed_folder, get_evaluation_results_main_folder

def create_evaluation_dataloaders(config):
    processed_data_folder = get_processed_folder(config)
    
    # load config
    window_size = config["training"]["window_size"]
    batch_size = config["evaluation"]["batch_size"]

    arrays = np.load(f"{processed_data_folder}/arrays.npy")

    train_dataset = TimeSeriesDataset(arrays["train"], window_size)
    val_dataset = TimeSeriesDataset(arrays["val"], window_size)
    actor2_test_dataset = TimeSeriesDataset(arrays["actor2_test"], window_size)
    actor1_test_dataset = TimeSeriesDataset(arrays["actor1_test"], window_size)
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=True)
    actor2_test_loader = DataLoader(actor2_test_dataset, batch_size, shuffle=True)
    actor1_test_loader = DataLoader(actor1_test_dataset, batch_size, shuffle=True)

    data_loaders = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "actor2_test_loader": actor2_test_loader,
        "actor1_test_loader": actor1_test_loader
    }
    return data_loaders

def compute_errors_per_loader(model, dataloader):
    device = get_device()
    model.eval()

    forecast_errors = []
    recon_errors = []

    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x = x.to(device)
            y = y.to(device)

            output = model(x)

            # 🔵 Case 1: MTAD-GAT (dict output)
            if isinstance(output, dict):
                pred = output["pred"]
                recon = output.get("recon", None)
            else:
                # 🔵 Case 2: GDN (tensor output)
                pred = output
                recon = None

            # --- Forecast error ---
            f_err = torch.abs(pred - y)   # (B, k)
            forecast_errors.append(f_err.cpu().numpy())

            # --- Reconstruction error (if exists) ---
            if recon is not None:
                r_err = torch.abs(recon - x)      # (B, n, k)
                r_err_last = r_err[:, -1, :]      # align with prediction
                recon_errors.append(r_err_last.cpu().numpy())

    forecast_errors = np.concatenate(forecast_errors, axis=0)

    if len(recon_errors) > 0:
        recon_errors = np.concatenate(recon_errors, axis=0)
    else:
        recon_errors = None

    return {
        "forecast": forecast_errors,   # shape [T, k]
        "reconstruction": recon_errors # shape [T, k] or None
    }

def compute_errors_all_loaders(model, config):
    data_loaders = create_evaluation_dataloaders(config)

    train_errors = compute_errors_per_loader(model, data_loaders["train_loader"])
    val_errors = compute_errors_per_loader(model, data_loaders["val_loader"])
    actor2_test_errors = compute_errors_per_loader(model, data_loaders["actor2_test_loader"])
    actor1_test_errors = compute_errors_per_loader(model, data_loaders["actor1_test_loader"])

    evaluation_results_main_folder = get_evaluation_results_main_folder(config)
    errors_folder = f"{evaluation_results_main_folder}/errors"
    os.makedirs(errors_folder, exist_ok=True)

    # Check once
    has_recon = train_errors["reconstruction"] is not None

    def save_split(name, errors):
        if has_recon:
            np.savez(
                f"{errors_folder}/{name}.npz",
                forecast=errors["forecast"],
                reconstruction=errors["reconstruction"]
            )
        else:
            np.savez(
                f"{errors_folder}/{name}.npz",
                forecast=errors["forecast"]
            )

    save_split("train", train_errors)
    save_split("val", val_errors)
    save_split("actor2_test", actor2_test_errors)
    save_split("actor1_test", actor1_test_errors)

def compute_errors(config):
    model = load_best_checkpoint(config)
    # Compute errors for all loaders
    compute_errors_all_loaders(model, config)