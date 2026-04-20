import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
import torchvision.transforms.v2.functional as TF
from torchvision import tv_tensors
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets import load_dataset
import matplotlib.pyplot as plt
import pandas as pd
import random 


class NYUDataset(Dataset):
    def __init__(self, hf_dataset, is_train=True):
        self.dataset = hf_dataset
        self.is_train = is_train
        if self.is_train:
            self.spatial_ops = v2.Compose([
                v2.Resize(280, antialias=True),
                v2.RandomCrop(256),
                v2.RandomHorizontalFlip(p=0.5)
            ])
        else:
            self.spatial_ops = v2.Compose([
                v2.Resize(256, antialias=True),
                v2.CenterCrop(256)
            ])
        self.normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        img = tv_tensors.Image(TF.to_image(item["image"].convert("RGB")))
        depth = tv_tensors.Mask(torch.from_numpy(np.array(item["depth_map"])).unsqueeze(0).float() / 10.0)

        img, depth = self.spatial_ops(img, depth)
        img = TF.to_dtype(img, torch.float32, scale=True)
        img = self.normalize(img)
        
        return img, depth

class NYUDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=4):
        super().__init__()
        self.batch_size = batch_size
        self.path_dati = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
    
    def setup(self, stage=None):
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        dataset = load_dataset("sayakpaul/nyu_depth_v2", trust_remote_code=True, cache_dir=self.path_dati)
        self.train_ds = NYUDataset(dataset["train"], is_train=True)
        self.val_ds = NYUDataset(dataset["validation"], is_train=False)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True)

def load_teacher(model_type, device):
    print(f"Loading Teacher: {model_type}...")
    teacher = torch.hub.load("intel-isl/MiDaS", model_type)
    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class TeacherTask(pl.LightningModule):
    def __init__(self, model, lr=1e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.val_losses = []

    def forward(self, x):
        inv_depth = self.model(x).unsqueeze(1)
        inv_depth = (inv_depth - inv_depth.min()) / (inv_depth.max() - inv_depth.min() + 1e-6)
        return 1.0 / (inv_depth + 0.1)

    def training_step(self, batch, batch_idx):
        loss, _ = self._shared_step(batch)
        self.log("teacher/train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        if preds.shape[-2:] != y.shape[-2:]:
            preds = nn.functional.interpolate(preds.unsqueeze(1), size=y.shape[-2:], mode="bicubic", align_corners=False)
        else:
            preds = preds.unsqueeze(1)
        
        loss = self.criterion(preds, y)
        mae = nn.functional.l1_loss(preds, y)
        self.log("teacher/val_loss", loss)
        self.log("teacher/val_mae", mae, prog_bar=True)
        self.val_losses.append(loss.item())
        return loss


    def _shared_step(self, batch):
        x, y = batch
        preds = self(x) 
        
        if preds.shape[-2:] != y.shape[-2:]:
            preds = nn.functional.interpolate(preds, size=y.shape[-2:], mode="bicubic", align_corners=False)
        
        loss = nn.functional.smooth_l1_loss(preds, y)
        return loss, preds

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "teacher/val_loss"}
        }

# Funzioni di visualizzazione
def verify_batch(images, depths, n=4):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    fig, axes = plt.subplots(n, 2, figsize=(10, n * 3))
    for i in range(n):
        img = images[i].cpu() * std + mean
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()
        d = depths[i].cpu().squeeze().numpy() * 10.0
        axes[i, 0].imshow(img)
        axes[i, 0].axis("off")
        im_d = axes[i, 1].imshow(d, cmap="plasma")
        axes[i, 1].axis("off")
        plt.colorbar(im_d, ax=axes[i, 1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

def verify_predictions(images, depths, preds, n=3):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    fig, axes = plt.subplots(n, 3, figsize=(15, n * 4))

    for i in range(n):
        img = (images[i].cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Original RGB")
        axes[i, 0].axis("off")
        
        p = preds[i].cpu().squeeze().numpy()
        im_p = axes[i, 1].imshow(p, cmap="plasma_r")
        axes[i, 1].set_title("Teacher Prediction")
        axes[i, 1].axis("off")
        plt.colorbar(im_p, ax=axes[i, 1], fraction=0.046, pad=0.04)
        
        d = depths[i].cpu().squeeze().numpy() * 10.0
        im_d = axes[i, 2].imshow(d, cmap="plasma_r")
        axes[i, 2].set_title("Ground Truth (m)")
        axes[i, 2].axis("off")
        plt.colorbar(im_d, ax=axes[i, 2], fraction=0.046, pad=0.04)
        
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model_type", type=str, default="MiDaS_small", choices=["MiDaS_small", "DPT_Large", "DPT_Hybrid"])
    parser.add_argument("--epochs_fine_tuning", type=int, default=3)
    args = parser.parse_args()

    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
    os.makedirs(models_dir, exist_ok=True)
    torch.hub.set_dir(models_dir)

    os.environ["HF_DATASETS_OFFLINE"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dm = NYUDataModule(batch_size=args.batch_size)
    dm.setup()

    teacher = load_teacher(args.model_type, device)
    for p in teacher.parameters(): p.requires_grad = True
    
    teacher_task = TeacherTask(teacher, lr=1e-5) # LR basso per calibrazione
    print(f"Teacher Parameters: {count_parameters(teacher):,}")

    checkpoint_callback = ModelCheckpoint(
        dirpath=models_dir,
        filename="teacher-finetuned-{epoch:02d}-{val_mae:.2f}",
        save_top_k=1,
        monitor="teacher/val_mae",
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs_fine_tuning,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        accumulate_grad_batches=4,
        gradient_clip_val=0.5,    
        default_root_dir=models_dir
    )
    
    trainer.fit(teacher_task, dm)

    val_results = trainer.validate(teacher_task, dm)
    res_df = pd.DataFrame({
        'Metric': ['Teacher Val MSE', 'Teacher Val MAE'],
        'Value': [f"{val_results[0]['teacher/val_loss']:.4f}", f"{val_results[0]['teacher/val_mae']:.4f}"]
    })
    print("\n" + res_df.to_string(index=False))

    if args.verbose:
        print("\nVisualizzazione confronto post-finetuning: RGB | Teacher (Metri) | GT (Metri)")
        teacher_task.to(device).eval()
        
        val_loader = dm.val_dataloader()
        num_batches = len(val_loader)
        
        random_batch_idx = random.randint(0, num_batches - 1)
        
        for i, batch in enumerate(val_loader):
            if i == random_batch_idx:
                imgs, labels = batch
                break
        
        with torch.no_grad():
            preds = teacher_task(imgs.to(device))
            if preds.shape[-2:] != labels.shape[-2:]:
                preds = torch.nn.functional.interpolate(preds, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        
        verify_predictions(imgs, labels, preds, n=min(4, args.batch_size))

if __name__ == "__main__":
    main()