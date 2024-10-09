import glob
import os
import model
import preprocess
import torch
import wandb
import logging
import argparse
import monai as mn
import numpy as np
from contextlib import nullcontext
from tqdm import tqdm
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

logging.getLogger("monai").setLevel(logging.ERROR)
import warnings

warnings.filterwarnings(
    "ignore",
    ".*pixdim*.",
)
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

def add_bg(x):
    return torch.cat([1-x.sum(dim=0, keepdim=True), x], dim=0)

def get_loaders(
    batch_size=1,
    device="cpu",
    lowres=False,
    ptch=128,
):

    train_label_list = list(
        np.loadtxt("/home/lchalcroft/git/lab-vae/atlas_train.txt", dtype=str)
    )
    val_label_list = list(
        np.loadtxt("/home/lchalcroft/git/lab-vae/atlas_val.txt", dtype=str)
    )

    train_dict = [
        {
            "seg": f.replace("1mm_sub", "sub"),
            "label": f.replace("_label-L_desc-T1lesion_mask", "_T1w").replace(
                "1mm_sub", "sub"
            ),
        }
        for f in train_label_list
    ]
    val_dict = [
        {
            "seg": f.replace("1mm_sub", "sub"),
            "image": f.replace("_label-L_desc-T1lesion_mask", "_T1w").replace(
                "1mm_sub", "sub"
            ),
        }
        for f in val_label_list
    ]

    print(f"train_dict: {len(train_dict)}, val_dict: {len(val_dict)}")
    print(f"train_dict[0]: {train_dict[0]}")

    train_transform = mn.transforms.Compose(
        transforms=[
            mn.transforms.LoadImageD(
                keys=["image", "seg"], image_only=True, allow_missing_keys=True
            ),
            mn.transforms.EnsureChannelFirstD(
                keys=["image", "seg"], allow_missing_keys=True
            ),
            mn.transforms.LambdaD(keys="seg", func=add_bg),
            mn.transforms.OrientationD(
                keys=["image", "seg"], axcodes="RAS", allow_missing_keys=True
            ),
            mn.transforms.SpacingD(
                keys=["image", "seg"],
                pixdim=1 if not lowres else 2,
                allow_missing_keys=True,
            ),
            mn.transforms.ResizeWithPadOrCropD(
                keys=["image", "seg"],
                spatial_size=(256, 256, 256) if not lowres else (128, 128, 128),
                allow_missing_keys=True,
            ),
            mn.transforms.ToTensorD(
                dtype=float,
                keys=["image", "seg"],
                allow_missing_keys=True,
            ),
            mn.transforms.LambdaD(
                keys=["image", "seg"],
                func=mn.transforms.SignalFillEmpty(),
                allow_missing_keys=True,
            ),
            mn.transforms.Rand2DElasticD(
                keys=["image", "seg"],
                prob=1, spacing=40, magnitude_range=(0, 2),
                rotate_range=20, shear_range=0.5, translate_range=10, scale_range=0.1,
                mode=["bilinear", "nearest"], padding_mode="zeros"),
            mn.transforms.LambdaD(
                keys=["image", "seg"],
                func=mn.transforms.SignalFillEmpty(),
                allow_missing_keys=True,
            ),
            mn.transforms.RandBiasFieldD(keys="image", prob=0.8),
            mn.transforms.RandAxisFlipd(
                keys=["image", "seg"],
                prob=0.8,
                allow_missing_keys=True,
            ),
            mn.transforms.RandAxisFlipd(
                keys=["image", "seg"],
                prob=0.8,
                allow_missing_keys=True,
            ),
            mn.transforms.NormalizeIntensityD(
                keys="image", nonzero=False, channel_wise=True
            ),
            mn.transforms.RandGaussianNoiseD(keys="image", prob=0.8),
            mn.transforms.RandSpatialCropD(
                keys=["image", "seg"],
                roi_size=(ptch, ptch),
                random_size=False,
                allow_missing_keys=True,
            ),
            mn.transforms.ResizeD(
                keys=["image", "seg"],
                spatial_size=(ptch, ptch),
                allow_missing_keys=True,
            ),
            mn.transforms.ThresholdIntensityD(
                keys=["seg"],
                threshold=0.5,
                above=True,
                cval=0,
                allow_missing_keys=True,
            ),
            mn.transforms.ToTensorD(keys=["image", "seg"], dtype=torch.float32),
        ]
    )

    train_data = mn.data.Dataset(train_dict, transform=train_transform)
    val_data = mn.data.Dataset(val_dict, transform=train_transform)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        sampler=None,
        batch_sampler=None,
        num_workers=24,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=24,
    )

    return train_loader, val_loader


def compute_dice(y_pred, y, eps=1e-8):
    y_pred = torch.flatten(y_pred)
    y = torch.flatten(y)
    y = y.float()
    intersect = (y_pred * y).sum(-1)
    denominator = (y_pred * y_pred).sum(-1) + (y * y).sum(-1)
    return 2 * (intersect / denominator.clamp(min=eps))

def run_model(args, device, train_loader, train_transform):
    
    if args.net == "cnn":
        model = model.CNNUNet(
            spatial_dims=2, 
            in_channels=1,
            out_channels=2,
            features=(32, 64, 128, 256, 512, 768),
            act="GELU", 
            norm="instance", 
            bias=True, 
            dropout=0.2,
            upsample="deconv",
        ).to(device)
    elif args.net == "vit":
        model = model.ViTUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=2,
            img_size=(96 if args.lowres else 192),
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            dropout_rate=0.2,
            qkv_bias=True,
            save_attn=False,
        ).to(device)

    if args.resume or args.resume_best:
        ckpts = glob.glob(
            os.path.join(
                args.logdir,
                args.name,
                "checkpoint.pt" if args.resume else "checkpoint_best.pt",
            )
        )
        if len(ckpts) == 0:
            args.resume = False
            print("\nNo checkpoints found. Beginning from epoch #0")
        else:
            checkpoint = torch.load(ckpts[0], map_location=device)
            print(
                "\nResuming from epoch #{} with WandB ID {}".format(
                    checkpoint["epoch"], checkpoint["wandb"]
                )
            )
    print()

    wandb.init(
        project="contrast-squared",
        entity="atlas-ploras",
        save_code=True,
        name=args.name,
        settings=wandb.Settings(start_method="fork"),
        resume="must" if args.resume else None,
        id=checkpoint["wandb"] if args.resume or args.resume_best else None,
    )
    if not args.resume and not args.resume_best:
        wandb.config.update(args)
    wandb.watch(model)

    crit = mn.losses.CEDic

    class WandBID:
        def __init__(self, wandb_id):
            self.wandb_id = wandb_id

        def state_dict(self):
            return self.wandb_id

    class Epoch:
        def __init__(self, epoch):
            self.epoch = epoch

        def state_dict(self):
            return self.epoch

    class Metric:
        def __init__(self, metric):
            self.metric = metric

        def state_dict(self):
            return self.metric

    # Load backbone weights if provided
    # Any layers loaded from the backbone will be frozen during training
    cnt_frozen = 0
    cnt_trainable = 0
    if args.backbone_weights is not None:
        checkpoint = torch.load(args.backbone_weights, map_location=device)
        print(f"\nLoading encoder weights from {args.backbone_weights}")
        for name, param in model.named_parameters():
            if name in checkpoint:
                param.data = checkpoint["encoder"][name]
                param.requires_grad = False
                cnt_frozen += 1
            else:
                param.requires_grad = True
                cnt_trainable += 1
        print(f"Frozen layers: {cnt_frozen}, trainable layers: {cnt_trainable}")
    else:
        print("No backbone weights provided, all layers will be trainable.")

    params = list(model.parameters())
    try:
        opt = torch.optim.AdamW(params, args.lr, fused=torch.cuda.is_available())
    except:
        opt = torch.optim.AdamW(params, args.lr, fused=False)
    # Try to load most recent weight
    if args.resume or args.resume_best:
        model.load_state_dict(
            checkpoint["model"], strict=False
        )
        opt.load_state_dict(checkpoint["opt"])
        start_epoch = checkpoint["epoch"] + 1
        metric_best = checkpoint["metric"]
        # correct scheduler in cases where max epochs has changed
        def lambda1(epoch):
            return (1 - (epoch + start_epoch) / args.epochs) ** 0.9

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=[lambda1])
    else:
        start_epoch = 0
        metric_best = 0

        def lambda1(epoch):
            return (1 - (epoch) / args.epochs) ** 0.9

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=[lambda1])

    os.makedirs(os.path.join(args.logdir, args.name), exist_ok=True)

    train_iter = None
    for epoch in range(start_epoch, args.epochs):
        if args.debug:
            saver1 = mn.transforms.SaveImage(
                output_dir=os.path.join(args.logdir, args.name, "debug-train"),
                output_postfix="img",
                separate_folder=False,
                print_log=False,
            )
            saver2 = mn.transforms.SaveImage(
                output_dir=os.path.join(args.logdir, args.name, "debug-train"),
                output_postfix="seg",
                separate_folder=False,
                print_log=False,
            )
        model.train()
        epoch_loss = 0
        step_deficit = -1e-7
        if args.amp:
            ctx = torch.autocast("cuda" if torch.cuda.is_available() else "cpu")
            scaler = torch.cuda.amp.GradScaler()
        else:
            ctx = nullcontext()
        progress_bar = tqdm(range(args.epoch_length), total=args.epoch_length, ncols=90)
        progress_bar.set_description(f"Epoch {epoch}")
        if train_iter is None:
            train_iter = iter(train_loader)

        for step in progress_bar:
            try:
                batch = next(train_iter)
            except:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            img = batch["image"].to(device)
            seg = batch["seg"].to(device)
            opt.zero_grad(set_to_none=True)

            if args.debug and step < 5:
                saver1(torch.Tensor(img[0].cpu().float()))
                saver2(torch.Tensor(seg[0].cpu().float()))

            with ctx:
                logits = model(img)
                loss = crit(logits, seg)

            if type(loss) == float or loss.isnan().sum() != 0:
                print("NaN found in loss!")
                step_deficit += 1
            else:
                if args.amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
                    opt.step()

                epoch_loss += loss.item()

                wandb.log({"train/loss": loss.item()})

            progress_bar.set_postfix(
                {"loss": epoch_loss / (step + 1 - step_deficit)}
            )
        wandb.log({"train/lr": opt.param_groups[0]["lr"]})
        lr_scheduler.step()

        # Upload some sample pairs to wandb
        img_list = []
        seg_list = []
        for i in range(16):
            img_list.append(img[i])
            seg_list.append(seg[i])
        grid_image1 = make_grid(
                      img_list,
                      nrow=int(4),
                      padding=5,
                      normalize=True,
                      scale_each=True,
                  )
        grid_image2 = make_grid(
                      seg_list,
                      nrow=int(4),
                      padding=5,
                      normalize=True,
                      scale_each=True,
                  )
        print(grid_image1.shape)
        wandb.log(
              {
                  "examples": [
                      wandb.Image(
                          grid_image1[0].cpu().numpy(), caption="Images"
                      ),
                      wandb.Image(
                          grid_image2[0].cpu().numpy(), caption="Segmentations"
                      ),
                  ]
              }
          )


        if epoch_loss < metric_best:
            metric_best = epoch_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "lr": lr_scheduler.state_dict(),
                    "wandb": WandBID(wandb.run.id).state_dict(),
                    "epoch": Epoch(epoch).state_dict(),
                    "metric": Metric(metric_best).state_dict(),
                },
                os.path.join(args.logdir, args.name, "checkpoint_best.pt"),
            )
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "lr": lr_scheduler.state_dict(),
                "wandb": WandBID(wandb.run.id).state_dict(),
                "epoch": Epoch(epoch).state_dict(),
                "metric": Metric(metric_best).state_dict(),
            },
            os.path.join(args.logdir, args.name, "checkpoint.pt"),
        )


def set_up():
    parser = argparse.ArgumentParser(argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--name", type=str, help="Name of WandB run.")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs for training.")
    parser.add_argument("--epoch_length", type=int, default=200, help="Number of iterations per epoch.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--val_interval", type=int, default=2, help="Validation interval.")
    parser.add_argument("--batch_size", type=int, default=512, help="Number of subjects to use per batch.")
    parser.add_argument(
        "--loss",
        type=str,
        help="Loss function to use. Options: [simclr]",
        choices=["simclr", "barlow"],
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Data setup to use. Options: [mprage, bloch, bloch-paired]",
        choices=["mprage", "bloch", "bloch-paired"],
    )
    parser.add_argument(
        "--net", 
        type=str, 
        help="Encoder network to use. Options: [cnn, vit]. Defaults to cnn.", 
        choices=["cnn", "vit"],
        default="cnn"
    )
    parser.add_argument("--amp", default=False, action="store_true")
    parser.add_argument("--logdir", type=str, default="./", help="Path to saved outputs")
    parser.add_argument("--resume", default=False, action="store_true")
    parser.add_argument("--resume_best", default=False, action="store_true")
    parser.add_argument("--device", type=str, default=None, help="Device to use. If not specified then will check for CUDA.")
    parser.add_argument("--lowres", default=False, action="store_true", help="Train with 2mm resolution images.")
    parser.add_argument("--debug", default=False, action="store_true", help="Save sample images before training.")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.logdir, args.name), exist_ok=True)
    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print("Using device:", device)
    print()
    # Additional Info when using cuda
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
        print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")
    debug_loader, _ = preprocess.get_mprage_loader(batch_size=1, device=device, lowres=args.lowres)
    train_loader, train_transform = preprocess.get_mprage_loader(batch_size=args.batch_size, device=device, lowres=args.lowres)
    
    if args.debug:
        saver1 = mn.transforms.SaveImage(
            output_dir=os.path.join(args.logdir, args.name, "debug"),
            output_postfix="img1",
            separate_folder=False,
            print_log=False,
        )
        saver2 = mn.transforms.SaveImage(
            output_dir=os.path.join(args.logdir, args.name, "debug"),
            output_postfix="img2",
            separate_folder=False,
            print_log=False,
        )
        for i, batch in enumerate(debug_loader):
            if i > 5:
                break
            else:
                print(
                    "Image #1: ",
                    batch["image1"].shape,
                    "min={}".format(batch["image1"].min()),
                    "max={}".format(batch["image1"].max()),
                )
                saver1(
                    torch.Tensor(batch["image1"][0].cpu().float()),
                )
                print(
                    "Image #2: ",
                    batch["image2"].shape,
                    "min={}".format(batch["image2"].min()),
                    "max={}".format(batch["image2"].max()),
                )
                saver2(
                    torch.Tensor(batch["image2"][0].cpu().float())
                )

    return args, device, train_loader, train_transform


def main():
    args, device, train_loader, train_transform = set_up()
    run_model(args, device, train_loader, train_transform)


if __name__ == "__main__":
    main()
