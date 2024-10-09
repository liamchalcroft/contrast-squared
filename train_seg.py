import glob
import os

# from monai.networks.nets import UNet
from model import UNet
from monai import transforms
from generative.metrics import SSIMMetric
from generative.losses import PerceptualLoss
import torch
from preprocess import get_loaders
from loss import BarronLoss, BaurLoss
from tqdm import tqdm
import monai as mn
import custom_cc
import preprocess_baseline
import preprocess_freesurfer
import preprocess_real_mpms
import wandb
import argparse
import random
from torchinfo import summary
from torch.optim.lr_scheduler import LambdaLR
import hyperlight as hl
from contextlib import nullcontext
from torchvision.utils import make_grid
import logging

logging.getLogger("monai").setLevel(logging.ERROR)
import warnings

warnings.filterwarnings(
    "ignore",
    ".*pixdim*.",
)
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")


def activate(x):
    # apply activation function to match postproc in nitorch qmri
    x[:, 0] = x[:, 0].exp()  # pd = e^f(x)
    x[:, 1] = x[:, 1].exp()  # r1 = e^f(x)
    x[:, 2] = x[:, 2].exp()  # r2 = e^f(x)
    x[:, 3] = x[:, 3].neg().exp().add(1).reciprocal().mul(100)  # mt = 100/(1+e^-f(x))
    return x


class HyperUNet(torch.nn.Module):
    def __init__(self, hyperparams=13, hyper_layers=[16, 64, 128], **kwargs):
        super().__init__()
        mainnet = UNet(**kwargs)
        # print('Main network:')
        # print(summary(mainnet, input_size=(1,1,192,192,192)))
        # print(mainnet)
        # print()
        modules = hl.find_modules_from_patterns(
            mainnet,
            [
                "model.0.conv",
                "model.0.residual",
                "model*.submodule.0.conv",
                "model*.submodule.0.residual",
                "model*.submodule.1.conv",
                "model*.submodule.1.residual",
            ],
        )
        print("Using hypernetwork to parametrise the following modules:")
        print(list(modules.keys()))
        print()
        self.mainnet = hl.hypernetize(mainnet, modules=modules)
        self.hypernet = hl.HyperNet(
            input_shapes={"h": (hyperparams,)},
            output_shapes=self.mainnet.external_shapes(),
            hidden_sizes=hyper_layers,
            # encoding=None
        )
        print("Hypernetwork:")
        print(self.hypernet)
        print()

    def forward(self, main_input, hyper_input):
        parameters = self.hypernet(h=hyper_input)
        with self.mainnet.using_externals(parameters):
            prediction = self.mainnet(main_input)
        return prediction


def run_model(args, device, train_loader, val_loader, train_transform):
    model = HyperUNet(
        hyperparams=13 * args.num_ch,
        hyper_layers=[16, 64, 128],
        spatial_dims=3,
        in_channels=args.num_ch,
        out_channels=4,
        channels=[24, 48, 96, 192, 384],
        strides=[2, 2, 2, 2],
        dropout=0.1,
        kernel_size=3,
        up_kernel_size=3,
        num_res_units=2,
        act="GELU",
        norm="INSTANCE",
        adn_ordering="NDA",
        upsample=args.upsample,
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
        project="synthbloch",
        entity="atlas-ploras",
        save_code=True,
        name=args.name,
        settings=wandb.Settings(start_method="fork"),
        resume="must" if args.resume and not args.reset_wandb else None,
        id=checkpoint["wandb"] if args.resume or args.resume_best and not args.reset_wandb else None,
    )
    if not args.resume and not args.resume_best and not args.reset_wandb:
        wandb.config.update(args)
    wandb.watch(model)

    if args.loss == "l1":
        l1_loss = torch.nn.L1Loss(reduction="none")
    elif args.loss == "l2":
        l1_loss = torch.nn.MSELoss(reduction="none")
    elif args.loss == "huber":
        l1_loss = torch.nn.HuberLoss(reduction="none")
    elif args.loss == "baur":
        l1_loss = BaurLoss(reduction="none")
    elif args.loss == "barron":
        l1_loss = BarronLoss(reduction="none")
    else:
        raise ValueError(
            "Loss '{}' not found. Please use train.py --help to see available options.".format(
                args.loss
            )
        )
    loss_perceptual = PerceptualLoss(
        spatial_dims=3, network_type="medicalnet_resnet50_23datasets", is_fake_3d=False
    ).to(device)
    perceptual_weight = 0.1
    ssim = SSIMMetric(spatial_dims=3, data_range=1.0, reduction="mean", kernel_size=5)

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

    try:
        opt = torch.optim.AdamW(
            model.parameters(), args.lr, fused=torch.cuda.is_available()
        )
    except:
        opt = torch.optim.AdamW(model.parameters(), args.lr)
    # Try to load most recent weight
    if args.resume or args.resume_best:
        model.load_state_dict(
            checkpoint["net"], strict=False
        )  # strict False in case of switch between subpixel and transpose
        opt.load_state_dict(checkpoint["opt"])
        start_epoch = checkpoint["epoch"] + 1
        metric_best = checkpoint["metric"]
        # correct scheduler in cases where max epochs has changed
        def lambda1(epoch):
            return (1 - (epoch + start_epoch) / args.epochs) ** 0.9

        lr_scheduler = LambdaLR(opt, lr_lambda=[lambda1])
    else:
        start_epoch = 0
        metric_best = 0

        def lambda1(epoch):
            return (1 - (epoch) / args.epochs) ** 0.9

        lr_scheduler = LambdaLR(opt, lr_lambda=[lambda1])

    if args.mix_real:
        train_rl_loader, val_rl_loader, train_transform = (
            preprocess_baseline.get_loaders(
                args.batch_size, device, args.lowres, args.num_ch, args.local_paths
            )
        )

        def chunk(indices, size):
            return torch.split(torch.tensor(indices), size)

        class MyBatchSampler(torch.utils.data.Sampler):
            def __init__(self, a_indices, b_indices, batch_size):
                self.a_indices = a_indices
                self.b_indices = b_indices
                self.batch_size = batch_size

            def __iter__(self):
                random.shuffle(self.a_indices)
                random.shuffle(self.b_indices)
                a_batches = chunk(self.a_indices, self.batch_size)
                b_batches = chunk(self.b_indices, self.batch_size)
                all_batches = list(a_batches + b_batches)
                all_batches = [batch.tolist() for batch in all_batches]
                random.shuffle(all_batches)
                return iter(all_batches)

        new_dataset = torch.utils.data.ConcatDataset(
            (train_loader.dataset, train_rl_loader.dataset)
        )
        a_len = train_loader.__len__()
        ab_len = a_len + train_rl_loader.__len__()
        train_len = ab_len
        a_indices = list(range(a_len))
        b_indices = list(range(a_len, ab_len))
        batch_sampler = MyBatchSampler(a_indices, b_indices, train_loader.batch_size)
        train_loader = torch.utils.data.DataLoader(
            new_dataset, batch_sampler=batch_sampler
        )

        new_dataset = torch.utils.data.ConcatDataset(
            (val_loader.dataset, val_rl_loader.dataset)
        )
        a_len = val_loader.__len__()
        ab_len = a_len + val_rl_loader.__len__()
        val_len = ab_len
        a_indices = list(range(a_len))
        b_indices = list(range(a_len, ab_len))
        batch_sampler = MyBatchSampler(a_indices, b_indices, val_loader.batch_size)
        val_loader = torch.utils.data.DataLoader(
            new_dataset, batch_sampler=batch_sampler
        )

    print()
    print("Beginning training with dataset of size:")
    if args.mix_real:
        print("TRAIN: {}".format(int(train_len * args.batch_size)))
        print("VAL: {}".format(int(val_len * args.batch_size)))
    else:
        print("TRAIN: {}".format(int(len(train_loader) * args.batch_size)))
        print("VAL: {}".format(int(len(val_loader) * args.batch_size)))
    print()

    os.makedirs(os.path.join(args.logdir, args.name), exist_ok=True)

    def normalise_mpm(target, pred, clip=False):
        pred = torch.stack(
            [pred[:, i] / target[:, i].max() for i in range(pred.size(1))], dim=1
        )
        target = torch.stack(
            [target[:, i] / target[:, i].max() for i in range(pred.size(1))], dim=1
        )
        if clip:
            target = torch.clamp(target, 0, 1)
            pred = torch.clamp(pred, 0, 1)
        return target, pred

    def rescale_fwd(pred, ref, clip=False):
        ref_min = ref[:, 0]
        ref_max = ref[:, 1]
        pred = (pred - ref_min) / (ref_max - ref_min)
        if clip:
            pred = torch.clamp(pred, 0, 1)
        return pred

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
                output_postfix="pred",
                separate_folder=False,
                print_log=False,
            )
            saver3 = mn.transforms.SaveImage(
                output_dir=os.path.join(args.logdir, args.name, "debug-train"),
                output_postfix="target",
                separate_folder=False,
                print_log=False,
            )
            saver4 = mn.transforms.SaveImage(
                output_dir=os.path.join(args.logdir, args.name, "debug-train"),
                output_postfix="recon",
                separate_folder=False,
                print_log=False,
            )
        crit = l1_loss if epoch > args.l2_epochs else torch.nn.MSELoss(reduction="none")
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

        torch.autograd.set_detect_anomaly(args.anomaly)

        for step in progress_bar:
            # valid = False # hacky while statement to filter bad brains
            # while not valid:
            #     valid = True
            #     try:
            #         batch = next(train_iter)
            #     except:
            #         train_iter = iter(train_loader)
            #         batch = next(train_iter)
            #     valid *= batch["image"].size(1) == 1 # should be single channel nifti
            #     if isinstance(batch["target"], (torch.Tensor, mn.data.meta_tensor.MetaTensor)):
            #         valid *= (torch.Tensor(list(batch["target"].shape[2:])) > 5).sum().item() == 3 # should be 3D
            #     valid *= isinstance(batch["params"], (torch.Tensor, mn.data.meta_tensor.MetaTensor))
            #     # if above not satisfied, loop back again
            try:
                batch = next(train_iter)
            except:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            images = batch["image"].to(device)
            if "mpm" in batch.keys():
                gt = True
                target = batch["mpm"].to(images)
            else:
                gt = False
                # mask = batch["mask_image"].to(device)
                target_ref = batch["target"]
                target_pixdim = target_ref.pixdim
            params = batch["params"].to(images)
            if (params > 1).sum().item() > 0:
                print("\nParams too large:")
                print("File: ", batch["path"])
                print(params)
            opt.zero_grad(set_to_none=True)

            if args.debug and step < 5:
                saver1(torch.Tensor(images[0].cpu().float()))

            with ctx:
                # reconstruction = torch.nn.functional.softplus(model(main_input=images, hyper_input=params))
                reconstruction = activate(model(main_input=images, hyper_input=params))
                if args.debug and step < 5:
                    saver2(torch.Tensor(reconstruction[0].cpu().float()))
                if gt:
                    if args.debug and step < 5:
                        recon_ = custom_cc.forward_model(
                            reconstruction, params, args.num_ch
                        )
                        target_ = torch.zeros_like(recon_)
                    target, reconstruction = normalise_mpm(
                        target, reconstruction, clip=False
                    )
                    recons_loss = crit(reconstruction, target)
                    # recons_loss = (mask * recons_loss) # mask to only calculate foreground
                    recons_loss = recons_loss.mean()
                    if epoch > args.l2_epochs:
                        recons_loss = recons_loss + perceptual_weight * loss_perceptual(
                            reconstruction[:, 0][None], target[:, 0][None]
                        )
                        recons_loss = recons_loss + perceptual_weight * loss_perceptual(
                            reconstruction[:, 1][None], target[:, 1][None]
                        )
                        recons_loss = recons_loss + perceptual_weight * loss_perceptual(
                            reconstruction[:, 2][None], target[:, 2][None]
                        )
                        recons_loss = recons_loss + perceptual_weight * loss_perceptual(
                            reconstruction[:, 3][None], target[:, 3][None]
                        )
                else:
                    recon_target = custom_cc.forward_model(
                        reconstruction, params, args.num_ch
                    )
                    params = torch.chunk(params[0], images.size(1))
                    recons_loss = 0.0
                    recon_img = []
                    # print()
                    # print(batch["image_quantiles"].shape)
                    for i, params_ in enumerate(params):
                        # print(params_)
                        # print(batch["image_quantiles"][0,i,1])
                        if params_.sum() > 0 and batch["image_quantiles"][0, i, 1] > 0:
                            # recon_target = custom_cc.forward_model(reconstruction[:,[i]], params_)
                            # reslice = mn.transforms.Spacing(pixdim=target_pixdim)
                            # recon_ = reslice(recon_target[0])[None]
                            # target_ = reslice(images[0])[None]
                            recon_ = recon_target[:, [i]]
                            target_ = images[:, [i]]
                            # mask_ = reslice(mask)
                            # print(recon_.shape, batch["image_quantiles"][:,i].shape)
                            recon_ = rescale_fwd(
                                recon_,
                                batch["image_quantiles"][:, i].to(recon_),
                                clip=False,
                            )
                            recons_loss_ = crit(recon_, target_)
                            # recons_loss = (mask_ * recons_loss) # mask to only calculate foreground
                            recons_loss_ = recons_loss_.mean()
                            if epoch > args.l2_epochs:
                                recons_loss_ = (
                                    recons_loss_
                                    + perceptual_weight
                                    * loss_perceptual(recon_, target_)
                                )
                            recons_loss += recons_loss_
                            recon_img.append(recon_)
                        else:
                            recon_img.append(torch.zeros_like(reconstruction[:, [0]]))
                        # print(recons_loss)
                        # print(type(recons_loss))
                    recon_ = torch.cat(recon_img, dim=1)
                    target_ = torch.zeros_like(recon_)

            if type(recons_loss) == float or recons_loss.isnan().sum() != 0:
                print("NaN found in loss!")
                step_deficit += 1
            else:
                if args.amp:
                    scaler.scale(recons_loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
                    scaler.step(opt)
                    scaler.update()
                else:
                    recons_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
                    opt.step()

                epoch_loss += recons_loss.item()

                wandb.log({"train/recon_loss": recons_loss.item()})

            if args.debug and step < 5:
                saver3(torch.Tensor(target_[0].cpu().float()))
                saver4(torch.Tensor(recon_[0].cpu().float()))

            progress_bar.set_postfix(
                {"recon_loss": epoch_loss / (step + 1 - step_deficit)}
            )
        wandb.log({"train/lr": opt.param_groups[0]["lr"]})
        lr_scheduler.step()

        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            inputs = []
            recon_pd = []
            recon_r1 = []
            recon_r2s = []
            recon_mt = []
            val_loss = 0
            val_ssim = 0
            step_deficit = 1e-7
            # val_iter = None
            if args.debug:
                saver1 = mn.transforms.SaveImage(
                    output_dir=os.path.join(args.logdir, args.name, "debug-val"),
                    output_postfix="img",
                    separate_folder=False,
                    print_log=False,
                )
                saver2 = mn.transforms.SaveImage(
                    output_dir=os.path.join(args.logdir, args.name, "debug-val"),
                    output_postfix="pred",
                    separate_folder=False,
                    print_log=False,
                )
                saver3 = mn.transforms.SaveImage(
                    output_dir=os.path.join(args.logdir, args.name, "debug-val"),
                    output_postfix="target",
                    separate_folder=False,
                    print_log=False,
                )
                saver4 = mn.transforms.SaveImage(
                    output_dir=os.path.join(args.logdir, args.name, "debug-val"),
                    output_postfix="recon",
                    separate_folder=False,
                    print_log=False,
                )
            with torch.no_grad():
                # progress_bar = tqdm(range(len(val_loader)), total=len(val_loader), ncols=60)
                # progress_bar.set_description(f"Epoch {epoch}")
                # if val_iter is None:
                #     val_iter = iter(val_loader)
                # for val_step in progress_bar:
                #     valid = False # hacky while statement to filter bad brains
                #     while not valid:
                #         valid = True
                #         try:
                #             batch = next(val_iter)
                #         except:
                #             train_iter = iter(val_loader)
                #             batch = next(val_iter)
                #         valid *= batch["image"].size(1) == 1 # should be single channel nifti
                #         if isinstance(batch["target"], (torch.Tensor, mn.data.meta_tensor.MetaTensor)):
                #             valid *= (torch.Tensor(list(batch["target"].shape[2:])) > 5).sum().item() == 3 # should be 3D
                #         valid *= isinstance(batch["params"], (torch.Tensor, mn.data.meta_tensor.MetaTensor))
                #         # if above not satisfied, loop back again
                for val_step, batch in enumerate(val_loader):
                    images = batch["image"].to(device)
                    if "mpm" in batch.keys():
                        gt = True
                        target = batch["mpm"].to(images)
                    else:
                        gt = False
                        # mask = batch["mask_image"].to(device)
                        target_ref = batch["target"]
                        target_pixdim = target_ref.pixdim
                    params = batch["params"].to(images)
                    if args.amp:
                        ctx = torch.autocast(
                            "cuda" if torch.cuda.is_available() else "cpu"
                        )
                        scaler = torch.cuda.amp.GradScaler()
                    else:
                        ctx = nullcontext()
                    with ctx:
                        with torch.no_grad():
                            # reconstruction = torch.nn.functional.softplus(model(main_input=images, hyper_input=params))
                            reconstruction = activate(
                                model(main_input=images, hyper_input=params)
                            )
                        if args.debug and val_step < 9:
                            saver2(torch.Tensor(reconstruction[0].cpu().float()))
                        if gt:
                            if val_step < 9:
                                recon_ = custom_cc.forward_model(
                                    reconstruction, params, args.num_ch
                                )
                                target_ = torch.zeros_like(recon_)
                            target, reconstruction = normalise_mpm(
                                target, reconstruction, clip=True
                            )
                            recons_loss = crit(reconstruction, target)
                            # recons_loss = (mask * recons_loss) # mask to only calculate foreground
                            recons_loss = recons_loss.mean()
                            if epoch > args.l2_epochs:
                                recons_loss = (
                                    recons_loss
                                    + perceptual_weight
                                    * loss_perceptual(
                                        reconstruction[:, 0][None], target[:, 0][None]
                                    )
                                )
                                recons_loss = (
                                    recons_loss
                                    + perceptual_weight
                                    * loss_perceptual(
                                        reconstruction[:, 1][None], target[:, 1][None]
                                    )
                                )
                                recons_loss = (
                                    recons_loss
                                    + perceptual_weight
                                    * loss_perceptual(
                                        reconstruction[:, 2][None], target[:, 2][None]
                                    )
                                )
                                recons_loss = (
                                    recons_loss
                                    + perceptual_weight
                                    * loss_perceptual(
                                        reconstruction[:, 3][None], target[:, 3][None]
                                    )
                                )
                            recons_ssim = ssim(reconstruction, target)
                        else:
                            recon_target = custom_cc.forward_model(
                                reconstruction, params, args.num_ch
                            )
                            params = torch.chunk(params, images.size(1))
                            recons_loss = 0.0
                            recon_img = []
                            for i, params_ in enumerate(params):
                                if (
                                    params_.sum() > 0
                                    and batch["image_quantiles"][0, i, 1] > 0
                                ):
                                    # recon_target = custom_cc.forward_model(reconstruction[:,[i]], params_)
                                    # reslice = mn.transforms.Spacing(pixdim=target_pixdim)
                                    # recon_ = reslice(recon_target[0])[None]
                                    # target_ = reslice(images[0])[None]
                                    recon_ = recon_target[:, [i]]
                                    target_ = images[:, [i]]
                                    # mask_ = reslice(mask)
                                    recon_ = rescale_fwd(
                                        recon_,
                                        batch["image_quantiles"][:, i].to(recon_),
                                        clip=True,
                                    )
                                    recons_loss_ = crit(recon_, target_)
                                    # recons_loss = (mask_ * recons_loss) # mask to only calculate foreground
                                    recons_loss_ = recons_loss_.mean()
                                    if epoch > args.l2_epochs:
                                        recons_loss_ = (
                                            recons_loss_
                                            + perceptual_weight
                                            * loss_perceptual(recon_, target_)
                                        )
                                    recons_loss += recons_loss_
                                    recons_ssim = ssim(recon_, target_)
                                    recon_img.append(recon_)
                                else:
                                    recon_img.append(
                                        torch.zeros_like(reconstruction[:, [0]])
                                    )
                            recon_ = torch.cat(recon_img, dim=1)
                            target_ = torch.zeros_like(recon_)
                            reconstruction, reconstruction = normalise_mpm(
                                reconstruction, reconstruction, clip=True
                            )

                    if type(recons_loss) == float or recons_loss.isnan().sum() != 0:
                        print("NaN found in loss!")
                        step_deficit += 1
                    else:
                        val_loss += recons_loss.item()
                        val_ssim += recons_ssim.item()

                    n_samples = 4 if args.use_real_mpms else 9
                    if val_step < n_samples:
                        inputs.append(
                            images[0, 0, ..., images.size(-1) // 2].cpu().float()[None]
                        )
                        reconstruction = torch.nan_to_num(
                            reconstruction
                        )  # if NaN, just show empty image
                        recon_pd.append(
                            reconstruction[0, 0, ..., images.size(-1) // 2]
                            .cpu()
                            .float()[None]
                        )
                        recon_r1.append(
                            reconstruction[0, 1, ..., images.size(-1) // 2]
                            .cpu()
                            .float()[None]
                        )
                        recon_r2s.append(
                            reconstruction[0, 2, ..., images.size(-1) // 2]
                            .cpu()
                            .float()[None]
                        )
                        recon_mt.append(
                            reconstruction[0, 3, ..., images.size(-1) // 2]
                            .cpu()
                            .float()[None]
                        )
                        if args.debug:
                            saver1(torch.Tensor(images[0].cpu().float()))
                            saver3(torch.Tensor(target_[0].cpu().float()))
                            saver4(torch.Tensor(recon_[0].cpu().float()))
                    elif val_step == n_samples:
                        grid_inputs = make_grid(
                            inputs,
                            nrow=int(n_samples**0.5),
                            padding=5,
                            normalize=True,
                            scale_each=True,
                        )
                        grid_recon_pd = make_grid(
                            recon_pd,
                            nrow=int(n_samples**0.5),
                            padding=5,
                            normalize=True,
                            scale_each=True,
                        )
                        grid_recon_r1 = make_grid(
                            recon_r1,
                            nrow=int(n_samples**0.5),
                            padding=5,
                            normalize=True,
                            scale_each=True,
                        )
                        grid_recon_r2s = make_grid(
                            recon_r2s,
                            nrow=int(n_samples**0.5),
                            padding=5,
                            normalize=True,
                            scale_each=True,
                        )
                        grid_recon_mt = make_grid(
                            recon_mt,
                            nrow=int(n_samples**0.5),
                            padding=5,
                            normalize=True,
                            scale_each=True,
                        )
                        wandb.log(
                            {
                                "val/examples": [
                                    wandb.Image(
                                        grid_inputs[0].numpy(), caption="Input"
                                    ),
                                    wandb.Image(
                                        grid_recon_pd[0].numpy(), caption="Predicted PD"
                                    ),
                                    wandb.Image(
                                        grid_recon_r1[0].numpy(), caption="Predicted R1"
                                    ),
                                    wandb.Image(
                                        grid_recon_r2s[0].numpy(),
                                        caption="Predicted R2s",
                                    ),
                                    wandb.Image(
                                        grid_recon_mt[0].numpy(), caption="Predicted MT"
                                    ),
                                ]
                            }
                        )

                metric = val_ssim / (val_step + 1 - step_deficit)
                wandb.log({"val/recon_loss": val_loss / (val_step + 1 - step_deficit)})
                wandb.log({"val/recon_ssim": val_ssim / (val_step + 1 - step_deficit)})
                print(
                    "Validation complete. Loss: {:.3f} // SSIM: {:.3f}".format(
                        val_loss / (val_step + 1 - step_deficit),
                        val_ssim / (val_step + 1 - step_deficit),
                    )
                )

                if metric > metric_best:
                    metric_best = metric
                    torch.save(
                        {
                            "net": model.state_dict(),
                            "opt": opt.state_dict(),
                            "lr": lr_scheduler.state_dict(),
                            "wandb": WandBID(wandb.run.id).state_dict(),
                            "epoch": Epoch(epoch).state_dict(),
                            "metric": Metric(metric_best).state_dict(),
                        },
                        os.path.join(
                            args.logdir, args.name, "checkpoint_best.pt".format(epoch)
                        ),
                    )
                torch.save(
                    {
                        "net": model.state_dict(),
                        "opt": opt.state_dict(),
                        "lr": lr_scheduler.state_dict(),
                        "wandb": WandBID(wandb.run.id).state_dict(),
                        "epoch": Epoch(epoch).state_dict(),
                        "metric": Metric(metric_best).state_dict(),
                    },
                    os.path.join(args.logdir, args.name, "checkpoint.pt".format(epoch)),
                )


def set_up():
    parser = argparse.ArgumentParser(argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--name", type=str, help="Name of WandB run.")
    parser.add_argument(
        "--epochs", type=int, default=1000, help="Number of epochs for training."
    )
    parser.add_argument(
        "--epoch_length", type=int, default=200, help="Number of iterations per epoch."
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument(
        "--val_interval", type=int, default=2, help="Validation interval."
    )
    parser.add_argument(
        "--l2_epochs",
        type=int,
        default=10,
        help="Number of warmup epochs using L2 loss.",
    )
    parser.add_argument(
        "--num_ch", type=int, default=1, help="Number of input image channels."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Number of subjects to use per batch."
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="baur",
        help="Loss function to use. Options: [l1, l2, huber, baur, barron]",
    )
    parser.add_argument("--amp", default=False, action="store_true")
    parser.add_argument(
        "--mni",
        default=False,
        action="store_true",
        help="Use images normalised to MNI template.",
    )
    parser.add_argument(
        "--logdir", type=str, default="./", help="Path to saved outputs"
    )
    parser.add_argument("--freesurfer", default=False, action="store_true")
    parser.add_argument("--resume", default=False, action="store_true")
    parser.add_argument("--resume_best", default=False, action="store_true")
    parser.add_argument("--baseline", default=False, action="store_true")
    parser.add_argument("--mix_real", default=False, action="store_true")
    parser.add_argument("--anomaly", default=False, action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use. If not specified then will check for CUDA.",
    )
    parser.add_argument(
        "--local_paths",
        default=False,
        action="store_true",
        help="Run with local paths (for FIL PC).",
    )
    parser.add_argument(
        "--lowres",
        default=False,
        action="store_true",
        help="Train with un-cropped 2D images.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Save sample images before training.",
    )
    parser.add_argument(
        "--no_augs",
        default=False,
        action="store_true",
        help="Perform no intensity augmentation.",
    )
    parser.add_argument(
        "--use_real_mpms",
        default=False,
        action="store_true",
        help="Train on real MPMs.",
    )
    parser.add_argument(
        "--upsample",
        default="transpose",
        type=str,
        help="Method of upsampling. Options: ['transpose', 'subpixel', 'interp'].",
    )
    parser.add_argument(
        "--lesion",
        action="store_true",
        help="Enable lesion synthesis.",
    )
    parser.add_argument(
        "--pseudolabels",
        action="store_true",
        help="Enable pseudolabels.",
    )
    parser.add_argument(
        "--reset_wandb",
        action="store_true",
        help="Reset WandB.",
    )
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
    assert args.batch_size == 1, "Currently only support batch size of 1."
    if args.baseline:
        train_loader, val_loader, train_transform = preprocess_baseline.get_loaders(
            args.batch_size,
            device,
            args.lowres,
            args.num_ch,
            args.local_paths,
            args.mni,
        )
    elif args.use_real_mpms:
        train_loader, val_loader, train_transform = preprocess_real_mpms.get_loaders(
            args.batch_size,
            device,
            args.lowres,
            args.num_ch,
            args.local_paths,
            args.mni,
            args.no_augs,
        )
    elif args.freesurfer:
        train_loader, val_loader, train_transform = preprocess_freesurfer.get_loaders(
            args.batch_size,
            device,
            args.lowres,
            args.num_ch,
            args.local_paths,
            args.mni or args.lesion, # if lesion, use mni
            args.lesion,
            args.no_augs,
            args.pseudolabels,
        )
    else:
        train_loader, val_loader, train_transform = get_loaders(
            args.batch_size,
            device,
            args.lowres,
            args.num_ch,
            args.local_paths,
            args.mni or args.lesion, # if lesion, use mni
            use_lesion=args.lesion,
            pseudolabels=args.pseudolabels,
        )
    # train_loader = val_loader = None

    if args.debug:
        saver1 = mn.transforms.SaveImage(
            output_dir=os.path.join(args.logdir, args.name, "debug"),
            output_postfix="img",
            separate_folder=False,
            print_log=False,
        )
        saver2 = mn.transforms.SaveImage(
            output_dir=os.path.join(args.logdir, args.name, "debug"),
            output_postfix="lab",
            separate_folder=False,
            print_log=False,
        )
        saver3 = mn.transforms.SaveImage(
            output_dir=os.path.join(args.logdir, args.name, "debug"),
            output_postfix="mpm",
            separate_folder=False,
            print_log=False,
        )
        for i, batch in enumerate(val_loader):
            if i > 5:
                break
            else:
                print(
                    "Image: ",
                    batch["image"].shape,
                    "min={}".format(batch["image"].min()),
                    "max={}".format(batch["image"].max()),
                )
                saver1(
                    torch.Tensor(batch["image"][0].cpu().float()),
                )
                if "label" in batch.keys():
                    print(
                        "Label: ",
                        batch["label"].shape,
                        "min={}".format(batch["label"].min()),
                        "max={}".format(batch["label"].max()),
                    )
                    saver2(
                        (
                            torch.Tensor(batch["label"][0].cpu().float())
                            if args.freesurfer
                            else torch.Tensor(
                                torch.argmax(batch["label"][0], dim=0)[None]
                                .cpu()
                                .float()
                            )
                        ),
                    )
                if "mpm" in batch.keys():
                    print(
                        "MPM: ",
                        batch["mpm"].shape,
                        "min={}".format(batch["mpm"].min()),
                        "max={}".format(batch["mpm"].max()),
                    )
                    saver3(
                        torch.Tensor(batch["mpm"][0].cpu().float()),
                    )

    return args, device, train_loader, val_loader, train_transform


def main():
    args, device, train_loader, val_loader, train_transform = set_up()
    run_model(args, device, train_loader, val_loader, train_transform)


if __name__ == "__main__":
    main()
