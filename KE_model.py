import os
import time
import data
import torch
import random
import importlib
import torch.optim
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
from utils import net_utils
from utils import csv_utils
from utils import gpu_utils
from utils import path_utils
# from layers import ml_losses
from datetime import timedelta
import torch.utils.data.distributed
from utils.schedulers import get_policy
from torch.utils.tensorboard import SummaryWriter
from utils.logging import AverageMeter, ProgressMeter



def ke_cls_train(cfg, model,generation):
    cfg.logger.info(cfg)
    if cfg.seed is not None:
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

    # train, validate, modifier = get_trainer(cfg)
    train, validate = get_trainer(cfg)

    if cfg.gpu is not None:
        cfg.logger.info("Use GPU: {} for training".format(cfg.gpu))

    # if cfg.pretrained:
    #     net_utils.load_pretrained(cfg.pretrained,cfg.multigpu[0], model)

    optimizer = get_optimizer(cfg, model)
    cfg.logger.info(f"=> Getting {cfg.set} dataset")
    dataset = getattr(data, cfg.set)(cfg)

    lr_policy = get_policy(cfg.lr_policy)(optimizer, cfg)

    if cfg.label_smoothing is None:
        softmax_criterion = nn.CrossEntropyLoss().cuda()
    else:
        softmax_criterion = net_utils.LabelSmoothing(smoothing=cfg.label_smoothing).cuda()


    criterion = lambda output,target: softmax_criterion(output, target)


    # optionally resume from a checkpoint
    best_val_acc1 = 0.0
    best_val_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0

    if cfg.resume:
        best_val_acc1 = resume(cfg, model, optimizer)

    # Data loading code
    # if cfg.evaluate:
    #     last_val_acc1, last_val_acc5 = validate(
    #         dataset.val_loader, model, criterion, cfg, writer=None, epoch=cfg.start_epoch
    #     )
    #
    #     return

    run_base_dir, ckpt_base_dir, log_base_dir = path_utils.get_directories(cfg,generation)
    cfg.ckpt_base_dir = ckpt_base_dir

    writer = SummaryWriter(log_dir=log_base_dir)
    epoch_time = AverageMeter("epoch_time", ":.4f", write_avg=False)
    validation_time = AverageMeter("validation_time", ":.4f", write_avg=False)
    train_time = AverageMeter("train_time", ":.4f", write_avg=False)
    progress_overall = ProgressMeter(
        1, [epoch_time, validation_time, train_time],cfg, prefix="Overall Timing"
    )

    end_epoch = time.time()
    cfg.start_epoch = cfg.start_epoch or 0
    last_val_acc1 = None


    start_time = time.time()
    gpu_info = gpu_utils.GPU_Utils(gpu_index=cfg.gpu)


    # Start training
    for epoch in range(cfg.start_epoch, cfg.epochs):
        lr_policy(epoch, iteration=None)
        # if epoch  == cfg.start_epoch:
        #     modifier(cfg, epoch, model)

        cur_lr = net_utils.get_lr(optimizer)
        # print(cur_lr)
        # train for one epoch
        start_train = time.time()
        train_acc1, train_acc5 = train(
            dataset.train_loader, model, criterion, optimizer, epoch, cfg, writer=writer
        )
        train_time.update((time.time() - start_train) / 60)

        if (epoch+1) % cfg.test_interval == 0:
            # evaluate on validation set
            start_validation = time.time()
            last_val_acc1, last_val_acc5 = validate(dataset.val_loader, model, criterion, cfg, writer, epoch)
            validation_time.update((time.time() - start_validation) / 60)

            # remember best acc@1 and save checkpoint
            is_best = last_val_acc1 > best_val_acc1
            best_val_acc1 = max(last_val_acc1, best_val_acc1)
            best_val_acc5 = max(last_val_acc5, best_val_acc5)
            best_train_acc1 = max(train_acc1, best_train_acc1)
            best_train_acc5 = max(train_acc5, best_train_acc5)

            save = ((epoch % cfg.save_every) == 0) and cfg.save_every > 0

            elapsed_time = time.time() - start_time
            seconds_todo = (cfg.epochs - epoch) * (elapsed_time/cfg.test_interval)
            estimated_time_complete = timedelta(seconds=int(seconds_todo))
            start_time = time.time()
            cfg.logger.info(f"==> ETA: {estimated_time_complete}\tGPU-M: {gpu_info.gpu_mem_usage()}\tGPU-U: {gpu_info.gpu_utilization()}")
            if is_best or save or epoch == cfg.epochs - 1:
                if is_best:
                    cfg.logger.info(f"==> best {last_val_acc1:.02f} saving at {ckpt_base_dir / 'model_best.pth'}")

                net_utils.save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "arch": cfg.arch,
                        "state_dict": model.state_dict(),
                        "best_acc1": best_val_acc1,
                        "best_acc5": best_val_acc5,
                        "best_train_acc1": best_train_acc1,
                        "best_train_acc5": best_train_acc5,
                        "optimizer": optimizer.state_dict(),
                        "curr_acc1": last_val_acc1,
                        "curr_acc5": last_val_acc5,
                    },
                    is_best,
                    filename=ckpt_base_dir / f"epoch_{epoch}.state",
                    save=save or epoch == cfg.epochs - 1,
                )

            epoch_time.update((time.time() - end_epoch) / 60)
            progress_overall.display(epoch)
            progress_overall.write_to_tensorboard(
                writer, prefix="diagnostics", global_step=epoch
            )


            writer.add_scalar("test/lr", cur_lr, epoch)
            end_epoch = time.time()


    if cfg.eval_tst:
        last_tst_acc1, last_tst_acc5 = validate(dataset.tst_loader, model, criterion, cfg, writer, 0)
        best_tst_acc1 = 0
        best_tst_acc5 = 0
        # net_utils.load_pretrained(ckpt_base_dir / 'model_best.pth',cfg.multigpu[0],model)
        # best_tst_acc1, best_tst_acc5 = validate(dataset.tst_loader, model, criterion, cfg, writer, 0)
    else:
        last_tst_acc1 = 0
        last_tst_acc5 = 0
        best_tst_acc1 = 0
        best_tst_acc5 = 0


    csv_utils.write_cls_result_to_csv(
        ## Validation
        curr_acc1=last_val_acc1,
        curr_acc5=last_val_acc5,
        best_acc1=best_val_acc1,
        best_acc5=best_val_acc5,

        ## Test
        last_tst_acc1=last_tst_acc1,
        last_tst_acc5=last_tst_acc5,
        best_tst_acc1=best_tst_acc1,
        best_tst_acc5=best_tst_acc5,

        ## Train
        best_train_acc1=best_train_acc1,
        best_train_acc5=best_train_acc5,


        split_rate=cfg.split_rate,
        bias_split_rate=cfg.bias_split_rate,

        base_config=cfg.name,
        name=cfg.name,
    )

    cfg.logger.info(f"==> Final Best {best_val_acc1:.02f}, saving at {ckpt_base_dir / 'model_best.pth'}")
    return ckpt_base_dir ## Do Not return the model because you just reloaded the best one


def get_trainer(args):
    args.logger.info(f"=> Using trainer from trainers.{args.trainer}")
    trainer = importlib.import_module(f"trainers.{args.trainer}")

    return trainer.train, trainer.validate #, trainer.modifier


def resume(args, model, optimizer):
    if os.path.isfile(args.resume):
        args.logger.info(f"=> Loading checkpoint '{args.resume}'")

        checkpoint = torch.load(args.resume, map_location=f"cuda:{args.gpu}")
        if args.start_epoch is None:
            args.logger.info(f"=> Setting new start epoch at {checkpoint['epoch']}")
            args.start_epoch = checkpoint["epoch"]

        best_acc1 = checkpoint["best_acc1"]

        model.load_state_dict(checkpoint["state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer"])

        args.logger.info(f"=> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")

        return best_acc1
    else:
        args.logger.info(f"=> No checkpoint found at '{args.resume}'")




def get_optimizer(args, model,fine_tune=False,criterion=None):
    for n, v in model.named_parameters():
        if v.requires_grad:
            args.logger.info("<DEBUG> gradient to {}".format(n))

        if not v.requires_grad:
            args.logger.info("<DEBUG> no gradient to {}".format(n))

    param_groups = model.parameters()
    if fine_tune:
        # Train Parameters
        param_groups = [
            {'params': list(
                set(model.parameters()).difference(set(model.model.embedding.parameters()))) if args.gpu != -1 else
            list(set(model.module.parameters()).difference(set(model.module.model.embedding.parameters())))},
            {
                'params': model.model.embedding.parameters() if args.gpu != -1 else model.module.model.embedding.parameters(),
                'lr': float(args.lr) * 1},
        ]
        if args.ml_loss == 'Proxy_Anchor':
            param_groups.append({'params': criterion.proxies, 'lr': float(args.lr) * 100})

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                          momentum=args.momentum, weight_decay=args.weight_decay)
        # parameters = list(model.named_parameters())
        # bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
        # rest_params = [v for n, v in parameters if ("bn" not in n) and v.requires_grad]
        # optimizer = torch.optim.SGD(
        #     [
        #         {
        #             "params": bn_params,
        #             "weight_decay": 0 if args.no_bn_decay else args.weight_decay,
        #         },
        #         {"params": rest_params, "weight_decay": args.weight_decay},
        #     ],
        #     args.lr,
        #     momentum=args.momentum,
        #     weight_decay=args.weight_decay,
        #     nesterov=args.nesterov,
        # )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, param_groups), lr=args.lr
        )
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(param_groups, lr=args.lr, alpha=0.9, weight_decay = args.weight_decay, momentum = 0.9)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay = args.weight_decay)
    else:
        raise NotImplemented('Invalid Optimizer {}'.format(args.optimizer))

    return optimizer

