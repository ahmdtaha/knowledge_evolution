import time
import torch
import numpy as np
import torch.nn as nn
from utils import net_utils
from layers.CS_KD import KDLoss
from utils.eval_utils import accuracy
from utils.logging import AverageMeter, ProgressMeter



__all__ = ["train", "validate"]



kdloss = KDLoss(4).cuda()

def train(train_loader, model, criterion, optimizer, epoch, cfg, writer):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        train_loader.num_batches,
        [batch_time, data_time, losses, top1, top5],cfg,
        prefix=f"Epoch: [{epoch}]",
    )

    # switch to train mode
    model.train()

    batch_size = train_loader.batch_size
    num_batches = train_loader.num_batches
    end = time.time()

    for i , data in enumerate(train_loader):
        # images, target = data[0]['data'],data[0]['label'].long().squeeze()
        images, target = data[0].cuda(),data[1].long().squeeze().cuda()
        # measure data loading time
        data_time.update(time.time() - end)

        if cfg.cs_kd:

            batch_size = images.size(0)
            loss_batch_size = batch_size // 2
            targets_ = target[:batch_size // 2]
            outputs = model(images[:batch_size // 2])
            loss = torch.mean(criterion(outputs, targets_))
            # loss += loss.item()

            with torch.no_grad():
                outputs_cls = model(images[batch_size // 2:])
            cls_loss = kdloss(outputs[:batch_size // 2], outputs_cls.detach())
            lamda = 3
            loss += lamda * cls_loss
            acc1, acc5 = accuracy(outputs, targets_, topk=(1, 5))
        else:
            batch_size = images.size(0)
            loss_batch_size = batch_size
            #compute output
            output = model(images)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # print(i, batch_size, loss)

        # measure accuracy and record loss

        losses.update(loss.item(), loss_batch_size)
        top1.update(acc1.item(), loss_batch_size)
        top5.update(acc5.item(), loss_batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        if i % cfg.print_freq == 0 or i == num_batches-1:
            t = (num_batches * epoch + i) * batch_size
            progress.display(i)
            progress.write_to_tensorboard(writer, prefix="train", global_step=t)

    # train_loader.reset()
    # print(top1.count)
    return top1.avg, top5.avg


def validate(val_loader, model, criterion, args, writer, epoch):
    batch_time = AverageMeter("Time", ":6.3f", write_val=False)
    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    progress = ProgressMeter(
        val_loader.num_batches, [batch_time, losses, top1, top5],args, prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()

        # confusion_matrix = torch.zeros(args.num_cls,args.num_cls)
        for i, data in enumerate(val_loader):
            # images, target = data[0]['data'], data[0]['label'].long().squeeze()
            images, target = data[0].cuda(), data[1].long().squeeze().cuda()


            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            # print(target,torch.mean(images),acc1,acc5,loss,torch.mean(output))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))


            # _, preds = torch.max(output, 1)
            # for t, p in zip(target.view(-1), preds.view(-1)):
            #     confusion_matrix[t.long(), p.long()] += 1

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()



            if i % args.print_freq == 0:
                progress.display(i)

        progress.display(val_loader.num_batches)

        if writer is not None:
            progress.write_to_tensorboard(writer, prefix="test", global_step=epoch)

    # torch.save(confusion_matrix,'./conf_mat.pt')
    # print(top1.count)
    return top1.avg, top5.avg




