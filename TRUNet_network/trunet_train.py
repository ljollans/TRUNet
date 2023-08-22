import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from statistics import fmean


def dice_score(pred, gt):  # data in shape [batch, classes, h, w, d]
    dice = []
    for batchloop in range(gt.shape[0]):
        dice_tmp = []
        for roi in range(gt.shape[1]):
            if roi > 0:  # skip background
                pred_tmp = pred[int(batchloop), int(roi)]
                gt_tmp = gt[int(batchloop), int(roi)]
                a = np.sum(pred_tmp[gt_tmp == 1])
                b = np.sum(pred_tmp)
                c = np.sum(gt_tmp)
                if a == 0:
                    metric = 0
                else:
                    metric_ = a * 2.0 / (b + c)
                    metric = metric_.item()
                dice_tmp.append(metric)
        dice.append(fmean(dice_tmp))
    return fmean(dice)


def one_hot_encoder(input_tensor, n_classes):
    tensor_list = []
    for i in range(n_classes):
        temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()


def to_one_arr_encoding(input_tensor):  # input shape: [batch, channels, h, w, d]
    new_arr = torch.zeros(input_tensor.shape)
    for batchloop in range(input_tensor.shape[0]):
        for d in range(input_tensor.shape[1]):
            new_arr[batchloop, d] = torch.where(input_tensor[batchloop, d] == 1, d + 1, 0)
    return new_arr.sum(1).unsqueeze(1)


def trainer(args, config, model, savepath):
    # Initializations
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    model.to(device)

    # Parameters
    loss_function = config['loss_function']
    optimizer = config['optimizer']
    dataset_train = config['ds_train']
    dataset_val = config['ds_val']
    save_interval = config['save_interval']

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # Data Loaders
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True,
                              worker_init_fn=worker_init_fn)
    val_loader = DataLoader(dataset_val, batch_size=1, shuffle=True, num_workers=2, pin_memory=True,
                            worker_init_fn=worker_init_fn)

    max_iterations = args.max_epochs * len(train_loader)

    # logging
    logging.basicConfig(filename=args.save_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.info(args)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iterations))
    writer = SummaryWriter(savepath + '/log')

    best_metric = -1
    best_metric_epoch = -1
    metric_values = []
    iter_num = 0

    ############################
    #         Training         #
    ############################

    for epoch in range(args.max_epochs):
        epoch_loss = 0
        model.train()
        for i_batch, sampled_batch in enumerate(train_loader):
            # get inputs and targets
            inputs, targets = sampled_batch['image'], sampled_batch['label']
            # here the input and target have the shape [batch, H, L, D]
            # so we need to add the channel dimension
            inputs, targets = inputs.unsqueeze(1), targets.unsqueeze(1)
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)

            loss.backward()
            optimizer.step()

            # update learning rate
            epoch_loss += loss
            lr_ = args.base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            lrlog = lr_
            writer.add_scalar('info/lr', lr_, iter_num)
            iter_num = iter_num + 1

            # write to log
            writer.add_scalar('info/total_loss', loss, iter_num)
            # logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))

        epoch_loss = epoch_loss / len(train_loader)

        logging.info('epoch %d : mean loss : %f' % (epoch, epoch_loss))

        ############################
        #        Validation        #
        ############################

        model.eval()
        with torch.no_grad():
            dice_tmp = []
            for i_batch, sampled_batch in enumerate(val_loader):
                # get inputs and targets
                inputs, targets = sampled_batch['image'], sampled_batch['label']
                # targets needs to be transformed to one-hot encoding
                targets = targets.unsqueeze(1)
                targets = one_hot_encoder(targets, args.num_classes)
                inputs, targets = inputs.to(device), targets.to(device)

                val_outputs = model(inputs)
                m = nn.Softmax(dim=1)
                val_outputs = m(val_outputs)

                # compute metric for current iteration
                dice_tmp.append(dice_score(val_outputs.cpu().data.numpy(), targets.cpu().data.numpy()))

            # aggregate the final mean dice result
            metric = fmean(dice_tmp)

            # write to log
            writer.add_scalar('info/validation_metric', metric, epoch)
            logging.info('iteration %d : dice score : %f' % (epoch, metric))

            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(args.save_path, "best_metric_model.pth"))
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}. Current learning rate {lrlog}"
                f"\nbest mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )

        ############################
        #          Saving          #
        ############################

        # add an example to tensorboard logging
        labs = to_one_arr_encoding(targets)
        outputs = torch.argmax(torch.softmax(val_outputs, dim=1), dim=1, keepdim=True)

        if len(inputs.shape) == 5:
            image = inputs[:, :, :, :, round(args.img_size / 2)]
            labs = labs[:, :, :, :, round(args.img_size / 2)]
            outputs = outputs[:, :, :, round(args.img_size / 2)]
        else:
            image = inputs

        image = image[0]
        labs = torch.squeeze(labs * 50, 1)
        outputs = outputs[0] * 50
        image = (image - image.min()) / (image.max() - image.min())
        writer.add_image('train/Image', image, iter_num)
        writer.add_image('train/Prediction', outputs, iter_num)
        writer.add_image('train/GroundTruth', labs, iter_num)

        if (epoch + 1) % save_interval == 0:
            save_mode_path = os.path.join(args.save_path, 'epoch_' + str(epoch) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch >= args.max_epochs - 1:
            save_mode_path = os.path.join(args.save_path, 'epoch_' + str(epoch) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            break

    writer.close()
    return "Training Finished!"