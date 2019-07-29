import torch.optim as optim
from early_stopping import EarlyStopping
from load_data import *
from PIXOR_Net import PIXOR
import torch.nn as nn
import time

##############
# focal loss #
##############


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, size_average=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, prediction, target):

        # pass class prediction through sigmoid
        sigmoid = nn.Sigmoid()
        prediction = sigmoid(prediction)

        # get class probability
        pt = torch.where(target == 1.0, prediction, 1-prediction)

        # compute focal loss
        loss = -1 * (1-pt)**self.gamma * torch.log(pt)

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


##################
# calculate loss #
##################


def calc_loss(batch_predictions, batch_labels):

    # classification loss
    classification_prediction = batch_predictions[:, :, :, -1].contiguous().flatten()
    classification_label = batch_labels[:, :, :, -1].contiguous().flatten()
    focal_loss = FocalLoss(gamma=2)
    classification_loss = focal_loss(classification_prediction, classification_label)

    # regression loss
    regression_prediction = batch_predictions[:, :, :, :-1]
    regression_prediction = regression_prediction.contiguous().view([regression_prediction.size(0)*
                        regression_prediction.size(1)*regression_prediction.size(2), regression_prediction.size(3)])
    regression_label = batch_labels[:, :, :, :-1]
    regression_label = regression_label.contiguous().view([regression_label.size(0)*regression_label.size(1)*
                                                           regression_label.size(2), regression_label.size(3)])
    positive_mask = torch.nonzero(torch.sum(torch.abs(regression_label), dim=1))
    pos_regression_label = regression_label[positive_mask.squeeze(), :]
    pos_regression_prediction = regression_prediction[positive_mask.squeeze(), :]
    smooth_l1 = nn.SmoothL1Loss(reduction='sum')
    regression_loss = smooth_l1(pos_regression_prediction, pos_regression_label)

    # add two loss components
    multi_task_loss = classification_loss.add(regression_loss)

    return multi_task_loss


###############
# Train Model #
###############


def train_model(model, optimizer, scheduler, data_loaders, n_epochs=25, show_times=False):

    # evaluation dict
    # metrics = {'train_loss': [], 'val_loss': [], 'lr': []}
    metrics = np.load('Metrics/metrics_5.npz', allow_pickle=True)['history'].item()
    # early stopping object
    early_stopping = EarlyStopping(patience=8, verbose=True)

    # epochs
    moving_loss = {'train': metrics['train_loss'][-1], 'val': metrics['val_loss'][-1]}
    for epoch in range(5, n_epochs):

        # each epoch has a training and validation phase
        for phase in ['train', 'val']:

            # track average loss per batch
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            for batch_id, (batch_data, batch_labels) in enumerate(data_loaders[phase]):

                # zero the parameter gradients
                optimizer.zero_grad()

                # track history only if in train phase
                with torch.set_grad_enabled(phase == 'train'):

                    # forward pass
                    forward_pass_start_time = time.time()
                    batch_predictions = model(batch_data)
                    forward_pass_end_time = time.time()
                    batch_predictions = batch_predictions.permute([0, 2, 3, 1])

                    # calculate loss
                    calc_loss_start_time = time.time()
                    loss = calc_loss(batch_predictions, batch_labels)
                    calc_loss_end_time = time.time()

                    # accumulate loss
                    if moving_loss[phase] is None:
                        moving_loss[phase] = loss.item()
                    else:
                        moving_loss[phase] = 0.99 * moving_loss[phase] + 0.01 * loss.item()

                    # append loss for each phase
                    metrics[phase + '_loss'].append(moving_loss[phase])

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        backprop_start_time = time.time()
                        loss.backward()
                        optimizer.step()
                        backprop_end_time = time.time()

                        if show_times:
                            print('Forward Pass Time: {:.2f}'.format(forward_pass_end_time - forward_pass_start_time))
                            print('Calc Loss Time: {:.2f} '.format(calc_loss_end_time - calc_loss_start_time))
                            print('Backprop Time: {:.2f}'.format(backprop_end_time-backprop_start_time))

                        if (batch_id+1) % 10 == 0:
                            n_batches_per_epoch = data_loaders[phase].dataset.__len__() // data_loaders[phase].batch_size
                            print("{:d}/{:d} iterations\tAvg. Loss: {:.4f}".format(batch_id+1, n_batches_per_epoch, moving_loss[phase]))

        # keep track of learning rate
        for param_group in optimizer.param_groups:
            metrics['lr'].append(param_group['lr'])

        # scheduler step
        scheduler.step()

        # output progress
        print('###########################################################')
        print('Epoch: ' + str(epoch+1) + '/' + str(n_epochs))
        print('Learning Rate: ', metrics['lr'][-1])
        print('Training Loss: %.4f' % metrics['train_loss'][-1])
        print('Validation Loss: %.4f' % metrics['val_loss'][-1])

        # save metrics
        np.savez('./Metrics/metrics_' + str(epoch + 1) + '.npz', history=metrics)

        # check early stopping
        early_stopping(val_loss=metrics['val_loss'][-1], epoch=epoch, model=model)
        if early_stopping.early_stop:
            print('Early Stopping!')
            break

    print('Training Finished!')
    print('Final Model was trained for ' + str(early_stopping.best_epoch) + ' epochs and achieved minimum loss of '
                                                                            '%.4f!' % early_stopping.val_loss_min)

    return metrics


if __name__ == '__main__':

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # training parameters
    n_epochs = 30
    batch_size = 6
    initial_learning_rate = 1e-4

    # create data loader
    base_dir = 'kitti/'
    dataset_dir = 'object/'
    root_dir = os.path.join(base_dir, dataset_dir)
    data_loader = load_dataset(root=root_dir, batch_size=batch_size, device=device)

    # create model
    pixor = PIXOR().to(device)
    pixor.load_state_dict(torch.load('Models/PIXOR_Epoch_5.pt', map_location=device))

    # create optimizer and scheduler objects
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, pixor.parameters()), lr=initial_learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20, 25], gamma=0.1)

    # train model
    history = train_model(pixor, optimizer, scheduler, data_loader, n_epochs=n_epochs)

    # save training history
    np.savez('history.npz', history=history)
