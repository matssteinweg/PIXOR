import torch.optim as optim
from load_data import *
from PIXOR import PIXOR
import torch.nn as nn
import time

##################
# early stopping #
##################


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(self, patience=7, verbose=False):
        """
        :param patience: How many epochs wait after the last validation loss improvement
        :param verbose: If True, prints a message for each validation loss improvement.
        """

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, epoch, model):

        score = -val_loss

        # first epoch
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch + 1
            self.save_checkpoint(val_loss, model)

        # validation loss increased
        elif score < self.best_score:

            # increase counter
            self.counter += 1

            print('Validation loss did not decrease ({:.6f} --> {:.6f})'.format(self.val_loss_min, val_loss))
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            print('###########################################################')

            # stop training if patience is reached
            if self.counter >= self.patience:
                self.early_stop = True

        # validation loss decreased
        else:
            self.best_score = score
            self.best_epoch = epoch + 1
            self.save_checkpoint(val_loss, model)

            # reset counter
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves model when validation loss decreased.
        """

        if self.verbose:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  '
                  'Saving model ...'.format(self.val_loss_min, val_loss))
            print('###########################################################')

        # save model
        torch.save(model.state_dict(), 'Models/PIXOR_Epoch_' + str(self.best_epoch) + '.pt')

        # set current loss as new minimum loss
        self.val_loss_min = val_loss


##############
# focal loss #
##############


class FocalLoss(nn.Module):
    """
    Focal loss class. Stabilize training by reducing the weight of easily classified background sample and focussing
    on difficult foreground detections.
    """

    def __init__(self, gamma=0, size_average=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, prediction, target):

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
    """
    Calculate the final loss function as a sum of the classification and the regression loss.
    :param batch_predictions: predictions for the current batch | shape: [batch_size, OUTPUT_DIM_0, OUTPUT_DIM_1, OUTPUT_DIM_CLA+OUTPUT_DIM_REG]
    :param batch_labels: labels for the current batch | shape: [batch_size, OUTPUT_DIM_0, OUTPUT_DIM_1, OUTPUT_DIM_CLA+OUTPUT_DIM_REG]
    :return: compouted loss
    """

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
# train model #
###############


def train_model(model, optimizer, scheduler, data_loaders, n_epochs=25, show_times=False):

    # evaluation dict
    metrics = {'train_loss': [], 'val_loss': [], 'lr': []}

    # early stopping object
    early_stopping = EarlyStopping(patience=8, verbose=True)

    # moving loss
    moving_loss = {'train': metrics['train_loss'][-1], 'val': metrics['val_loss'][-1]}

    # epochs
    for epoch in range(n_epochs):

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
                            print("{:d}/{:d} iterations | training loss: {:.4f}".format(batch_id+1, n_batches_per_epoch, moving_loss[phase]))

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
    initial_learning_rate = 1e-3

    # create data loader
    root_dir = 'Data/'
    data_loader = load_dataset(root=root_dir, batch_size=batch_size, device=device)

    # create model
    pixor = PIXOR().to(device)

    # create optimizer and scheduler objects
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, pixor.parameters()), lr=initial_learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], gamma=0.1)

    # train model
    history = train_model(pixor, optimizer, scheduler, data_loader, n_epochs=n_epochs)

    # save training history
    np.savez('history.npz', history=history)
