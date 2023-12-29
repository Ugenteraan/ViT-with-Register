'''Helper functions.
'''

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch

def calculate_accuracy(batch_predictions, batch_targets):
    '''Function to calculate the accuracy of predictions given the targets.
    '''

    # return (batch_predictions.argmax(dim=1) == batch_targets.reshape(-1, 1)).float().mean().cpu().numpy()
    num_data = batch_targets.size()[0]
    predicted = torch.argmax(batch_predictions, dim=1)
    correct_pred = torch.sum(predicted.reshape(-1) == batch_targets.reshape(-1))


    accuracy = (correct_pred/num_data)*100 #convert to percentage.

    return accuracy.item()



def plot_loss_acc(path, num_epoch,  train_accuracies, train_losses,
                    test_accuracies, test_losses, rank=None, epoch_step=1):
    '''
    Plot line graphs for the accuracies and loss at every epochs for both training and testing.
    '''

    fig = plt.figure(figsize=(20, 5))
    plt.clf()

    epochs = [x for x in range(0, num_epoch+1, epoch_step)]

    train_accuracy_df = pd.DataFrame({"Epochs":epochs, "Accuracy":train_accuracies, "Mode":['train']*len(epochs)})
    test_accuracy_df = pd.DataFrame({"Epochs":epochs, "Accuracy":test_accuracies, "Mode":['test']*len(epochs)})

    data = pd.concat([train_accuracy_df, test_accuracy_df])

    sns.lineplot(data=data, x='Epochs', y='Accuracy', hue='Mode')
    plt.title('Accuracy Graph')
    if not rank is None:
        plt.savefig(path+f'accuracy_epoch.png')
    else:
        plt.savefig(path+f'accuracy_epoch_rank-{rank}.png')

    plt.clf()


    train_loss_df = pd.DataFrame({"Epochs":epochs, "Loss":train_losses, "Mode":['train']*len(epochs)})
    test_loss_df = pd.DataFrame({"Epochs":epochs, "Loss":test_losses, "Mode":['test']*len(epochs)})

    data = pd.concat([train_loss_df, test_loss_df])

    sns.lineplot(data=data, x='Epochs', y='Loss', hue='Mode')
    plt.title('Loss Graph')

    if not rank is None:
        plt.savefig(path+f'loss_epoch.png')
    else:
        plt.savefig(path+f'loss_epoch_rank-{rank}.png')

    plt.close()

    return None


def get_opti_lr(optimizer):
    '''Returns the optimizer's current learning rate parameter.
    '''

    for param_group in optimizer.param_groups:
        return param_group['lr']


