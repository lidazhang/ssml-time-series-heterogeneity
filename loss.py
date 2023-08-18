import torch
import torch.nn as nn
from sklearn import metrics
import numpy as np

from utils import get_bin_custom, get_bin_log, CustomBins, LogBins, get_estimate_custom

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 0.1))) * 100

def print_metrics_regression(y_true, predictions, verbose=1):
    predictions = np.array(predictions)
    #predictions = np.max(predictions, axis=-1)#.flatten()
    y_true = np.array(y_true)

    y_true_bins = [get_bin_custom(x, CustomBins.nbins) for x in y_true]
    prediction_bins = [get_bin_custom(x, CustomBins.nbins) for x in predictions]
    # print()
    # print(333,np.shape(predictions),np.shape(y_true))
    # print(444,np.shape(prediction_bins),np.shape(y_true_bins))

    cf = metrics.confusion_matrix(y_true_bins, prediction_bins)
    if verbose:
        print("Custom bins confusion matrix:")
        print(cf)

    kappa = metrics.cohen_kappa_score(y_true_bins, prediction_bins,
                                      weights='linear')
    mad = metrics.mean_absolute_error(y_true, predictions)
    mse = metrics.mean_squared_error(y_true, predictions)
    mape = mean_absolute_percentage_error(y_true, predictions)

    if verbose:
        print("Mean absolute deviation (MAD) = {}".format(mad))
        print("Mean squared error (MSE) = {}".format(mse))
        print("Mean absolute percentage error (MAPE) = {}".format(mape))
        print("Cohen kappa score = {}".format(kappa))


def compute_loss(net: torch.nn.Module,
                 dataloader,#dataloader: torch.utils.data.DataLoader,
                 loss_function: torch.nn.Module,
                 device: torch.device = 'cpu') -> torch.Tensor:
    """Compute the loss of a network on a given dataset.
    Does not compute gradient.
    Parameters
    ----------
    net:
        Network to evaluate.
    dataloader:
        Iterator on the dataset.
    loss_function:
        Loss function to compute.
    device:
        Torch device, or :py:class:`str`.
    Returns
    -------
    Loss as a tensor with no grad.
    """
    running_loss = 0
    y_true_all, y_all, y_pred_all = [], [], []
    with torch.no_grad():
        for (x, y, y_true) in dataloader:
            x= torch.Tensor(x)
            y= torch.Tensor(y)
            y_true= torch.Tensor(y_true)
            netout = net(x.to(device)).cpu()
            running_loss += loss_function(y, netout)
            y_all.append(y)
            y_pred_all.append(netout)
            y_true_all.append(y_true)
            #print(x.size(),y.size(),y_true.size(),netout.size())

    y_true_all = torch.reshape(torch.stack(y_true_all), (-1,1))
    y_pred_all = torch.reshape(torch.stack(y_pred_all), (-1,10))
    predictions = np.array([get_estimate_custom(x, 10) for x in y_pred_all])
    print_metrics_regression(y_true_all, predictions)

    return running_loss / len(dataloader)


class OZELoss(nn.Module):
    """Custom loss for TRNSys metamodel.

    Compute, for temperature and consumptions, the intergral of the squared differences
    over time. Sum the log with a coeficient ``alpha``.

    .. math::
        \Delta_T = \sqrt{\int (y_{est}^T - y^T)^2}

        \Delta_Q = \sqrt{\int (y_{est}^Q - y^Q)^2}

        loss = log(1 + \Delta_T) + \\alpha \cdot log(1 + \Delta_Q)

    Parameters:
    -----------
    alpha:
        Coefficient for consumption. Default is ``0.3``.
    """

    def __init__(self, alpha: float = 0.3):
        super().__init__()

        self.alpha = alpha
        self.base_loss = nn.MultiLabelSoftMarginLoss() #nn.MSELoss() nn.CrossEntropyLoss()

    def forward(self,
                y_true: torch.Tensor,
                y_pred: torch.Tensor) -> torch.Tensor:
        """Compute the loss between a target value and a prediction.

        Parameters
        ----------
        y_true:
            Target value.
        y_pred:
            Estimated value.

        Returns
        -------
        Loss as a tensor with gradient attached.
        """
        # print(111,y_pred.size(),y_pred)
        # print(222,y_true.size(),y_true)
        # delta_Q = self.base_loss(y_pred[..., :-1], y_true[..., :-1])
        # delta_T = self.base_loss(y_pred[..., -1], y_true[..., -1])
        loss = self.base_loss(y_pred, y_true) #.long()

        return loss # torch.log(1 + delta_T) + self.alpha * torch.log(1 + delta_Q)
