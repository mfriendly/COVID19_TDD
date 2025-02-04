import os
import random
import numpy as np
import torch


def MAE(predictions, targets):
    r = np.abs(predictions - targets).mean()
    return r


def MAPE_(predictions, targets):
    return np.mean(np.abs((targets - predictions) / targets)) * 100


def RMSE_(predictions, targets):
    return np.sqrt(((predictions - targets)**2).mean())


def NRMSE_(predictions, targets):
    rmse = np.sqrt(((predictions - targets)**2).mean())
    range_val = targets.max() - targets.min()
    print("def NRMSE traiingutils v4 - range_val", range_val)
    return rmse / range_val


def IA_(predictions, targets):
    numerator = np.sum((targets - predictions)**2)
    y_mean = np.mean(targets)
    denominator = np.sum(
        (np.abs(predictions - y_mean) + np.abs(targets - y_mean))**2)
    return numerator / denominator


def RRSE_(predictions, targets):
    numerator = np.sqrt(np.sum((targets - predictions)**2))
    y_mean = np.mean(targets)
    denominator = np.sqrt(np.sum((targets - y_mean)**2))
    return numerator / denominator


def R2_(predictions, targets):
    y_mean = np.mean(targets)
    numerator = np.sum((predictions - targets)**2)
    denominator = np.sum((targets - y_mean)**2)
    return 1 - (numerator / denominator)


def sMAPE_(predictions, targets):
    return np.mean(
        np.abs(predictions - targets) /
        ((np.abs(predictions) + np.abs(targets)) / 2))


def NRMSE_(predictions, targets):
    mse = np.mean((targets - predictions)**2)
    range_val = targets.max() - targets.min()
    return np.sqrt(mse) / range_val


def RAE_(predictions, targets):
    numerator = np.sum(np.abs(targets - predictions))
    y_mean = np.mean(targets)
    denominator = np.sum(np.abs(targets - y_mean))
    return numerator / denominator


def RMSPE_(predictions, targets):
    return np.sqrt(np.mean(((targets - predictions) / targets)**2))


def NMAE_(predictions, targets):
    return np.sum(np.abs(targets - predictions)) / np.sum(targets)


def CORR_(pred, true):
    true = np.atleast_1d(true)
    pred = np.atleast_1d(pred)
    true_mean = true.mean() if true.size > 0 else 0
    pred_mean = pred.mean() if pred.size > 0 else 0
    u = ((true - true_mean) * (pred - pred_mean)).sum()
    d = np.sqrt(((true - true_mean)**2).sum() * ((pred - pred_mean)**2).sum())
    if d == 0:
        return 0
    corr = u / d
    return corr if np.isscalar(corr) else corr.mean() if corr.size > 0 else 0


class EarlyStopping:

    def __init__(self, patience=7, delta=0, min_epochs=3):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved
            delta (float): Minimum change in monitored quantity to qualify as an improvement
            min_epochs (int): Minimum number of epochs before early stopping can trigger
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")
        self.delta = delta
        self.min_epochs = min_epochs
        self.current_epoch = 0

    def __call__(self, val_loss, model, save_path):
        self.current_epoch += 1
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_path)
        elif score < self.best_score + self.delta:
            if self.current_epoch >= self.min_epochs:
                self.counter += 1
                print(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
                if self.counter >= self.patience:
                    self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, save_path):
        """Save model when validation loss decreases."""
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), os.path.join(save_path,
                                                    "checkpoint.pth"))
        self.val_loss_min = val_loss


def set_seed(seed=42, make_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if make_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def correlation_loss(predictions, targets):
    predictions = predictions.squeeze(-1)
    targets = targets.squeeze(-1)
    pred_flat = predictions.view(-1)
    target_flat = targets.view(-1)
    pred_mean = torch.mean(pred_flat)
    target_mean = torch.mean(target_flat)
    pred_std = torch.std(pred_flat)
    target_std = torch.std(target_flat)
    correlation = torch.mean(
        (pred_flat - pred_mean) *
        (target_flat - target_mean)) / (pred_std * target_std + 1e-8)
    corr_loss = 1 - correlation
    return corr_loss


def save_range_target(targets):
    range_val = targets.max() - targets.min()
    return range_val


def window_evaluation_dict(config, args, pred_dict, actual_dict):
    actual_dates, actual_values = [], []
    pred_dates, pred_values = [], []
    start_dates = list(actual_dict.keys())
    metric_dict_ = {}
    for index, start_date in enumerate(start_dates):
        args.index = index
        pred_list = pred_dict[start_date]
        actual_list = actual_dict[start_date]
        metrics = window_evaluation(config, args, actual_list, pred_list)
        print("metrics", metrics)
        metric_dict_[start_date] = metrics
    import copy
    safe_dict = copy.deepcopy(metric_dict_)

    def calculate_average_metrics(metrics_data):
        averaged_metrics = {}
        for metric_name in metrics_data[next(iter(metrics_data))].keys():
            metric_values = []
            for date, metrics in metrics_data.items():
                metric_values.append(metrics[metric_name])
            averaged_metrics[metric_name] = list(np.mean(metric_values,
                                                         axis=0))
        return averaged_metrics

    average_metrics_dict = calculate_average_metrics(safe_dict)
    for metric, avg_values in average_metrics_dict.items():
        print(f"{metric}: {avg_values}")
    return safe_dict, average_metrics_dict


def window_evaluation_dict_by_TYPE(config, args, pred_dict, actual_dict):
    actual_dates, actual_values = [], []
    pred_dates, pred_values = [], []
    start_dates = list(actual_dict.keys())
    metric_dict_ = {}
    for index, start_date in enumerate(start_dates):
        args.index = index
        pred_list = pred_dict[start_date]
        actual_list = actual_dict[start_date]
        metrics = window_evaluation_by_TYPE(config, args, actual_list,
                                            pred_list)
        print("metrics", metrics)
        metric_dict_[start_date] = metrics
    import copy
    safe_dict = copy.deepcopy(metric_dict_)

    def calculate_average_metrics(metrics_data):
        averaged_metrics = {}
        for metric_name in metrics_data[next(iter(metrics_data))].keys():
            metric_values = []
            for date, metrics in metrics_data.items():
                metric_values.append(metrics[metric_name])
            averaged_metrics[metric_name] = list(np.mean(metric_values,
                                                         axis=0))
        return averaged_metrics

    average_metrics_dict = calculate_average_metrics(safe_dict)
    for metric, avg_values in average_metrics_dict.items():
        print(f"{metric}: {avg_values}")
    return safe_dict, average_metrics_dict


def window_evaluation_by_TYPE(config, args, actual, predictions):
    actual = np.array(actual).reshape(1, -1)
    predictions = np.array(predictions).reshape(1, -1)
    mode = args.mode = args.eval_type
    if args.eval_type != "window":
        pass
    else:
        pass
        print("args.horizon_steps", args.horizon_steps)

    def get_slice(h, mode):
        if mode == "point":
            return slice(h, h + 1)
        elif mode == "cumulative":
            return slice(0, h + 1)
        elif mode == "window":
            ws = args.ws
            return slice(h - ws, h)

    metrics = {
        "RMSE": [
            RMSE_(predictions[:, get_slice(h, mode)],
                  actual[:, get_slice(h, mode)]) for h in args.horizon_steps
        ],
    }
    return metrics


def correlation_coefficient_loss(predictions, targets):
    mean_x = torch.mean(predictions, dim=2, keepdim=True)
    mean_y = torch.mean(targets, dim=2, keepdim=True)
    vx = predictions - mean_x
    vy = targets - mean_y
    cost = torch.sum(vx * vy, dim=2) / (torch.sqrt(torch.sum(vx**2, dim=2)) *
                                        torch.sqrt(torch.sum(vy**2, dim=2)))
    return torch.mean(cost)
