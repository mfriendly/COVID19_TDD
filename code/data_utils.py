import argparse
import glob
import json
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from pandas import DatetimeIndex
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Subset

pastel_palette = sns.color_palette("pastel")
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=pastel_palette)
import scienceplots  # noqa

plt.style.use(["science", "no-latex"])


class DifferenceTransformer:

    def __init__(self):
        self.initial_values = None

    def fit_transform(self, df, column, days=1):
        self.initial_values = df[column].shift(days)
        return df[column].diff(days).ffill().bfill()

    def inverse_transform(self, transformed, original):
        return transformed.cumsum() + self.initial_values


class PercentageChangeTransformer:

    def fit_transform(self, df, column):
        return df[column].pct_change().ffill().bfill()

    def inverse_transform(self, transformed, original):
        inverse = [original[0]]
        for pct in transformed[1:]:
            inverse.append(inverse[-1] * (1 + pct))
        return np.array(inverse)


class LogTransformer:

    def fit_transform(self, df, column):
        return np.log1p(df[column].replace(0, np.nan)).ffill().bfill()

    def inverse_transform(self, transformed):
        return np.expm1(transformed)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_last_processed_number(path):
    files = glob.iglob(os.path.join(path, "*.pth"))
    max_int = -1
    nums = []
    for file_path in files:
        base_name = os.path.basename(file_path)
        number = base_name.split("-v")[1].replace(".pth", "")
        print("number", number)
        nums.append(int(number))
    if nums:
        max_int = max(max_int, max(nums))
    return max_int if max_int != -1 else 0


def save_last_processed_number(save_path, number):
    with open(save_path + "/last_processed.txt", "w") as file:
        file.write(str(number))


def retrieve_last_processed_give_lc_path(config, args):
    CKPT_SAVE = config["save_path"] + "/y_ckpt/"
    if not os.path.exists(CKPT_SAVE):
        os.makedirs(CKPT_SAVE)
    os.chmod(CKPT_SAVE, 0o700)
    last_processed_num = int(get_last_processed_number(CKPT_SAVE))
    last_processed_num = str(last_processed_num)
    C_PATH_num = int(last_processed_num) + 1
    save_last_processed_number(CKPT_SAVE, last_processed_num)
    L_PATH = CKPT_SAVE + f"bests-v{last_processed_num}.pth"
    C_PATH = CKPT_SAVE + f"bests-v{C_PATH_num}.pth"
    config["CKPT_SAVE"] = CKPT_SAVE
    config["L_PATH"] = L_PATH
    config["C_PATH"] = C_PATH
    return config, args


def count_param_and_update_config(config, model_):
    model_parameters = filter(lambda p: p.requires_grad, model_.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    config["PARAMS"] = params
    return config


def difference_from_previous_time(config,
                                  args,
                                  df,
                                  column_name="new_confirmed",
                                  abbr=None,
                                  days=1):
    date_column_name = "date"
    initial_values = df[[date_column_name, column_name
                         ]].set_index(date_column_name)[column_name].to_dict()
    start_date = datetime.strptime(args.test_start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.test_end_date, "%Y-%m-%d")
    initial_values = {
        key.strftime("%Y-%m-%d"): value
        for key, value in initial_values.items()
        if start_date <= key <= end_date
    }
    df[f"{column_name}_time_diff"] = df[column_name] - df[column_name].shift(
        days)
    df[f"{column_name}_time_diff"] = df[f"{column_name}_time_diff"].fillna(0.0)
    df[column_name] = df[f"{column_name}_time_diff"]
    return df, initial_values


def do_standard_scaler(config, args, df_train_val, df_test, cols):
    scaler = StandardScaler()
    df_train_val[cols] = scaler.fit_transform(df_train_val[cols])
    df_test[cols] = scaler.transform(df_test[cols])
    scaler_stats = {"mean": scaler.mean_, "std": scaler.scale_}
    return df_train_val, df_test, scaler_stats


def inverse_standard_scaler(config, args, scaled_values, scaler_stats):
    mean = scaler_stats["mean"]
    std = scaler_stats["std"]
    original_values = (scaled_values * std) + mean
    return original_values


def do_minmax_scaler(config, args, df_train_val, df_test, cols):
    print(f"==>> df_train_val.shape: {df_train_val.shape}")
    print(f"==>> df_test.shape: {df_test.shape}")
    scaler = MinMaxScaler()
    df_train_val[cols] = scaler.fit_transform(df_train_val[cols])
    df_test[cols] = scaler.transform(df_test[cols])
    scaler_stats = {"min": scaler.data_min_, "max": scaler.data_max_}
    return df_train_val, df_test, scaler_stats


def inverse_minmax_scaler(config, args, scaled_values, scaler_stats):
    min_val = scaler_stats["min"]
    max_val = scaler_stats["max"]
    original_values = scaled_values * (max_val - min_val) + min_val
    return original_values


def inverse_pct_change(pct_series, initial_value):
    original_values = [initial_value]
    for pct in pct_series:
        new_value = original_values[-1] * (1 + pct)
        original_values.append(new_value)
    return original_values


def inverse_log(NParray):
    np.expm1(NParray)


def split_rolling_scale(config, args, df, cols, test_boundary):
    print("args", args)
    initial_values = None
    df["date"] = pd.to_datetime(df["date"])
    df_train_val = df[df["date"] < args.test_start_date].reset_index(drop=True)
    df_test = df[df["date"] >= args.test_start_date].reset_index(drop=True)
    if args.replace_zero_to_nan:
        df_train_val["new_confirmed"] = df_train_val["new_confirmed"].replace(
            0.0, np.nan)
        df_train_val["new_confirmed"] = df_train_val["new_confirmed"].ffill(
        ).bfill()
        df_test["new_confirmed"] = df_test["new_confirmed"].replace(
            0.0, np.nan)
        df_test["new_confirmed"] = df_test["new_confirmed"].ffill().bfill()
    if args.rolling:
        rolling_window = args.rolling_window
        df_train_val["new_confirmed"] = (df_train_val["new_confirmed"].rolling(
            window=rolling_window).mean().ffill().bfill())
        df_test["new_confirmed"] = df_test["new_confirmed"].rolling(
            window=rolling_window).mean().ffill().bfill()
        df_train_val = df_train_val.iloc[rolling_window - 1:, :]
        df_test = df_test.iloc[rolling_window - 1:, :]
    df_test.to_csv(f"{args.save_path}/df_test_unscaled.csv", index=False)
    if args.diff:
        df_train_val, initial_values = difference_from_previous_time(
            config,
            args,
            df_train_val,
            "new_confirmed",
            abbr=args.abbr,
            days=1)
        df_test, initial_values = difference_from_previous_time(
            config, args, df_test, "new_confirmed", abbr=args.abbr, days=1)
    scaler_stats = {}
    if args.scaler == "standard":
        df_train_val, df_test, scaler_stats = do_standard_scaler(
            config, args, df_train_val, df_test, cols)
    elif args.scaler == "minmax":
        df_train_val, df_test, scaler_stats = do_minmax_scaler(
            config, args, df_train_val, df_test, cols)
    else:
        pass
    df_train_val["date"] = pd.to_datetime(df_train_val["date"])
    df_train = df_train_val[df_train_val["date"] <
                            args.val_start_date].reset_index(drop=True)
    df_val = df_train_val[df_train_val["date"] >=
                          args.val_start_date].reset_index(drop=True)
    return df_train, df_val, df_test, scaler_stats, initial_values


def reconstruct_series_to_array(
    config,
    args,
    differences,
    initial_values,
    start_date_,
    end_date_,
    date_format="%Y-%m-%d",
    abbr=None,
    days=1,
):
    initial_values
    future_steps = args.future_steps
    start_date = datetime.strptime(start_date_, date_format) if isinstance(
        start_date_, str) else start_date_
    start_date = start_date + timedelta(days=future_steps)
    end_date = datetime.strptime(end_date_, date_format) if isinstance(
        end_date_, str) else end_date_
    dates = pd.date_range(start_date, end_date)
    reconstructed_values = [
        initial_values[(start_date -
                        timedelta(days=days)).strftime(date_format)]
        for start_date in dates
    ]
    reconstructed_values
    for i in range(len(differences)):
        new_value = reconstructed_values[i] + differences[i][0]
        reconstructed_values[i] = new_value
    reconstructed_values = np.array(reconstructed_values).reshape(-1, 1)
    return reconstructed_values


def replace_negatives_and_interpolate(config, args, arr):

    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    arr = np.where(arr <= 0.0, np.nan, arr)
    nans, x = nan_helper(arr)
    arr[nans] = np.interp(x(nans), x(~nans), arr[~nans])
    return arr


def fill_missing_dates(df, date_column):
    df[date_column] = pd.to_datetime(df[date_column])
    full_date_range = pd.date_range(start=df[date_column].min(),
                                    end=df[date_column].max())
    df = df.set_index(date_column).reindex(full_date_range).reset_index()
    df = df.rename(columns={"index": date_column})
    df = df.fillna(np.nan)
    return df


def prepare_test_loader(args, test_dataset, device):
    if args.eval_stride == 1:
        pass
    else:
        test_indices = np.array(
            [i for i in range(len(test_dataset)) if i % args.eval_stride == 1])
        test_dataset = Subset(test_dataset, test_indices)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        generator=torch.Generator(device="cpu"),
        drop_last=True,
    )
    return test_loader


def format_date(unix_timestamp):
    if isinstance(unix_timestamp, int):
        try:
            dt = datetime.fromtimestamp(unix_timestamp)
        except Exception:
            return None
    elif isinstance(unix_timestamp, datetime.datetime):
        dt = unix_timestamp
    else:
        return None
    return f"{dt.year}-{dt.month:02d}-{dt.day:02d}"


def process_test_output(
    config,
    args,
    initial_values_date_dict,
    test_loader,
    model,
    scaler_stats,
    criterion,
    device,
    test_df,
):
    actual_all = {}
    pred_all = {}
    total_loss = 0.0
    model.eval()
    for idx, (x, y, past_date_tup, future_date_tup) in enumerate(test_loader):
        past_date_start = format_date(past_date_tup[0].item())
        format_date(past_date_tup[1].item())
        future_date_start = format_date(future_date_tup[0].item())
        future_date_end = format_date(future_date_tup[1].item())
        x = x.to(
            device,
            dtype=torch.float64 if args.dtype == "double" else torch.float32)
        y = y.to(
            device,
            dtype=torch.float64 if args.dtype == "double" else torch.float32)
        prediction = model(x, y, args.future_steps,
                           0.0).cpu().detach().numpy().flatten().reshape(
                               -1, 1)
        print(f"==>> prediction.shape: {prediction.shape}")
        y = y.cpu().detach().numpy().flatten()
        csv_path = f"{args.save_path}/df_test_unscaled.csv"
        print("csv_path                           ", csv_path)

        def get_values_in_date_range(csv_path,
                                     start_date,
                                     end_date,
                                     value_column="new_confirmed"):
            df = pd.read_csv(csv_path)
            df["date"] = pd.to_datetime(df["date"])
            mask = (df["date"] >= start_date) & (df["date"] <= end_date)
            filtered_df = df.loc[mask]
            values = filtered_df["new_confirmed"]
            values_array = values.to_numpy()
            return values_array

        actuals = get_values_in_date_range(csv_path, future_date_start,
                                           future_date_end, "new_confirmed")
        actuals_np = np.array(actuals).reshape(-1, 1)
        print(f"==>> actuals_np.shape: {actuals_np.shape}")

        def inverse_transform_predictions_StandardScaler(
                config, args, predictions_to_scale, abbr, scaler_stats_dict):
            mean = scaler_stats_dict["mean"]
            std = scaler_stats_dict["std"]
            inverse_transformed_predictions = (predictions_to_scale *
                                               std) + mean
            return inverse_transformed_predictions

        def inverse_transform_predictions_MinMaxScaler(config, args,
                                                       predictions_to_scale,
                                                       abbr,
                                                       scaler_stats_dict):
            min_val = scaler_stats_dict["min"]
            max_val = scaler_stats_dict["max"]
            """"""
            inverse_transformed_predictions = predictions_to_scale * (
                max_val - min_val) + min_val
            return inverse_transformed_predictions

        if args.scaler == "standard":
            pred_final = inverse_transform_predictions_StandardScaler(
                config, args, prediction, args.abbr, scaler_stats)
            actual_final = actuals_np
        elif args.scaler == "minmax":
            pred_final = inverse_transform_predictions_MinMaxScaler(
                config, args, prediction, args.abbr, scaler_stats)
            actual_final = actuals_np
        else:
            pred_final = prediction
            actual_final = actuals_np
        if args.diff:
            initial_test_value = initial_values_date_dict
            combined_tar = reconstruct_series_to_array(
                config,
                args,
                pred_final,
                initial_test_value,
                past_date_start,
                future_date_end,
                "%Y-%m-%d",
                args.abbr,
            )
            print(f"==>> combined_tar.shape: {combined_tar.shape}")
            pred_final = combined_tar.reshape(-1, 1)
        else:
            pred_final = pred_final.reshape(-1, 1)
        if False:
            pred_final = replace_negatives_and_interpolate(
                config, args, pred_final)
        actual_all[future_date_start] = actual_final.tolist()
        pred_all[future_date_start] = pred_final.tolist()
        loss = criterion(
            torch.tensor(pred_final).to(device,
                                        dtype=torch.float64 if args.dtype
                                        == "double" else torch.float32),
            torch.tensor(actual_final).to(device,
                                          dtype=torch.float64 if args.dtype
                                          == "double" else torch.float32),
        )
        loss = torch.sqrt(loss)
        total_loss += loss.cpu().item()
    if args.eval_stride == 1:
        print("args.eval_stride ==1", )
        avg_loss = total_loss
    else:
        avg_loss = total_loss / len(test_loader)
    step_str = "".join([str(i) for i in args.horizon_steps])
    args.eval_step_str = f"stride{args.eval_stride}-ws{args.ws}-steps{step_str}"
    a = f"{args.save_path}/{args.eval_type}/{args.eval_step_str}"
    os.makedirs(a, exist_ok=True)
    with open(
            f"{args.save_path}/{args.eval_type}/{args.eval_step_str}/pred_all.json",
            "w") as f:
        json.dump(pred_all, f)
    with open(
            f"{args.save_path}/{args.eval_type}/{args.eval_step_str}/actual_all.json",
            "w") as f:
        json.dump(actual_all, f)
    args.eval_dates = list(pred_all.keys())
    os.makedirs(f"x_data_json/{args.nation}/eval_dates", exist_ok=True)
    with open(
            f"x_data_json/{args.nation}/eval_dates/{args.eval_step_str}.json",
            "w") as f:
        json.dump(args.eval_dates, f)
    return actual_all, pred_all, avg_loss


def plot_datetime_multiple_starts(args,
                                  actual_dict,
                                  pred_dict,
                                  output_file="pred_and_actual.png"):
    actual_dates, actual_values = [], []
    pred_dates, pred_values = [], []
    for date_str, nested_values in actual_dict.items():
        date_start = datetime.strptime(date_str, "%Y-%m-%d")
        avg_actual_values = [
            sum(sublist) / len(sublist) for sublist in nested_values
        ]
        for i, value in enumerate(avg_actual_values):
            actual_dates.append(date_start + timedelta(days=i))
            actual_values.append(value)
    for date_str, nested_values in pred_dict.items():
        date_start = datetime.strptime(date_str, "%Y-%m-%d")
        avg_pred_values = [
            sum(sublist) / len(sublist) for sublist in nested_values
        ]
        for i, value in enumerate(avg_pred_values):
            pred_dates.append(date_start + timedelta(days=i))
            pred_values.append(value)
    actual_df = pd.DataFrame({
        "Date": actual_dates,
        "Actual Values": actual_values
    })
    pred_df = pd.DataFrame({
        "Date": pred_dates,
        "Predicted Values": pred_values
    })
    plt.figure(figsize=(10, 6))
    plt.plot(
        actual_df["Date"],
        actual_df["Actual Values"],
        marker="o",
        label="Actual",
        color="blue",
    )
    plt.plot(
        pred_df["Date"],
        pred_df["Predicted Values"],
        marker="x",
        label="Predicted",
        color="red",
    )
    plt.xlabel("Date")
    plt.ylabel("Values")
    plt.title("Actual vs Predicted Values Over Time")
    plt.xticks(rotation=45)
    plt.legend()
    plt.savefig(output_file)
    plt.close()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def get_nth_value_from_columns(csv_path, location, columns):
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"keyword": "feature"})
    if location > len(df):
        raise ValueError(
            f"Location {location} exceeds DataFrame length {len(df)}")
    for column in columns:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' does not exist in DataFrame.")
    nth_values = {}
    for column in columns:
        col_values = df.loc[location, column]
        nth_values[column] = col_values
    return nth_values

def serialize_dict(data):
    if isinstance(data, dict):
        return {k: serialize_dict(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [serialize_dict(item) for item in data]
    elif isinstance(data, np.ndarray):
        return serialize_dict(data.tolist())
    elif isinstance(data, torch.device):
        return str(data)
    if isinstance(data, torch.dtype):
        return str(data)
    elif isinstance(
            data,
        (
            np.int_,
            np.intc,
            np.intp,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    ):
        return int(data)
    elif isinstance(data, (np.float_, np.float16, np.float32, np.float64)):
        return float(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, DatetimeIndex):
        return data.strftime("%Y-%m-%d").tolist()
    return data
