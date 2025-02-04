import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from data_utils import *
from training_utils import *
import seaborn as sns
import scienceplots  # noqa

pastel_palette = sns.color_palette("pastel")
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=pastel_palette)
plt.style.use(["science", "no-latex"])



def timestamp_to_unix(timestamp):
    try:
        return timestamp.value // 10**9
    except:
        return pd.to_datetime(timestamp).value // 10**9


def compute_exog_scaling_stats(config, args, list_of_exog_dicts, abbr, abbr2id,
                               df_start_date, df_end_date):
    """
    Compute scaling statistics for exogenous features and handle JSON storage.
    Handles both string and datetime inputs for dates.
    """
    if isinstance(df_start_date, str):
        df_start_date = pd.to_datetime(df_start_date)
    if isinstance(df_end_date, str):
        df_end_date = pd.to_datetime(df_end_date)
    scaling_dir = os.path.join(args.save_path, "scaling_stats")
    os.makedirs(scaling_dir, exist_ok=True)
    scaling_id = f"{args.nation}_{abbr}_{df_start_date.strftime('%Y%m%d')}_{df_end_date.strftime('%Y%m%d')}"
    json_path = os.path.join(scaling_dir, f"scaler_stats_{scaling_id}.json")
    if os.path.exists(json_path):
        print(f"Loading existing scaling statistics from {json_path}")
        with open(json_path, "r") as f:
            return json.load(f)
    print(f"Computing new scaling statistics...")
    scaler_stats = {}
    for exog_dict in list_of_exog_dicts:
        exog_values = None
        if args.exog_type == "region":
            region = exog_dict["feature"]
            node_id = str(abbr2id[region]).zfill(2)
            path = f"../data/x_data_epidemiology/{args.nation}/{node_id}_{region}.csv"
            df_exog = pd.read_csv(path, usecols=["date", "new_confirmed"])
            df_exog["date"] = pd.to_datetime(df_exog["date"])
            df_exog = df_exog[(df_exog["date"] >= df_start_date)
                              & (df_exog["date"] <= df_end_date)].reset_index(
                                  drop=True)
            exog_values = df_exog["new_confirmed"].to_numpy()
        elif args.exog_type == "all":
            if exog_dict["type"].lower() != "region":
                exog_key = exog_dict["type"]
                feature = exog_dict["feature"]
                node_id = str(abbr2id[abbr]).zfill(2)
                path = f"../data/x_data_{exog_key}/{args.nation}/{args.nation}_{abbr}.csv"
                df_exog = pd.read_csv(path, usecols=["date", feature])
                df_exog[feature] = df_exog[feature].fillna(
                    df_exog[feature].mean())
                df_exog["date"] = pd.to_datetime(df_exog["date"])
                df_exog = df_exog[(df_exog["date"] >= df_start_date) & (
                    df_exog["date"] <= df_end_date)].reset_index(drop=True)
                exog_values = df_exog[feature].to_numpy()
            else:
                region = exog_dict["feature"]
                node_id = str(abbr2id[region]).zfill(2)
                path = f"../data/x_data_epidemiology/{args.nation}/{node_id}_{region}.csv"
                df_exog = pd.read_csv(path, usecols=["date", "new_confirmed"])
                df_exog["date"] = pd.to_datetime(df_exog["date"])
                df_exog = df_exog[(df_exog["date"] >= df_start_date) & (
                    df_exog["date"] <= df_end_date)].reset_index(drop=True)
                exog_values = df_exog["new_confirmed"].to_numpy()
        if exog_values is not None and len(exog_values) > 0:
            scaler = StandardScaler()
            exog_values_reshaped = exog_values.reshape(-1, 1)
            scaler.fit(exog_values_reshaped)
            scaler_stats[exog_dict["feature"]] = {
                "mean": float(scaler.mean_[0]),
                "std": float(scaler.scale_[0]),
            }
        else:
            print(
                f"Warning: No valid data found for feature {exog_dict['feature']}"
            )
    with open(json_path, "w") as f:
        json.dump(scaler_stats, f, indent=4)
    print(f"Saved scaling statistics to {json_path}")
    return scaler_stats


def load_scaler_stats(args, abbr, df_start_date, df_end_date):
    """
    Helper function to load existing scaler statistics.
    Handles both string and datetime inputs for dates.
    """
    if isinstance(df_start_date, str):
        df_start_date = pd.to_datetime(df_start_date)
    if isinstance(df_end_date, str):
        df_end_date = pd.to_datetime(df_end_date)
    scaling_dir = os.path.join(args.save_path, "scaling_stats")
    scaling_id = f"{args.nation}_{abbr}_{df_start_date.strftime('%Y%m%d')}_{df_end_date.strftime('%Y%m%d')}"
    json_path = os.path.join(scaling_dir, f"scaler_stats_{scaling_id}.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    return None


def handle_dates(date_value):
    """Helper function to convert string dates to datetime if necessary"""
    if isinstance(date_value, str):
        return pd.to_datetime(date_value)
    return date_value


class windowDatasetmulti(Dataset):

    def __init__(
        self,
        config,
        args,
        df,
        df_start_date,
        df_end_date,
        abbr,
        abbr2id,
        input_window=7,
        output_window=7,
        stride=1,
        list_of_exog_dicts=None,
        scaler_stats=None,
    ):
        self.args = args
        self.df_start_date = handle_dates(df_start_date)
        self.df_end_date = handle_dates(df_end_date)
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= self.df_start_date)
                & (df["date"] <= self.df_end_date)].reset_index(drop=True)
        y = df["new_confirmed"].to_numpy().reshape(-1, 1)
        self.dates = df["date"].tolist()
        num_time_steps, _ = y.shape
        num_samples = (num_time_steps - input_window -
                       output_window) // stride + 1
        num_features = 1 + len(list_of_exog_dicts) if args.CFS else 1
        X = np.zeros([num_samples, input_window, num_features])
        Y = np.zeros([num_samples, output_window, 1])
        self.future_dates = []
        self.past_dates = []
        for i in np.arange(num_samples):
            start_x = stride * i
            end_x = start_x + input_window
            X[i, :, 0] = y[start_x:end_x, 0]
            self.past_dates.append(self.dates[start_x:end_x])
            start_y = stride * i + input_window
            end_y = start_y + output_window
            Y[i, :, 0] = y[start_y:end_y, 0]
            self.future_dates.append(self.dates[start_y:end_y])
        if args.CFS and list_of_exog_dicts:
            if scaler_stats is None:
                print(
                    "Warning: No scaler_stats provided. Loading from saved file..."
                )
                scaling_dir = Path(args.save_path) / "scaling_stats"
                scaling_id = f"{args.nation}_{abbr}_{self.df_start_date.strftime('%Y%m%d')}_{self.df_end_date.strftime('%Y%m%d')}"
                json_path = scaling_dir / f"scaler_stats_{scaling_id}.json"
                if json_path.exists():
                    with open(json_path, "r") as f:
                        scaler_stats = json.load(f)
                    print(f"Loaded scaling statistics from {json_path}")
                else:
                    print(
                        f"Warning: No scaling statistics found at {json_path}")
            for j, exog_dict in enumerate(list_of_exog_dicts, start=1):
                exog_values = None
                feature_name = exog_dict["feature"]
                try:
                    if args.exog_type == "region":
                        region = feature_name
                        lag = exog_dict["lag"]
                        node_id = str(abbr2id[region]).zfill(2)
                        path = f"../data/x_data_epidemiology/{args.nation}/{node_id}_{region}.csv"
                        df_exog = pd.read_csv(
                            path, usecols=["date", "new_confirmed"])
                        df_exog["date"] = pd.to_datetime(df_exog["date"])
                        df_exog = df_exog[
                            (df_exog["date"] >= self.df_start_date) &
                            (df_exog["date"] <= self.df_end_date)].reset_index(
                                drop=True)
                        exog_values = df_exog["new_confirmed"].to_numpy()
                    elif args.exog_type == "all":
                        if exog_dict["type"].lower() != "region":
                            exog_key = exog_dict["type"]
                            feature = feature_name
                            lag = exog_dict["lag"]
                            node_id = str(abbr2id[abbr]).zfill(2)
                            path = f"../data/x_data_{exog_key}/{args.nation}/{args.nation}_{abbr}.csv"
                            df_exog = pd.read_csv(path,
                                                  usecols=["date", feature])
                            df_exog[feature] = df_exog[feature].fillna(
                                df_exog[feature].mean())
                            df_exog["date"] = pd.to_datetime(df_exog["date"])
                            df_exog = df_exog[
                                (df_exog["date"] >= self.df_start_date)
                                & (df_exog["date"] <= self.df_end_date
                                   )].reset_index(drop=True)
                            exog_values = df_exog[feature].to_numpy()
                        else:
                            region = feature_name
                            lag = exog_dict["lag"]
                            node_id = str(abbr2id[region]).zfill(2)
                            path = f"../data/x_data_epidemiology/{args.nation}/{node_id}_{region}.csv"
                            df_exog = pd.read_csv(
                                path, usecols=["date", "new_confirmed"])
                            df_exog["date"] = pd.to_datetime(df_exog["date"])
                            df_exog = df_exog[
                                (df_exog["date"] >= self.df_start_date)
                                & (df_exog["date"] <= self.df_end_date
                                   )].reset_index(drop=True)
                            exog_values = df_exog["new_confirmed"].to_numpy()
                    if exog_values is not None:
                        if scaler_stats is not None and feature_name in scaler_stats:
                            mean = scaler_stats[feature_name]["mean"]
                            std = scaler_stats[feature_name]["std"]
                            exog_values = (exog_values - mean) / std
                            print(
                                f"Applied pre-computed scaling to feature {feature_name}"
                            )
                        else:
                            print(
                                f"Warning: No scaling stats found for {feature_name}, fitting new scaler"
                            )
                            scaler = StandardScaler()
                            exog_values = scaler.fit_transform(
                                exog_values.reshape(-1, 1)).reshape(-1)
                        if len(exog_values) < input_window:
                            padding_length = input_window - len(exog_values)
                            exog_values = np.pad(
                                exog_values,
                                (padding_length, 0),
                                "constant",
                                constant_values=0,
                            )
                        for i in np.arange(num_samples):
                            lagged_start_x = max(0, stride * i - lag)
                            lagged_end_x = min(len(exog_values),
                                               lagged_start_x + input_window)
                            current_window = exog_values[
                                lagged_start_x:lagged_end_x]
                            if len(current_window) < input_window:
                                padding_length = input_window - len(
                                    current_window)
                                current_window = np.pad(
                                    current_window,
                                    (padding_length, 0),
                                    "constant",
                                    constant_values=0,
                                )
                            X[i, :, j] = current_window
                except Exception as e:
                    print(f"Error processing feature {feature_name}: {str(e)}")
                    for i in np.arange(num_samples):
                        X[i, :, j] = np.zeros(input_window)
        self.x = X
        self.y = Y
        self.len = len(X)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        start_date_unix = timestamp_to_unix(self.past_dates[idx][0])
        end_date_unix = timestamp_to_unix(self.past_dates[idx][-1])
        future_start_unix = timestamp_to_unix(self.future_dates[idx][0])
        future_end_unix = timestamp_to_unix(self.future_dates[idx][-1])
        past_dates_tuple = (start_date_unix, end_date_unix)
        future_dates_tuple = (future_start_unix, future_end_unix)
        return self.x[idx], self.y[idx], past_dates_tuple, future_dates_tuple


def slicing_by_date_dalstm(config, args, df):
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    train_start = "2010-01-01"
    train_end = "2018-04-30"
    val_start = "2018-05-01"
    val_end = "2019-04-30"
    test_start = "2019-05-01"
    print("test_start", test_start)
    test_end = "2021-10-31"
    print("test_end", test_end)
    trainval_set = df.loc[:val_end]
    train_set = df.loc[train_start:train_end]
    val_set = df.loc[val_start:val_end]
    test_set = df.loc[test_start:test_end]
    return train_set, val_set, test_set, trainval_set


def initialize_optimizer(model, lr):
    return optim.Adam(model.parameters(), betas=(0.9, 0.99), eps=0.01, lr=lr)


class ModelFactory:

    @staticmethod
    def initialize_model(config, args, device):
        args.MODEL_only = args.MODEL.split("-")[0]
        print("args.MODEL_only", args.MODEL_only)
        if args.MODEL_only.lower() == "SimpleLSTM".lower():
            return ModelFactory._initialize_SimpleLSTM(config, args, device)

        elif args.MODEL_only.lower() == "TriAttLSTM".lower():
            return ModelFactory._initialize_TriAttLSTM(config, args, device)
        else:
            raise ValueError(f"Unknown model: {args.MODEL}")

    @staticmethod
    def _initialize_SimpleLSTM(config, args, device):
        from SimpleRNN.model import SimpleLSTM
        model = SimpleLSTM(args)
        return model.to(
            device,
            dtype=torch.float64 if args.dtype == "double" else torch.float32)

    @staticmethod
    def _initialize_TriAttLSTM(config, args, device):
        from forecaster.model import TriAttLSTM
        model = TriAttLSTM(args)
        return model.to(
            device,
            dtype=torch.float64 if args.dtype == "double" else torch.float32)


def load_checkpoint(model, optimizer, config, device):
    try:
        checkpoint = torch.load(config["L_PATH"],
                                map_location="cpu",
                                weights_only=True)
        if args.dtype == "double":
            for k, v in checkpoint["modelstate_dict"].items():
                checkpoint["modelstate_dict"][k] = v.double()
        model.load_state_dict(checkpoint["modelstate_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"■■■Loaded L_PATH{config['L_PATH']}")
    except Exception as e:
        print(f"Error loading checkpoint from {config['L_PATH']}: {e}")
        try:
            checkpoint = torch.load(config["C_PATH"],
                                    map_location="cpu",
                                    weights_only=True)
            if args.dtype == "double":
                for k, v in checkpoint["modelstate_dict"].items():
                    checkpoint["modelstate_dict"][k] = v.double()
            model.load_state_dict(checkpoint["modelstate_dict"], strict=False)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = 0
            print(f"■■■Loaded C_PATH{config['C_PATH']}")
        except Exception as e:
            print(f"Error loading checkpoint from {config['C_PATH']}: {e}")
            start_epoch = 0
    return config, start_epoch, model, optimizer


def initialize_model(config, args, device):
    model = ModelFactory.initialize_model(config, args, device)
    model.train()
    return model


def select_features_combined(config, args, df_train, df_val, df_test, abbr,
                             abbr2id, list_of_exog_dicts):
    filtered_exog_dicts = list_of_exog_dicts
    if not args.CFS:
        feature_info = {
            "num_features": 1,
            "all_features": [],
            "selected_features": [],
            "removed_features": [],
        }
    else:
        df_combined = pd.concat([df_train, df_val
                                 ]).sort_values("date").reset_index(drop=True)
        temp_exog_features = {}
        all_features = [
            f"{d['type']}_{d['feature']}"
            if "type" in d else f"region_{d['feature']}"
            for d in list_of_exog_dicts
        ]
        for exog_dict in list_of_exog_dicts:
            if args.exog_type == "region":
                region = exog_dict["feature"]
                node_id = str(abbr2id[region]).zfill(2)
                path = f"../data/x_data_epidemiology/{args.nation}/{node_id}_{region}.csv"
                df_exog = pd.read_csv(path, usecols=["date", "new_confirmed"])
                df_exog["date"] = pd.to_datetime(df_exog["date"])
                df_exog = df_exog[(df_exog["date"] >= min(df_combined["date"]))
                                  & (df_exog["date"] <= max(
                                      df_combined["date"]))].reset_index(
                                          drop=True)
                feature_key = f"region_{region}"
                temp_exog_features[feature_key] = df_exog[
                    "new_confirmed"].values
            elif args.exog_type == "all":
                if exog_dict["type"].lower() != "region":
                    exog_key = exog_dict["type"]
                    feature = exog_dict["feature"]
                    node_id = str(abbr2id[abbr]).zfill(2)
                    path = f"../data/x_data_{exog_key}/{args.nation}/{args.nation}_{abbr}.csv"
                    df_exog = pd.read_csv(path, usecols=["date", feature])
                    df_exog[feature] = df_exog[feature].fillna(
                        df_exog[feature].mean())
                    df_exog["date"] = pd.to_datetime(df_exog["date"])
                    df_exog = df_exog[
                        (df_exog["date"] >= min(df_combined["date"]))
                        & (df_exog["date"] <= max(df_combined["date"])
                           )].reset_index(drop=True)
                    feature_key = f"{exog_key}_{feature}"
                    temp_exog_features[feature_key] = df_exog[feature].values
                else:
                    region = exog_dict["feature"]
                    node_id = str(abbr2id[region]).zfill(2)
                    path = f"../data/x_data_epidemiology/{args.nation}/{node_id}_{region}.csv"
                    df_exog = pd.read_csv(path,
                                          usecols=["date", "new_confirmed"])
                    df_exog["date"] = pd.to_datetime(df_exog["date"])
                    df_exog = df_exog[
                        (df_exog["date"] >= min(df_combined["date"]))
                        & (df_exog["date"] <= max(df_combined["date"])
                           )].reset_index(drop=True)
                    feature_key = f"region_{region}"
                    temp_exog_features[feature_key] = df_exog[
                        "new_confirmed"].values
        feature_keys = list(temp_exog_features.keys())
        features_to_remove = set()
        for i in range(len(feature_keys)):
            for j in range(i + 1, len(feature_keys)):
                if i not in features_to_remove and j not in features_to_remove:
                    array1 = temp_exog_features[feature_keys[i]]
                    array2 = temp_exog_features[feature_keys[j]]
                    max_length = max(len(array1), len(array2))
                    if len(array1) < max_length:
                        array1 = np.pad(
                            array1,
                            (0, max_length - len(array1)),
                            mode="constant",
                            constant_values=0,
                        )
                    if len(array2) < max_length:
                        array2 = np.pad(
                            array2,
                            (0, max_length - len(array2)),
                            mode="constant",
                            constant_values=0,
                        )
                    corr = np.corrcoef(array1, array2)[0, 1]
                    if abs(corr) >= args.threshold2:
                        features_to_remove.add(j)
        filtered_exog_dicts = [
            dict_ for i, dict_ in enumerate(list_of_exog_dicts)
            if i not in features_to_remove
        ]
        num_features = 1 + len(filtered_exog_dicts)
        selected_features = [
            all_features[i] for i in range(len(list_of_exog_dicts))
            if i not in features_to_remove
        ]
        removed_features = [all_features[i] for i in features_to_remove]
        feature_info = {
            "num_features": num_features,
            "all_features": all_features,
            "selected_features": selected_features,
            "removed_features": removed_features,
        }
        os.makedirs(args.save_path, exist_ok=True)
        feature_file = os.path.join(args.save_path, f"feature_info.json")
        with open(feature_file, "w") as f:
            json.dump(feature_info, f, indent=4)
    args.in_channels = feature_info["num_features"]
    if 1:
        scaler_stats = compute_exog_scaling_stats(
            config=config,
            args=args,
            list_of_exog_dicts=list_of_exog_dicts,
            abbr=abbr,
            abbr2id=abbr2id,
            df_start_date=args.train_start_date,
            df_end_date=args.train_end_date,
        )
        scaler_stats = load_scaler_stats(
            args=args,
            abbr=abbr,
            df_start_date=args.train_start_date,
            df_end_date=args.train_end_date,
        )
    train_dataset = windowDatasetmulti(
        config=config,
        args=args,
        df=df_train,
        df_start_date=args.train_start_date,
        df_end_date=args.train_end_date,
        abbr=abbr,
        abbr2id=abbr2id,
        input_window=args.past_steps,
        output_window=args.future_steps,
        stride=1,
        list_of_exog_dicts=filtered_exog_dicts,
    )
    val_dataset = windowDatasetmulti(
        config=config,
        args=args,
        df=df_val,
        df_start_date=args.val_start_date,
        df_end_date=args.val_end_date,
        abbr=abbr,
        abbr2id=abbr2id,
        input_window=args.past_steps,
        output_window=args.future_steps,
        stride=1,
        list_of_exog_dicts=filtered_exog_dicts,
    )
    test_dataset = windowDatasetmulti(
        config=config,
        args=args,
        df=df_test,
        df_start_date=args.test_start_date,
        df_end_date=args.test_end_date,
        abbr=abbr,
        abbr2id=abbr2id,
        input_window=args.past_steps,
        output_window=args.future_steps,
        stride=1,
        list_of_exog_dicts=filtered_exog_dicts,
    )
    return train_dataset, val_dataset, test_dataset


def run_training(config, args, device):
    abbr = args.abbr
    if args.exog_type == "region":
        csv_file = f"../code/PCC_results/z_corr_REGION_v2/{args.nation}/{args.corr_type}/{args.abbr}_LAG-50.csv"
        columns_to_parse = ["feature", "lag", "best_corr", "p_value"]
    elif args.exog_type == "gtrend":
        csv_file = f"../code/PCC_results/z_corr_GTREND_v2/{args.nation}/{args.corr_type}/{args.abbr}_LAG-50.csv"
        columns_to_parse = ["feature", "lag", "best_corr", "p_value"]
    elif args.exog_type == "all":
        csv_file = f"../code/PCC_results/z_corr_agg_v1/{args.nation}/{args.corr_type}/{args.abbr}_LAG-50.csv"
        columns_to_parse = ["feature", "lag", "best_corr", "p_value", "type"]
    """추가"""
    d = []
    count = 0
    print("csv_file", csv_file)
    if args.CFS:
        for row_num in range(0, 50):
            print("row_num", row_num)
            args.threshold
            print("args.threshold", args.threshold)
            try:
                values = get_nth_value_from_columns(csv_file, row_num,
                                                    columns_to_parse)
                print("values", values)
                if values["p_value"] > 0.1:
                    continue
                if values["best_corr"] < args.threshold:
                    continue
                count += 1
                d.append(values)
            except Exception as e:
                print(f"Error reading row {row_num}: {e}")
                break
        args.N_MAX = count
        args.in_channels = args.N_MAX + 1
        print(f"args.in_channels: {args.in_channels}")
    else:
        pass
    args.N_MAX = count
    args.in_channels = args.N_MAX + 1
    print("args.in_channels", args.in_channels)
    args.nation
    json_path = f"../data/x_data_aux/{args.nation}/abbr2id.json"
    with open(json_path, "r") as f:
        abbr2id = json.load(f)
    list(abbr2id.keys())
    node_id = str(abbr2id[abbr]).zfill(2)
    path = f"../data/x_data_epidemiology/{args.nation}/{node_id}_{abbr}.csv"
    df = pd.read_csv(path, usecols=["date", "new_confirmed"])
    if True:
        df["new_confirmed"] = df["new_confirmed"].astype(float).replace(
            0.0, np.nan)
        df["new_confirmed"] = df["new_confirmed"].ffill().bfill()
    df_train, df_val, df_test, scaler_stats, initial_values = split_rolling_scale(
        config,
        args,
        df,
        cols=["new_confirmed"],
        test_boundary=args.test_start_date)
    train_dataset, val_dataset, test_dataset = select_features_combined(
        config, args, df_train, df_val, df_test, abbr, abbr2id, d)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.BATCHSIZE,
                              shuffle=True,
                              drop_last=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.BATCHSIZE,
                            shuffle=False,
                            drop_last=False)
    test_loader = prepare_test_loader(args, test_dataset, args.device)
    model = initialize_model(config, args, device)
    learning_rate = args.LR
    epochs = args.EPOCHS
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    args.eary_stop_MIN_EPOCHS = 1
    print("args.eary_stop_MIN_EPOCHS", args.eary_stop_MIN_EPOCHS)
    early_stopping = EarlyStopping(patience=args.patience,
                                   delta=0.0000000001,
                                   min_epochs=args.eary_stop_MIN_EPOCHS)
    if args.EPOCHS != 0:
        config, args = retrieve_last_processed_give_lc_path(config, args)
        dir_path = os.path.dirname(config["C_PATH"])
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        args.start_epoch = 0
        if True:
            print("loading checkpoints")
            import time
            time.time()
            optimizer = initialize_optimizer(model, config["LR"])
            if args.LOAD_CHECKPOINT:
                config, args.start_epoch, model, optimizer = load_checkpoint(
                    model, optimizer, config, device)
                optimizer = initialize_optimizer(model, config["LR"])
    model.train()
    print("Training starts")
    best_val_loss = np.inf

    train_loss_list = []
    val_loss_list = []
    ACC_STEPS = int(args.BATCHSIZE / 64)
    ACC_STEPS = max(1, int(args.BATCHSIZE / 64))
    len(train_loader) * epochs
    global_step = args.start_epoch * len(train_loader)
    if True:
        config_path = os.path.join(args.save_path, "config.json")
        with open(config_path, "w") as f:
            json.dump(serialize_dict(config), f)

    def check_training_condition(args):
        args.condition_eval_ckpt = False
        config_path = os.path.join(args.save_path, "config.json")
        if not os.path.exists(config_path):
            return True
        try:
            with open(config_path, "r") as f:
                saved_config = json.load(f)
            if saved_config.get("end_reason") == "replacement":
                args.condition_eval_ckpt = True
            should_train = args.start_epoch <= args.EPOCHS or args.condition_eval_ckpt
            return should_train
        except Exception as e:
            print(f"Error reading config: {e}")
            return True

    if not check_training_condition(args):
        return
    else:
        with tqdm(range(args.start_epoch, epochs)) as tr:
            for epoch in tr:
                total_train_loss = 0.0
                model.train()
                for step, (x, y, _, _) in enumerate(train_loader):
                    x = x.to(
                        device,
                        dtype=(torch.float64
                               if args.dtype == "double" else torch.float32),
                    )
                    y = y.to(
                        device,
                        dtype=(torch.float64
                               if args.dtype == "double" else torch.float32),
                    )
                    global_step = epoch * len(train_loader) + step
                    output = model(x, y, args.future_steps, global_step).to(
                        device,
                        dtype=(torch.float64
                               if args.dtype == "double" else torch.float32),
                    )
                    print(f"==>> output.shape: {output.shape}")
                    loss = criterion(output, y)
                    print("TRAIN loss", loss)
                    loss = torch.sqrt(loss)
                    if ACC_STEPS > 1:
                        loss = loss / ACC_STEPS
                    loss.backward()
                    if (step + 1) % ACC_STEPS == 0 or (step +
                                                       1) == len(train_loader):
                        optimizer.step()
                        optimizer.zero_grad()
                    total_train_loss += loss.item() * ACC_STEPS
                avg_train_loss = total_train_loss / len(train_loader)
                train_loss_list.append(avg_train_loss)
                print(
                    f"Epoch {epoch+1}/{epochs}, Average Train Loss: {avg_train_loss}"
                )
                model.eval()
                total_val_loss = 0.0
                with torch.no_grad():
                    for vstep, (x_val, y_val, _, _) in enumerate(val_loader):
                        x_val = x_val.to(
                            device,
                            dtype=(torch.float64 if args.dtype == "double" else
                                   torch.float32),
                        )
                        y_val = y_val.to(
                            device,
                            dtype=(torch.float64 if args.dtype == "double" else
                                   torch.float32),
                        )
                        output_val = model(
                            x_val, y_val, args.future_steps, global_step).to(
                                device,
                                dtype=(torch.float64 if args.dtype == "double"
                                       else torch.float32),
                            )
                        val_loss = criterion(output_val, y_val)
                        val_loss = torch.sqrt(val_loss)
                        total_val_loss += val_loss.item()
                avg_val_loss = total_val_loss / len(val_loader)
                val_loss_list.append(avg_val_loss)
                if avg_val_loss < best_val_loss and not np.isnan(avg_val_loss):
                    best_val_loss = avg_val_loss
                    print("■ Saving best_val_loss")
                    torch.save(
                        {
                            "epoch": epoch,
                            "modelstate_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "global_step": global_step,
                        },
                        args.C_PATH,
                    )
                print(
                    f"Epoch {epoch+1}/{epochs}, Average Validation Loss: {avg_val_loss}"
                )
                config["last_epoch"] = epoch
                if False:
                    config_path = os.path.join(args.save_path, "config.json")
                    with open(config_path, "w") as f:
                        json.dump(serialize_dict(config), f)
                p_early = os.path.join(args.save_path, f"early_ckpt/")
                early_stopping(avg_val_loss, model, p_early)
                if early_stopping.early_stop or epoch == args.EPOCHS - 1:
                    if early_stopping.early_stop:
                        config["end_reason"] = "early_stop"
                    elif epoch == args.EPOCHS - 1:
                        config["end_reason"] = "train_end"
                    log_path = os.path.join(args.save_path, "training_log.txt")
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    stop_reason = "Early stopping triggered" if early_stopping.early_stop else "Training completed"
                    with open(log_path, "a") as f:
                        f.write(
                            f"\nTraining at {current_time} - Last epoch: {epoch} - {stop_reason}"
                        )
                    config["last_epoch"] = epoch
                    config_path = os.path.join(args.save_path, "config.json")
                    with open(config_path, "w") as f:
                        json.dump(serialize_dict(config), f)
                    print(f"\n{stop_reason} at epoch {epoch}")
                    break
                tr.set_postfix(train_loss=f"{avg_train_loss:.5f}",
                               val_loss=f"{avg_val_loss:.5f}")

    if 1:
        print("Loading checkpoint for testing")
        try:
            checkpoint = torch.load(config["C_PATH"],
                                    map_location="cpu",
                                    weights_only=True)
        except:
            checkpoint = torch.load(config["L_PATH"],
                                    map_location="cpu",
                                    weights_only=True)
        model.load_state_dict(checkpoint["modelstate_dict"], strict=False)
        with torch.no_grad():
            actual_dict, pred_dict, avg_loss = process_test_output(
                config,
                args,
                initial_values,
                test_loader,
                model,
                scaler_stats,
                criterion,
                device,
                df_test,
            )
        print("actual_dict, pred_dict, avg_loss", actual_dict, pred_dict,
              avg_loss)

        def flatten_and_combine(data):
            combined_list = []
            for key, value in data.items():
                for sublist in value:
                    combined_list.extend(sublist)
            return combined_list

        with torch.no_grad():
            metric_dict, avg_metric_dict = window_evaluation_dict_by_TYPE(
                config, args, pred_dict, actual_dict)
        print("=============args.save_path=============", )
        print(
            "save path", args.save_path +
            "/window/stride7-ws5-steps591317212529333741/avg_metric.json")
        print("=====================================")
        print()
        step_str = "".join([str(i) for i in args.horizon_steps])
        args.eval_step_str = f"stride{args.eval_stride}-ws{args.ws}-steps{step_str}"
        a = f"{args.save_path}/{args.eval_type}/{args.eval_step_str}"
        os.makedirs(a, exist_ok=True)
        with open(
                os.path.join(args.save_path, args.eval_type,
                             str(args.eval_step_str), "metric.json"),
                "w",
        ) as f:
            json.dump(metric_dict, f)
        with open(
                os.path.join(
                    args.save_path,
                    args.eval_type,
                    str(args.eval_step_str),
                    "avg_metric.json",
                ),
                "w",
        ) as f:
            json.dump(avg_metric_dict, f)
    return


def get_nearest_monday(date_str):
    input_date = pd.to_datetime(date_str)
    weekday = input_date.weekday()
    if weekday <= 3:
        nearest_monday = input_date - pd.Timedelta(days=weekday)
    else:
        nearest_monday = input_date + pd.Timedelta(days=(7 - weekday))
    return nearest_monday.strftime("%Y-%m-%d")


def get_date_tuples_train_val_test(args):
    args.future_steps
    dates = pd.date_range(args.train_start_date, args.test_end_date)
    ldates = len(dates)
    train_val_split = int(ldates * args.SPLIT[0])
    val_test_split = int(ldates * args.SPLIT[1])
    train_dates = dates[:train_val_split]
    val_dates = dates[train_val_split:val_test_split]
    test_dates = dates[val_test_split:]
    args.train_start_date = train_dates[0].strftime("%Y-%m-%d")
    args.val_start_date = get_nearest_monday(val_dates[0].strftime("%Y-%m-%d"))
    args.train_end_date = (pd.to_datetime(args.val_start_date) -
                           pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    args.test_start_date = get_nearest_monday(
        test_dates[0].strftime("%Y-%m-%d"))
    args.val_end_date = (pd.to_datetime(args.test_start_date) -
                         pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    args.test_end_date = (pd.to_datetime(test_dates[-1]) -
                          pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return args


if __name__ == "__main__":
    import argparse
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7, 8"
    print("Available devices ", torch.cuda.device_count())
    device = torch.device("cuda:0")
    input_path = "../data"
    output_path = "../results"
    BATCHSIZE = 512
    SEED = 1
    embedding_dim = 1
    past_steps = 42
    future_steps = 42
    total_steps = past_steps + future_steps
    patience = 1
    CONTINUE = 1
    MAKE_DFS = 0
    EPOCHS = 100
    MAX_STEPS = 300
    V_MAX_STEPS = 9
    L_PATH = ""
    C_PATH = ""
    output_size = 1
    if MAKE_DFS:
        CONTINUE = False
    a = ["TriAttLSTM", "SimpleLSTM"]
    MODEL_only = a[0]
    print("BATCHSIZE", BATCHSIZE)
    parser = argparse.ArgumentParser(description="COVID-19")
    parser.add_argument("--device", default=device)
    parser.add_argument("--MODEL_only", default=MODEL_only)
    parser.add_argument("--SEED", type=int, default=SEED)
    parser.add_argument("--use_curriculum_learning",
                        type=str2bool,
                        default=True)
    parser.add_argument("--embedding_dim", type=int, default=embedding_dim)
    parser.add_argument("--past_steps", type=int, default=past_steps)
    parser.add_argument("--future_steps", type=int, default=future_steps)
    parser.add_argument("--total_steps", type=int, default=total_steps)
    parser.add_argument("--patience", type=int, default=patience)
    parser.add_argument("--diff", type=str2bool, default=True)
    parser.add_argument("--CONTINUE", type=str2bool, default=CONTINUE)
    parser.add_argument("--MAKE_DFS", type=str2bool, default=MAKE_DFS)
    parser.add_argument("--EPOCHS", type=int, default=EPOCHS)
    parser.add_argument("--MAX_STEPS", type=int, default=MAX_STEPS)
    parser.add_argument("--V_MAX_STEPS", type=int, default=V_MAX_STEPS)
    parser.add_argument("--BATCHSIZE", type=int, default=BATCHSIZE)
    parser.add_argument("--TEST", action="store_true")
    parser.add_argument("--input_path", type=str, default="../data")
    parser.add_argument("--output_path", type=str, default="../results")
    parser.add_argument("--scaler", type=str, default="standard")
    parser.add_argument("--L_PATH",
                        type=str,
                        required=False,
                        help="Path to load checkpoint")
    parser.add_argument("--C_PATH",
                        type=str,
                        required=False,
                        help="Path to save checkpoint")
    parser.add_argument("--nation", type=str, default="AU")
    parser.add_argument("--activations", type=list, default=["gelu"])
    parser.add_argument("--attn_heads", type=int, default=8)
    parser.add_argument("--range_val",
                        type=int,
                        default=6,
                        help="Number of top variables to use")
    parser.add_argument(
        "--loss_type",
        type=str,
        default="Huber",
        choices=["RMSE", "Huber", "MultiRMSE", "WeightedHuber"],
    )
    parser.add_argument("--train_start_date", type=str, default="2020-01-22")
    parser.add_argument("--train_end_date", type=str, default="2022-04-22")
    parser.add_argument("--val_start_date", type=str, default="2022-06-08")
    parser.add_argument("--val_end_date", type=str, default="2022-06-09")
    parser.add_argument("--test_start_date", type=str, default="2022-06-10")
    parser.add_argument("--test_end_date", type=str, default="2022-09-14")
    parser.add_argument("--eval_start_date", type=str, default="2022-06-20")
    parser.add_argument("--CFS", type=str2bool, default="1")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--threshold2", type=float, default=0.8)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--abbr", type=str, default="")
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--layer_string", type=str, default="1111")
    parser.add_argument("--last_linear", type=str2bool, default=False)
    parser.add_argument("--dtype", type=str, default="double")
    parser.add_argument("--eval_stride", type=int, default=7)
    parser.add_argument("--eval_type",
                        default="window",
                        choices=["cumulative", "point", "window"])
    parser.add_argument("--add_skips", type=str2bool, default=True)
    parser.add_argument("--incre_START", type=int, default=0)
    parser.add_argument("--last_skip", type=str2bool, default=1)
    args = parser.parse_args()
    args.eval_type = "window"
    args.eval_stride = 7
    args.ws = 5
    args.horizon_steps = [2, 3, 4, 5, 8, 10, 15, 20, 25, 28]
    args.horizon_steps = [2, 3, 5, 10, 15, 20, 25, 30, 35, 40]
    args.horizon_steps = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
    args.horizon_steps = [h + 1 for h in args.horizon_steps]
    args.test_date_path = f"test_date_list_{str(args.eval_stride).zfill(2)}.json"
    nation = args.nation
    args.torch_dtype = torch.float64 if args.dtype == "double" else torch.float32
    print("args.CFS", args.CFS)
    print("nation", nation)
    set_seed(seed=args.SEED, make_deterministic=True)
    args.SPLIT = (0.6, 0.75)
    a = ["all", "gtrend", "region"]
    args.exog_type = a[0]
    print("args.exog_type", args.exog_type)
    args.LOAD_CHECKPOINT = True
    d = {
        "US": ("2020-01-22", "2022-09-14"),
        "AU": ("2020-02-29", "2022-09-09"),
        "CA": ("2020-02-24", "2022-04-24"),
    }
    args.train_start_date = d[args.nation][0]
    args.test_end_date = d[args.nation][1]
    """date_range = pd.date_range(start=start_date, end=end_date, freq='D')"""
    args = get_date_tuples_train_val_test(args)
    if 1:
        print(f"Train start date: {args.train_start_date}")
        print(f"Train end date: {args.train_end_date}")
        print(f"Validation start date: {args.val_start_date}")
        print(f"Validation end date: {args.val_end_date}")
        print(f"Test start date: {args.test_start_date}")
        print(f"Test end date: {args.test_end_date}")

    def parse_layer_numbers(args, layer_string):
        args.nlayer_geatt1 = int(layer_string[0])
        args.nlayer_geatt2 = int(layer_string[1])
        args.nlayer_geatt3a = int(layer_string[2])
        args.nlayer_geatt3b = int(layer_string[3])
        return args, args.nlayer_geatt1, args.nlayer_geatt2, args.nlayer_geatt3a, args.nlayer_geatt3b

    args, args.nlayer_geatt1, args.nlayer_geatt2, args.nlayer_geatt3a, args.nlayer_geatt3b = parse_layer_numbers(
        args, args.layer_string)
    device = str(args.device)
    device = torch.device(device)
    args.device = device
    torch.cuda.set_device(device)
    with open(f"../data/x_data_aux/{args.nation}/config.json", "r") as f:
        config_init = json.load(f)
    print("Current cuda device ", torch.cuda.current_device())
    print(f"Running with SEED={args.SEED}, Device={args.device}, ")
    args.replace_zero_to_nan = True
    args.interpolate_input_df = True
    args.current_time = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    args.attn_hidden_dim = 32
    args.mha_hidden = 32
    args.input_path = input_path
    args.output_path = output_path
    args.temperature = 0.1
    args.LR = 4.245451262763638e-06
    print("change")
    args.target_col = "new_confirmed_Y"
    args.unknown_real_x_target = ["new_confirmed"]
    args.data_name = "covid"
    args.time_col = "date"
    args.rolling = True
    args.rolling_window = 7
    args.chg = False
    args.log = False
    args.diff = True
    args.device = device
    """"""
    num_for_predict = args.num_for_predict = args.future_steps
    dropout_type = args.dropout_type = "zoneout"
    fusion_mode = args.fusion_mode = "mix"
    print("args.use_curriculum_learning", args.use_curriculum_learning)
    cl_decay_steps = args.cl_decay_steps = 200
    args.make_stats = False
    args.MODEL = f"{args.MODEL_only}-{args.exog_type}"
    if args.CFS:
        args.MODEL += f"-XRedun" + str(round(args.threshold2, 2)).replace(
            ".", "")
    else:
        args.MODEL += "-Top00"
    print(f"Running {args.MODEL} with Device={device}")
    if args.CFS:
        args.MODEL += "-Thr" + str(round(args.threshold, 2)).replace(".", "")
    if 1:
        args.MODEL += "-h" + str(args.hidden_size).zfill(3)
    if "Simple" not in args.MODEL:
        args.MODEL += f"-nlyr{args.nlayer_geatt1}{args.nlayer_geatt2}{args.nlayer_geatt3a}{args.nlayer_geatt3b}"
    args.last_linear = True
    print("args.MODEL", args.MODEL)
    args.input_cols = []
    args.encoder_layers = args.decoder_layers = 1
    args.output_size = 1
    op = {"LR": 4.245451262763638e-06}
    args.partial = True
    if args.diff:
        args.MODEL += "-diff"
    SCALERS = ["minmax", "standard", "none"]
    if "Simple" not in args.MODEL:
        args.activations_str = "-".join(args.activations)
        args.MODEL += "-" + args.activations_str
    print("EPOCHS", args.EPOCHS)
    print("args.MODEL", args.MODEL)
    args.corr_type = "pcc"
    json_path = f"../data/x_data_aux/{args.nation}/abbr2id.json"
    with open(json_path, "r") as f:
        abbr2id = json.load(f)
    args.incre_factor = 2
    abbr_list = list(abbr2id.keys())[args.incre_START *
                                     args.incre_factor:(args.incre_START + 1) *
                                     args.incre_factor]
    print("abbr_list", abbr_list)
    print("■■■ abbr_list - PARTIAL!!!!", abbr_list)
    if 0:
        abbr_list.reverse()
    for abbr in abbr_list:
        print("□ abbr", abbr)
        args.abbr = abbr
        print(
            "args.data_name, args.nation, args.MODEL_only, args.MODEL,  args.abbr",
            args.data_name,
            args.nation,
            args.MODEL_only,
            args.MODEL,
            args.abbr,
        )
        args.cols = ["new_confirmed"]
        args.save_path = os.path.join(
            f"../results",
            args.data_name,
            args.nation,
            args.MODEL_only,
            args.MODEL,
            args.abbr,
            str(args.SEED),
        )
        os.makedirs(args.save_path, exist_ok=True)
        print("config from ARGS")
        config = vars(args)
        config.update(op)
        run_training(config, args, device=device)
