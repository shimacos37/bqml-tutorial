import os
import logging
import pickle
from typing import List

import fire
import tensorflow as tf
import tensorflow.keras.layers as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

plt.style.use("ggplot")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NNModel(object):
    def __init__(
        self,
        feature_cols: List[str],
        label_col: str,
        batch_size: int = 512,
        learning_rate: float = 0.001,
        save_dir: str = "./output",
    ) -> None:
        self.feature_cols = feature_cols
        self.label_col = label_col
        # model setting
        self.model = self.__create_model()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        # Save setting
        self.save_dir = save_dir

    def __create_model(self) -> Model:
        input_ = tf.keras.Input(shape=(len(self.feature_cols),), name="input")
        # Layer 1
        out = L.Dense(512, activation="relu")(input_)
        out = L.BatchNormalization()(out)
        out = L.Dropout(0.5)(out)
        # Layer 2
        out = L.Dense(512, activation="relu")(out)
        out = L.BatchNormalization()(out)
        out = L.Dropout(0.5)(out)
        # Layer 3
        out = L.Dense(512, activation="relu")(out)
        out = L.BatchNormalization()(out)
        out = L.Dropout(0.5)(out)
        # Last Layer
        out = L.Dense(512, activation="relu")(out)
        out = L.Dense(9, activation="softmax", name="output")(out)
        model = Model(input_, out)
        return model

    def train(self, train_df: pd.DataFrame, valid_df: pd.DataFrame):
        es_cb = EarlyStopping(monitor="val_loss", patience=5, verbose=1, mode="auto")
        cp_cb = ModelCheckpoint(
            filepath=self.save_dir,
            monitor="val_loss",
            verbose=1,
            mode="auto",
        )
        self.model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
        )
        self.model.fit(
            train_df[self.feature_cols],
            train_df[self.label_col],
            epochs=100,
            verbose=True,
            batch_size=self.batch_size,
            validation_data=(
                valid_df[self.feature_cols],
                valid_df[self.label_col],
            ),
            callbacks=[es_cb, cp_cb],
        )
        # load best model
        self.model: Model = tf.keras.models.load_model(self.save_dir, compile=False)

    def predict(self, df: pd.DataFrame):
        pred = self.model.predict(df[self.feature_cols])
        return pred


def main(use_log_feature=False):
    train_df = pd.read_csv("./input/otto/train.csv")
    test_df = pd.read_csv("./input/otto/test.csv")
    # BQML上と同じ切り方をする
    train_df["fold"] = train_df["id"] % 5
    # ラベルを整数値に
    train_df["target"] = train_df["target"].str[-1:].astype(int) - 1
    # column setting
    feature_cols = [
        col for col in train_df.columns if col not in ["target", "id", "fold"]
    ]
    label_col = "target"

    if not use_log_feature:
        output_path = "./output/nn"
    else:
        output_path = "./output/nn_log"
        train_df[feature_cols] = np.log1p(train_df[feature_cols])
        test_df[feature_cols] = np.log1p(test_df[feature_cols])

    for n_fold in range(5):
        train_df_ = train_df.query("fold!=@n_fold").copy()
        valid_df = train_df.query("fold==@n_fold")
        test_df_ = test_df.copy()
        std = StandardScaler()
        train_df_[feature_cols] = std.fit_transform(train_df_[feature_cols])
        valid_df[feature_cols] = std.transform(valid_df[feature_cols])
        test_df_[feature_cols] = std.transform(test_df_[feature_cols])
        model = NNModel(
            feature_cols, label_col, save_dir=os.path.join(output_path, f"fold{n_fold}")
        )
        model.train(train_df_, valid_df)
        pred = model.predict(valid_df)
        for i in range(pred.shape[-1]):
            train_df.loc[valid_df.index, f"target_pred{i}"] = pred[:, i]
        pred = model.predict(test_df_)
        for i in range(pred.shape[-1]):
            test_df[f"target_pred{i}_fold{n_fold}"] = pred[:, i]
        with open(os.path.join(output_path, f"fold{n_fold}/std.pkl"), "wb") as f:
            pickle.dump(std, f)
    logger.info(
        f"LogLoss: {log_loss(train_df[label_col], train_df[[f'target_pred{i}' for i in range(9)]])}"
    )
    for i in range(9):
        test_df[f"Class_{i+1}"] = test_df[
            [f"target_pred{i}_fold{n}" for n in range(5)]
        ].mean(1)
    test_df[["id"] + [f"Class_{i}" for i in range(1, 10)]].to_csv(
        os.path.join(output_path, "sub.csv"), index=False
    )


if __name__ == "__main__":
    fire.Fire(main)
