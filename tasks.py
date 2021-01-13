import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob

import pandas as pd
from google.cloud import storage
from invoke import Collection, task
from jinja2 import Template

from utils import BQClient


def read_sql(path):
    with open(path, "r") as f:
        sql = "".join(f.readlines())
    return sql


@task
def upload_data(c):
    bq = BQClient(c.gcp.project_id, c.gcp.dataset)
    df = pd.read_csv("./input/otto/train.csv")
    bq.insert_df_to_table(df, c.gcp.dataset, "train")
    df = pd.read_csv("./input/otto/test.csv")
    bq.insert_df_to_table(df, c.gcp.dataset, "test")


# ====================================
# Logistic Regression, DNN, XGBoost
# ====================================


@task
def train_model(
    c, model_name="logistic", use_log_feature=False, use_kmeans_feature=False
):
    assert model_name in [
        "logistic",
        "dnn",
        "xgb",
    ], 'Please set the model_name parameter from "logistic", "dnn" and "xgb"'

    bq = BQClient(c.gcp.project_id, c.gcp.dataset)
    thread_executor = ThreadPoolExecutor()
    jobs = []
    if not use_log_feature:
        if not use_kmeans_feature:
            sql_template = Template(
                read_sql(f"sql/simple_models/train_{model_name}_model.sql")
            )
        else:
            sql_template = Template(
                read_sql(
                    f"sql/kmeans_models/simple_models/train_{model_name}_model.sql"
                )
            )
    else:
        if not use_kmeans_feature:
            sql_template = Template(
                read_sql(f"sql/log_models/train_{model_name}_model.sql")
            )
        else:
            sql_template = Template(
                read_sql(f"sql/kmeans_models/log_models/train_{model_name}_model.sql")
            )
    for n_fold in range(5):
        sql = sql_template.render({"n_fold": n_fold})
        jobs.append(thread_executor.submit(bq.execute_query, sql))
    for future in as_completed(jobs):
        jobs.remove(future)


@task
def predict_oof_model(
    c, model_name="logistic", use_log_feature=False, use_kmeans_feature=False
):
    assert model_name in [
        "logistic",
        "dnn",
        "xgb",
    ], 'Please set the model_name parameter from "logistic", "dnn" and "xgb"'

    bq = BQClient(c.gcp.project_id, c.gcp.dataset)
    if not use_log_feature:
        if not use_kmeans_feature:
            sql_template = Template(read_sql("sql/simple_models/predict_oof_model.sql"))
            sql = sql_template.render(
                {
                    "table_id": f"{model_name}_oof_pred",
                    "model_name": f"{model_name}_model",
                }
            )
        else:
            sql_template = Template(
                read_sql("sql/kmeans_models/simple_models/predict_oof_model.sql")
            )
            sql = sql_template.render(
                {
                    "table_id": f"{model_name}_oof_kmeans_pred",
                    "model_name": f"{model_name}_kmeans_model",
                }
            )
    else:
        if not use_kmeans_feature:
            sql_template = Template(read_sql("sql/log_models/predict_oof_model.sql"))
            sql = sql_template.render(
                {
                    "table_id": f"{model_name}_oof_log_pred",
                    "model_name": f"{model_name}_log_model",
                }
            )
        else:
            sql_template = Template(
                read_sql("sql/kmeans_models/log_models/predict_oof_model.sql")
            )
            sql = sql_template.render(
                {
                    "table_id": f"{model_name}_oof_kmeans_log_pred",
                    "model_name": f"{model_name}_kmeans_log_model",
                }
            )
    bq.execute_query(sql)


@task
def predict_test_model(
    c, model_name="logistic", use_log_feature=False, use_kmeans_feature=False
):
    assert model_name in [
        "logistic",
        "dnn",
        "xgb",
    ], 'Please set the model_name parameter from "logistic", "dnn" and "xgb"'

    bq = BQClient(c.gcp.project_id, c.gcp.dataset)
    if not use_log_feature:
        if not use_kmeans_feature:
            sql_template = Template(
                read_sql("sql/simple_models/predict_test_model.sql")
            )
            sql = sql_template.render(
                {
                    "table_id": f"{model_name}_test_pred",
                    "model_name": f"{model_name}_model",
                }
            )
        else:
            sql_template = Template(
                read_sql("sql/kmeans_models/simple_models/predict_test_model.sql")
            )
            sql = sql_template.render(
                {
                    "table_id": f"{model_name}_test_kmeans_pred",
                    "model_name": f"{model_name}_kmeans_model",
                }
            )
    else:
        if not use_kmeans_feature:
            sql_template = Template(read_sql("sql/log_models/predict_test_model.sql"))
            sql = sql_template.render(
                {
                    "table_id": f"{model_name}_test_log_pred",
                    "model_name": f"{model_name}_log_model",
                }
            )
        else:
            sql_template = Template(
                read_sql("sql/kmeans_models/log_models/predict_test_model.sql")
            )
            sql = sql_template.render(
                {
                    "table_id": f"{model_name}_test_kmeans_log_pred",
                    "model_name": f"{model_name}_kmeans_log_model",
                }
            )
    bq.execute_query(sql)


@task
def evaluate_model(
    c, model_name="logistic", use_log_feature=False, use_kmeans_feature=False
):
    assert model_name in [
        "logistic",
        "dnn",
        "xgb",
    ], 'Please set the model_name parameter from "logistic", "dnn" and "xgb"'

    bq = BQClient(c.gcp.project_id, c.gcp.dataset)
    sql_template = Template(read_sql("sql/evaluate_model.sql"))
    if not use_log_feature:
        if not use_kmeans_feature:
            sql = sql_template.render({"model_name": f"{model_name}_model"})
        else:
            sql = sql_template.render({"model_name": f"{model_name}_kmeans_model"})
    else:
        if not use_kmeans_feature:
            sql = sql_template.render({"model_name": f"{model_name}_log_model"})
        else:
            sql = sql_template.render({"model_name": f"{model_name}_kmeans_log_model"})
    bq.execute_query(sql)


# ====================================
# KMeans
# ====================================


@task
def train_kmeans_model(c, use_log_feature=False):
    bq = BQClient(c.gcp.project_id, c.gcp.dataset)
    if not use_log_feature:
        sql_template = Template(read_sql("sql/simple_models/train_kmeans_model.sql"))
    else:
        sql_template = Template(read_sql("sql/log_models/train_kmeans_model.sql"))
    sql = sql_template.render()
    bq.execute_query(sql)


# ====================================
# Tensorflow
# ====================================


@task
def train_tensorflow_model(c, use_log_feature=False):
    cmd = "poetry run python train_tensorflow_model.py"
    if use_log_feature:
        cmd += " --use_log_feature=True"
    c.run(cmd, echo=True)


@task
def upload_models(c):
    storage_client = storage.Client(c.gcp.project_id)
    bucket = storage_client.get_bucket(c.gcp.bucket_name)
    for model_name in ["nn", "nn_log"]:
        save_path = f"./output/{model_name}"
        filenames = glob(os.path.join(save_path, "**"), recursive=True)
        for filename in filenames:
            if os.path.isdir(filename):
                continue
            destination_blob_name = os.path.join(
                model_name,
                filename.split(save_path)[-1][1:],
            )
            blob = bucket.blob(destination_blob_name)
            blob.upload_from_filename(filename)


@task
def import_tensorflow_model(c, use_log_feature=False):
    bq = BQClient(c.gcp.project_id, c.gcp.dataset)
    thread_executor = ThreadPoolExecutor()
    jobs = []
    sql_template = Template(read_sql("sql/tensorflow_models/import_model.sql"))
    for n_fold in range(5):
        if not use_log_feature:
            sql = sql_template.render(
                {"bucket_name": c.gcp.bucket_name, "model_name": "nn", "n_fold": n_fold}
            )
        else:
            sql = sql_template.render(
                {
                    "bucket_name": c.gcp.bucket_name,
                    "model_name": "nn_log",
                    "n_fold": n_fold,
                }
            )
        jobs.append(thread_executor.submit(bq.execute_query, sql))
    for future in as_completed(jobs):
        jobs.remove(future)


@task
def predict_test_tensorflow_model(c, use_log_feature=False):
    bq = BQClient(c.gcp.project_id, c.gcp.dataset)
    params = {}
    if not use_log_feature:
        for n_fold in range(5):
            with open(f"./output/nn/fold{n_fold}/std.pkl", "rb") as f:
                std = pickle.load(f)
            means = std.mean_
            stds = std.scale_
            mean_keys = [f"mean_{i + 1}" for i in range(len(means))]
            std_keys = [f"std_{i + 1}" for i in range(len(stds))]
            params.update(
                {
                    f"means_fold{n_fold}": zip(mean_keys, means),
                    f"stds_fold{n_fold}": zip(std_keys, stds),
                }
            )
        params.update(
            {
                "table_id": "tensorflow_test_pred",
                "model_name": "tensorflow_nn_model",
            }
        )
        sql_template = Template(
            read_sql("sql/tensorflow_models/simple_models/predict_test_model.sql")
        )
        sql = sql_template.render(params)
    else:
        for n_fold in range(5):
            with open(f"./output/nn_log/fold{n_fold}/std.pkl", "rb") as f:
                std = pickle.load(f)
            means = std.mean_
            stds = std.scale_
            mean_keys = [f"mean_{i + 1}" for i in range(len(means))]
            std_keys = [f"std_{i + 1}" for i in range(len(stds))]
            params.update(
                {
                    f"means_fold{n_fold}": zip(mean_keys, means),
                    f"stds_fold{n_fold}": zip(std_keys, stds),
                }
            )
        params.update(
            {
                "table_id": "tensorflow_test_log_pred",
                "model_name": "tensorflow_nn_log_model",
            }
        )
        sql_template = Template(
            read_sql("sql/tensorflow_models/log_models/predict_test_model.sql")
        )
        sql = sql_template.render(params)
    bq.execute_query(sql)


# ====================================
# Stacking
# ====================================


# ====================================
# SUB
# ====================================


@task
def make_submission(c, pred_table_id="dnn_test_pred"):
    sql_template = Template(read_sql("sql/sub.sql"))
    sql = sql_template.render({"pred_table_id": pred_table_id})
    sub = pd.read_gbq(sql, project_id=c.gcp.project_id, use_bqstorage_api=True)
    sub.to_csv(f"./output/{pred_table_id}_sub.csv", index=False)


# ====================================
# Integration
# ====================================
@task
def predict_oof_all(c):
    thread_executor = ThreadPoolExecutor()
    jobs = []
    # Simple Model
    for model_name in ["logistic", "dnn", "xgb"]:
        jobs.append(
            thread_executor.submit(
                predict_oof_model,
                c,
                use_log_feature=False,
                use_kmeans_feature=False,
                model_name=model_name,
            )
        )
    # Log feature
    for model_name in ["logistic", "dnn", "xgb"]:
        jobs.append(
            thread_executor.submit(
                predict_oof_model,
                c,
                use_log_feature=True,
                use_kmeans_feature=False,
                model_name=model_name,
            )
        )
    # Simple Model + KMeans feature
    for model_name in ["logistic", "dnn", "xgb"]:
        jobs.append(
            thread_executor.submit(
                predict_oof_model,
                c,
                use_log_feature=False,
                use_kmeans_feature=True,
                model_name=model_name,
            )
        )
    # Log feature + KMeans feature
    for model_name in ["logistic", "dnn", "xgb"]:
        jobs.append(
            thread_executor.submit(
                predict_oof_model,
                c,
                use_log_feature=True,
                use_kmeans_feature=True,
                model_name=model_name,
            )
        )
    for future in as_completed(jobs):
        jobs.remove(future)


@task
def predict_test_all(c):
    thread_executor = ThreadPoolExecutor()
    jobs = []
    # Simple Model
    for model_name in ["logistic", "dnn", "xgb"]:
        jobs.append(
            thread_executor.submit(
                predict_test_model,
                c,
                use_log_feature=False,
                use_kmeans_feature=False,
                model_name=model_name,
            )
        )
    # Log feature
    for model_name in ["logistic", "dnn", "xgb"]:
        jobs.append(
            thread_executor.submit(
                predict_test_model,
                c,
                use_log_feature=True,
                use_kmeans_feature=False,
                model_name=model_name,
            )
        )
    # Simple Model + KMeans feature
    for model_name in ["logistic", "dnn", "xgb"]:
        jobs.append(
            thread_executor.submit(
                predict_test_model,
                c,
                use_log_feature=False,
                use_kmeans_feature=True,
                model_name=model_name,
            )
        )
    # Log feature + KMeans feature
    for model_name in ["logistic", "dnn", "xgb"]:
        jobs.append(
            thread_executor.submit(
                predict_test_model,
                c,
                use_log_feature=True,
                use_kmeans_feature=True,
                model_name=model_name,
            )
        )
    for future in as_completed(jobs):
        jobs.remove(future)


@task
def make_submittion_all(c):
    thread_executor = ThreadPoolExecutor()
    jobs = []
    for model_name in ["logistic", "dnn", "xgb"]:
        jobs.append(
            thread_executor.submit(make_submission, c, f"{model_name}_test_pred")
        )
        jobs.append(
            thread_executor.submit(make_submission, c, f"{model_name}_test_log_pred")
        )
        jobs.append(
            thread_executor.submit(make_submission, c, f"{model_name}_test_kmeans_pred")
        )
        jobs.append(
            thread_executor.submit(
                make_submission, c, f"{model_name}_test_kmeans_log_pred"
            )
        )
    jobs.append(thread_executor.submit(make_submission, c, "tensorflow_test_pred"))
    jobs.append(thread_executor.submit(make_submission, c, "tensorflow_test_log_pred"))
    for future in as_completed(jobs):
        jobs.remove(future)


@task
def submit_all(c):
    filenames = []
    for model_name in ["logistic", "dnn", "xgb"]:
        filenames.extend(
            [
                f"{model_name}_test_pred",
                f"{model_name}_test_log_pred",
                f"{model_name}_test_kmeans_pred",
                f"{model_name}_test_kmeans_log_pred",
            ]
        )
    filenames.extend(["tensorflow_test_pred", "tensorflow_test_log_pred"])
    for filename in filenames:
        c.run(
            f"kaggle competitions submit otto-group-product-classification-challenge -f ./output/{filename}_sub.csv -m '{filename}'"
        )
