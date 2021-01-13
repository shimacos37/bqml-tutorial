CREATE OR REPLACE MODEL `otto.dnn_log_model_fold{{n_fold}}`
OPTIONS(
  MODEL_TYPE="DNN_CLASSIFIER",
  ACTIVATION_FN="RELU",
  BATCH_SIZE=512,
  DROPOUT=0.3,
  EARLY_STOP=TRUE,
  LEARN_RATE=0.001,
  HIDDEN_UNITS=[256, 128, 256],
  OPTIMIZER='ADAM',
  DATA_SPLIT_METHOD="CUSTOM",
  DATA_SPLIT_COL="is_eval",
  MAX_ITERATIONS=100,
  MIN_REL_PROGRESS=0.0001,
  INPUT_LABEL_COLS=["target"]
) AS

SELECT
  -- idは特徴量に含めたくないのでコメントアウト
  -- id,
  -- fold
  CASE 
    WHEN MOD(id, 5) = {{n_fold}} THEN True
    ELSE False
  END AS is_eval,
  -- label
  target,
  -- feature
  {% for n in range(1, 94) %} LN(feat_{{n}} + 1) AS feat_{{n}},
  {% endfor %}
FROM (
  SELECT
    * except (target),
    CAST(SPLIT(target, '_')[OFFSET(1)] AS INT64) AS target,
  FROM
    otto.train
)
