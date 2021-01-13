CREATE OR REPLACE MODEL `otto.xgb_log_model_fold{{n_fold}}`
OPTIONS(
  MODEL_TYPE="BOOSTED_TREE_CLASSIFIER",
  L1_REG=0.1,
  L2_REG=0.1,
  MIN_TREE_CHILD_WEIGHT=2,
  COLSAMPLE_BYTREE=0.8,
  COLSAMPLE_BYLEVEL=1.0,
  SUBSAMPLE=0.8,
  LEARN_RATE=0.05,
  MAX_TREE_DEPTH=8,
  MAX_ITERATIONS=1000,
  EARLY_STOP=TRUE,
  MIN_REL_PROGRESS=0.0008,
  TREE_METHOD='HIST',
  DATA_SPLIT_METHOD="CUSTOM",
  DATA_SPLIT_COL="is_eval",
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
