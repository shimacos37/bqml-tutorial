CREATE OR REPLACE MODEL `otto.xgb_stacking_model_fold{{n_fold}}`
OPTIONS(
  MODEL_TYPE="BOOSTED_TREE_CLASSIFIER",
  L1_REG=0.1,
  L2_REG=0.1,
  MIN_TREE_CHILD_WEIGHT=2,
  COLSAMPLE_BYTREE=0.8,
  COLSAMPLE_BYLEVEL=1.0,
  SUBSAMPLE=0.8,
  LEARN_RATE=0.01,
  MAX_TREE_DEPTH=4,
  MAX_ITERATIONS=1000,
  EARLY_STOP=TRUE,
  MIN_REL_PROGRESS=0.0004,
  TREE_METHOD='HIST',
  DATA_SPLIT_METHOD="CUSTOM",
  DATA_SPLIT_COL="is_eval",
  INPUT_LABEL_COLS=["target"]
) AS

WITH TEMPLATE AS (
    SELECT
      id,
      -- fold
      CASE 
          WHEN MOD(id, 5) = {{n_fold}} THEN True
          ELSE False
      END AS is_eval,
      CAST(SPLIT(target, '_')[OFFSET(1)] AS INT64) AS target,
    FROM
      otto.train
  )
{% for table_name in table_names %}
, {{table_name}} AS (
    SELECT
      id,
      prob_array[OFFSET(0)] AS {{table_name}}_Class_1,
      prob_array[OFFSET(1)] AS {{table_name}}_Class_2,
      prob_array[OFFSET(2)] AS {{table_name}}_Class_3,
      prob_array[OFFSET(3)] AS {{table_name}}_Class_4,
      prob_array[OFFSET(4)] AS {{table_name}}_Class_5,
      prob_array[OFFSET(5)] AS {{table_name}}_Class_6,
      prob_array[OFFSET(6)] AS {{table_name}}_Class_7,
      prob_array[OFFSET(7)] AS {{table_name}}_Class_8,
      prob_array[OFFSET(8)] AS {{table_name}}_Class_9,
    FROM(
      SELECT
        id,
        ARRAY_AGG(prob) AS prob_array,
      FROM(
        SELECT
          *
        FROM
          otto.{{table_name}}
        ORDER BY id, label
    )
    GROUP BY id
    )
)
{% endfor %}
SELECT
  * except(id)
FROM
  TEMPLATE
{% for table_name in table_names %}
LEFT JOIN
  {{table_name}}
USING(id)
{% endfor %}