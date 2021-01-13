CREATE OR REPLACE MODEL `otto.dnn_stacking_model_fold{{n_fold}}`
OPTIONS(
  MODEL_TYPE="DNN_CLASSIFIER",
  ACTIVATION_FN="RELU",
  BATCH_SIZE=64,
  DROPOUT=0.3,
  EARLY_STOP=TRUE,
  LEARN_RATE=0.001,
  HIDDEN_UNITS=[128, 64, 128],
  OPTIMIZER='ADAM',
  DATA_SPLIT_METHOD="CUSTOM",
  DATA_SPLIT_COL="is_eval",
  MAX_ITERATIONS=100,
  MIN_REL_PROGRESS=0.0001,
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