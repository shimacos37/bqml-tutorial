CREATE OR REPLACE TABLE otto.{{table_id}} (
  id INT64 NOT NULL,
  n_fold INT64 NOT NULL,
  target INT64 NOT NULL,
  label INT64 NOT NULL,
  prob FLOAT64 NOT NULL,
)
AS
WITH TRAIN AS (
  SELECT
    id,
    CAST(SPLIT(target, '_')[OFFSET(1)] AS INT64) AS target,
    MOD(id, 5) AS n_fold,
    {% for n in range(1, 94) %} LN(feat_{{n}} + 1) AS feat_{{n}},
    {% endfor %}
  FROM
    otto.train
)
{% for n_fold in range(0, 5) %}
, PRED_{{n_fold}} AS (
  SELECT
    id,
    n_fold,
    target,
    predicted_target_probs,
  FROM
    ML.PREDICT
  (
    MODEL otto.{{model_name}}_fold{{n_fold}},
    (
        SELECT
            *
        FROM
            TRAIN
        WHERE
            n_fold = {{n_fold}}
    )
  )
)
{% endfor %}
SELECT
  id,
  n_fold,
  target,
  label,
  prob,
FROM (
  SELECT * FROM PRED_0
  UNION ALL
  SELECT * FROM PRED_1
  UNION ALL
  SELECT * FROM PRED_2
  UNION ALL
  SELECT * FROM PRED_3
  UNION ALL
  SELECT * FROM PRED_4
),
UNNEST(predicted_target_probs)

