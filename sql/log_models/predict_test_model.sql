CREATE OR REPLACE TABLE otto.{{table_id}} (
  id INT64 NOT NULL,
  label INT64 NOT NULL,
  prob FLOAT64 NOT NULL,
)
AS
WITH TEST AS (
  SELECT
    id,
    {% for n in range(1, 94) %} LN(feat_{{n}} + 1) AS feat_{{n}},
    {% endfor %}
  FROM
    otto.test
)
{% for n_fold in range(0, 5) %}
, PRED_{{n_fold}} AS (
  SELECT
    id,
    predicted_target_probs,
  FROM
    ML.PREDICT
  (
    MODEL otto.{{model_name}}_fold{{n_fold}},
    (
        SELECT
            *
        FROM
            TEST
    )
  )
)
{% endfor %}
SELECT
  id,
  label,
  AVG(prob) AS prob,
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
GROUP BY
  id, label

