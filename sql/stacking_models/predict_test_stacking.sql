CREATE OR REPLACE TABLE otto.{{table_id}} (
  id INT64 NOT NULL,
  label INT64 NOT NULL,
  prob FLOAT64 NOT NULL,
)
AS

WITH TEMPLATE AS (
    SELECT
      id
    FROM
      otto.test
  )
{% for culumn_name, table_name in column_table_names %}
, {{table_name}} AS (
    SELECT
      id,
      prob_array[OFFSET(0)] AS {{culumn_name}}_Class_1,
      prob_array[OFFSET(1)] AS {{culumn_name}}_Class_2,
      prob_array[OFFSET(2)] AS {{culumn_name}}_Class_3,
      prob_array[OFFSET(3)] AS {{culumn_name}}_Class_4,
      prob_array[OFFSET(4)] AS {{culumn_name}}_Class_5,
      prob_array[OFFSET(5)] AS {{culumn_name}}_Class_6,
      prob_array[OFFSET(6)] AS {{culumn_name}}_Class_7,
      prob_array[OFFSET(7)] AS {{culumn_name}}_Class_8,
      prob_array[OFFSET(8)] AS {{culumn_name}}_Class_9,
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
, TEST AS (
  SELECT
    *
  FROM
    TEMPLATE
  {% for table_name in table_names %}
  LEFT JOIN
    {{table_name}}
  USING(id)
  {% endfor %}
)
{% for n_fold in range(0, 5) %}
, PRED_{{n_fold}} AS (
  SELECT
    id,
    predicted_target_probs,
  FROM
    ML.PREDICT
  (
    MODEL otto.{{model_name}}_stacking_model_fold{{n_fold}},
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




