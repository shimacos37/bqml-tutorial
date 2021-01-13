CREATE OR REPLACE TABLE otto.{{table_id}} (
  id INT64 NOT NULL,
  n_fold INT64 NOT NULL,
  target INT64 NOT NULL,
  label INT64 NOT NULL,
  prob FLOAT64 NOT NULL,
)
AS
WITH KMEANS_PRED AS (
  SELECT
    *
  FROM
    ML.PREDICT
  (
    MODEL otto.kmeans_log_model,
    (
      SELECT
        id,
        CAST(SPLIT(target, '_')[OFFSET(1)] AS INT64) AS target,
        -- feature
        {% for n in range(1, 94) %} LN(feat_{{n}} + 1) AS feat_{{n}},
        {% endfor %}
      FROM
        otto.train
    )
  )
),
TRAIN AS (
  SELECT
    * except(CENTROID_ID, NEAREST_CENTROIDS_DISTANCE),
    MOD(id, 5) AS n_fold,
    -- kmeans_feature
    (SELECT DISTANCE FROM UNNEST(NEAREST_CENTROIDS_DISTANCE) WHERE CENTROID_ID = 1) AS distance_1,
    (SELECT DISTANCE FROM UNNEST(NEAREST_CENTROIDS_DISTANCE) WHERE CENTROID_ID = 2) AS distance_2,
    (SELECT DISTANCE FROM UNNEST(NEAREST_CENTROIDS_DISTANCE) WHERE CENTROID_ID = 3) AS distance_3,
    (SELECT DISTANCE FROM UNNEST(NEAREST_CENTROIDS_DISTANCE) WHERE CENTROID_ID = 4) AS distance_4,
    (SELECT DISTANCE FROM UNNEST(NEAREST_CENTROIDS_DISTANCE) WHERE CENTROID_ID = 5) AS distance_5,
  FROM KMEANS_PRED
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

