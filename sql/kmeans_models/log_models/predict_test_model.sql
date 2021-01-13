CREATE OR REPLACE TABLE otto.{{table_id}} (
  id INT64 NOT NULL,
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
        -- feature
        {% for n in range(1, 94) %} LN(feat_{{n}} + 1) AS feat_{{n}},
        {% endfor %}
      FROM
        otto.test
    )
  )
),
TEST AS (
  SELECT
    * except(CENTROID_ID, NEAREST_CENTROIDS_DISTANCE),
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

