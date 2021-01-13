CREATE OR REPLACE MODEL `otto.dnn_kmeans_log_model_fold{{n_fold}}`
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
)
  
  SELECT
    * except(id, CENTROID_ID, NEAREST_CENTROIDS_DISTANCE),
    -- fold
    CASE 
        WHEN MOD(id, 5) = {{n_fold}} THEN True
        ELSE False
    END AS is_eval,
    -- kmeans_feature
    (SELECT DISTANCE FROM UNNEST(NEAREST_CENTROIDS_DISTANCE) WHERE CENTROID_ID = 1) AS distance_1,
    (SELECT DISTANCE FROM UNNEST(NEAREST_CENTROIDS_DISTANCE) WHERE CENTROID_ID = 2) AS distance_2,
    (SELECT DISTANCE FROM UNNEST(NEAREST_CENTROIDS_DISTANCE) WHERE CENTROID_ID = 3) AS distance_3,
    (SELECT DISTANCE FROM UNNEST(NEAREST_CENTROIDS_DISTANCE) WHERE CENTROID_ID = 4) AS distance_4,
    (SELECT DISTANCE FROM UNNEST(NEAREST_CENTROIDS_DISTANCE) WHERE CENTROID_ID = 5) AS distance_5,
  FROM KMEANS_PRED