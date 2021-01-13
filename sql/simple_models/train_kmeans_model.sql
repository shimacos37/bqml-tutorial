CREATE OR REPLACE MODEL `otto.kmeans_model`
OPTIONS(
  MODEL_TYPE="KMEANS",
  KMEANS_INIT_METHOD='KMEANS++',
  STANDARDIZE_FEATURES=True,
  DISTANCE_TYPE='EUCLIDEAN',
  EARLY_STOP=TRUE,
  MIN_REL_PROGRESS=0.0008,
  MAX_ITERATIONS=50,
  NUM_CLUSTERS=5
) AS

SELECT
  -- feature
  {% for n in range(1, 94) %} feat_{{n}},
  {% endfor %}  
FROM
  otto.train
UNION ALL (
SELECT
  -- feature
  {% for n in range(1, 94) %} feat_{{n}},
  {% endfor %}  
FROM
  otto.test
)