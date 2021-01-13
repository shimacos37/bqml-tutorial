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
    MOD(id, 5) AS n_fold,
    CAST(SPLIT(target, '_')[OFFSET(1)] AS INT64) AS target,
    {% for n in range(1, 94) %} feat_{{n}},
    {% endfor %}
  FROM
    otto.train
)
, PRED_0 AS (
  SELECT
    id,
    n_fold,
    target,
    output,
  FROM
    ML.PREDICT
  (
    MODEL otto.{{model_name}}_fold0,
    (
      SELECT
        id,
        n_fold,
        target,
        [ (feat_1 - mean_1) / std_1
          {% for n in range(2, 94) %}, (feat_{{n}} - mean_{{n}}) / std_{{n}}
          {% endfor %}
        ] AS input
      FROM (
        SELECT
          *,
          {% for key, mean in means_fold0 %} {{mean}} AS {{key}},
          {% endfor %}
          {% for key, std in stds_fold0 %} {{std}} AS {{key}},
          {% endfor %}
        FROM
          TRAIN
        WHERE
            n_fold = 0
      )
    )
  )
)
, PRED_1 AS (
  SELECT
    id,
    n_fold,
    target,
    output,
  FROM
    ML.PREDICT
  (
    MODEL otto.{{model_name}}_fold1,
    (
      SELECT
        id,
        n_fold,
        target,
        [ (feat_1 - mean_1) / std_1
          {% for n in range(2, 94) %}, (feat_{{n}} - mean_{{n}}) / std_{{n}}
          {% endfor %}
        ] AS input
      FROM (
        SELECT
          *,
          {% for key, mean in means_fold1 %} {{mean}} AS {{key}},
          {% endfor %}
          {% for key, std in stds_fold1 %} {{std}} AS {{key}},
          {% endfor %}
        FROM
          TRAIN
        WHERE
            n_fold = 1
      )
    )
  )
)
, PRED_2 AS (
  SELECT
    id,
    n_fold,
    target,
    output,
  FROM
    ML.PREDICT
  (
    MODEL otto.{{model_name}}_fold2,
    (
      SELECT
        id,
        n_fold,
        target,
        [ (feat_1 - mean_1) / std_1
          {% for n in range(2, 94) %}, (feat_{{n}} - mean_{{n}}) / std_{{n}}
          {% endfor %}
        ] AS input
      FROM (
        SELECT
          *,
          {% for key, mean in means_fold2 %} {{mean}} AS {{key}},
          {% endfor %}
          {% for key, std in stds_fold2 %} {{std}} AS {{key}},
          {% endfor %}
        FROM
          TRAIN
        WHERE
            n_fold = 2
      )
    )
  )
)
, PRED_3 AS (
  SELECT
    id,
    n_fold,
    target,
    output,
  FROM
    ML.PREDICT
  (
    MODEL otto.{{model_name}}_fold3,
    (
      SELECT
        id,
        n_fold,
        target,
        [ (feat_1 - mean_1) / std_1
          {% for n in range(2, 94) %}, (feat_{{n}} - mean_{{n}}) / std_{{n}}
          {% endfor %}
        ] AS input
      FROM (
        SELECT
          *,
          {% for key, mean in means_fold3 %} {{mean}} AS {{key}},
          {% endfor %}
          {% for key, std in stds_fold3 %} {{std}} AS {{key}},
          {% endfor %}
        FROM
          TRAIN
        WHERE
            n_fold = 3
      )
    )
  )
)
, PRED_4 AS (
  SELECT
    id,
    n_fold,
    target,
    output,
  FROM
    ML.PREDICT
  (
    MODEL otto.{{model_name}}_fold4,
    (
      SELECT
        id,
        n_fold,
        target,
        [ (feat_1 - mean_1) / std_1
          {% for n in range(2, 94) %}, (feat_{{n}} - mean_{{n}}) / std_{{n}}
          {% endfor %}
        ] AS input
      FROM (
        SELECT
          *,
          {% for key, mean in means_fold4 %} {{mean}} AS {{key}},
          {% endfor %}
          {% for key, std in stds_fold4 %} {{std}} AS {{key}},
          {% endfor %}
        FROM
          TRAIN
        WHERE
            n_fold = 4
      )
    )
  )
)

SELECT
  id,
  n_fold,
  target,
  label + 1 AS label,
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
UNNEST(output) AS prob
WITH OFFSET AS label
