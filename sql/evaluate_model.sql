CREATE TABLE IF NOT EXISTS otto.eval (
  model_name STRING NOT NULL,
  n_fold INT64 NOT NULL,
  precision FLOAT64 NOT NULL,
  recall FLOAT64 NOT NULL,
  accuracy FLOAT64 NOT NULL,
  f1_score FLOAT64 NOT NULL,
  log_loss FLOAT64 NOT NULL,
  roc_auc FLOAT64 NOT NULL,
)
;


MERGE
  eval AS target
USING
(
  WITH TRAIN AS (
    SELECT
      * except (target),
      CAST(SPLIT(target, '_')[OFFSET(1)] AS INT64) AS target,
      MOD(id, 5) AS n_fold,
    FROM
      otto.train
  )
  {% for n_fold in range(0, 5) %}
  , EVAL_{{n_fold}} AS (
    SELECT
      "{{model_name}}" AS model_name,
      {{n_fold}} AS n_fold,
      *,
    FROM
      ML.EVALUATE
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
  SELECT * FROM EVAL_0
  UNION ALL
  SELECT * FROM EVAL_1
  UNION ALL
  SELECT * FROM EVAL_2
  UNION ALL
  SELECT * FROM EVAL_3
  UNION ALL
  SELECT * FROM EVAL_4
) AS source
  ON target.model_name = source.model_name AND target.n_fold = source.n_fold
WHEN MATCHED THEN UPDATE SET
  precision=source.precision,
  recall=source.recall,
  accuracy=source.accuracy,
  f1_score=source.f1_score,
  log_loss=source.log_loss,
  roc_auc=source.roc_auc
WHEN NOT MATCHED THEN INSERT ROW