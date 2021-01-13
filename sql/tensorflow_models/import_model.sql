CREATE OR REPLACE MODEL `otto.tensorflow_{{model_name}}_model_fold{{n_fold}}`
OPTIONS(
  MODEL_TYPE="TENSORFLOW",
  MODEL_PATH='gs://{{bucket_name}}/{{model_name}}/fold{{n_fold}}/*'
)