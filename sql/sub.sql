SELECT
  id,
  prob_array[OFFSET(0)] AS Class_1,
  prob_array[OFFSET(1)] AS Class_2,
  prob_array[OFFSET(2)] AS Class_3,
  prob_array[OFFSET(3)] AS Class_4,
  prob_array[OFFSET(4)] AS Class_5,
  prob_array[OFFSET(5)] AS Class_6,
  prob_array[OFFSET(6)] AS Class_7,
  prob_array[OFFSET(7)] AS Class_8,
  prob_array[OFFSET(8)] AS Class_9,
FROM (
  SELECT
    id,
    ARRAY_AGG(prob) AS prob_array,
  FROM otto.{{pred_table_id}}
  GROUP BY id
)
