from google.cloud import bigquery
from google.api_core.exceptions import NotFound


class BQClient:
    def __init__(self, project, default_dataset=None):
        self.project = project
        self.default_dataset = default_dataset
        self.client = bigquery.Client(project=project)

    def create_dataset(self, dataset_id):
        dataset = self.client.dataset(dataset_id)
        dataset.location = "US"
        dataset = self.client.create_dataset(dataset)
        print(f"Created dataset {self.client.project}.{dataset.dataset_id}")

    def exist_table(self, dataset_id, table_id):
        ref = self.client.dataset(dataset_id).table(table_id)
        try:
            self.client.get_table(ref)
            print(f"table: {dataset_id}.{table_id} exists.")
            return True
        except NotFound:
            print(f"table: {dataset_id}.{table_id} not found.")
            return False

    def delete_table(self, dataset_id, table_id):
        if not self.exist_table(dataset_id, table_id):
            return

        ref = self.client.dataset(dataset_id).table(table_id)
        self.client.delete_table(ref)
        print(f"table: {dataset_id}.{table_id} was deleted.")

    def delete_model(self, dataset_id, model_id):
        model_id = f"{dataset_id}.{model_id}"
        try:
            self.client.delete_model(model_id)
            print(f"model: {model_id} was deleted.")
        except NotFound:
            print(f"model: {model_id} was not found.")

    def create_table(self, dataset_id, table_id, setting, description=""):
        dataset_ref = self.client.dataset(dataset_id)
        table_ref = dataset_ref.table(table_id)

        schema = [
            bigquery.SchemaField(field["name"], field["type"], mode=field["mode"])
            for field in setting["schema"]["fields"]
        ]
        table = bigquery.Table(table_ref, schema)
        table.description = description

        if setting.get("timePartitioning"):
            tp = bigquery.table.TimePartitioning()
            if setting.get("timePartitioning").get("field"):
                tp.field = setting["timePartitioning"]["field"]
            table.time_partitioning = tp

        if setting.get("clustering"):
            table.clustering_fields = setting["clustering"]["fields"]

        self.client.create_table(table)
        print(f"Created table {table.project}.{table.dataset_id}.{table.table_id}")

    def execute_query(self, query):
        if self.default_dataset is not None:
            default_dataset = self.project + "." + self.default_dataset
        else:
            default_dataset = None

        job_config = bigquery.job.QueryJobConfig(default_dataset=default_dataset)
        job_config.use_legacy_sql = False

        insert_job = self.client.query(query, job_config=job_config)
        insert_job.result()
        print("Execute query.")

    def insert_table(
        self, dataset_id, table_id, query, write_disposition="WRITE_TRUNCATE"
    ):
        dataset_ref = self.client.dataset(dataset_id)
        table_ref = dataset_ref.table(table_id)

        job_config = bigquery.job.QueryJobConfig()
        job_config.destination = table_ref
        job_config.use_legacy_sql = False
        job_config.write_disposition = write_disposition
        job_config.allow_large_results = True

        insert_job = self.client.query(query, job_config=job_config)
        insert_job.result()
        print(f"Insert table {dataset_id}.{table_id}")

    def copy_table(self, src_project, src_dataset, tgt_dataset, table_id):
        if self.exist_table(tgt_dataset, table_id):
            self.delete_table(tgt_dataset, table_id)
        src_table_id = f"{src_project}.{src_dataset}.{table_id}"
        tgt_table_id = f"{self.project}.{tgt_dataset}.{table_id}"
        copy_job = self.client.copy_table(src_table_id, tgt_table_id)
        copy_job.result()
        print("A copy of the table created.")

    def update_table_description(self, dataset_id, table_id, description=""):
        dataset_ref = self.client.dataset(dataset_id)
        table_ref = dataset_ref.table(table_id)
        table = self.client.get_table(table_ref)

        table.description = description
        self.client.update_table(table, ["description"])

    def update_model_description(self, dataset_id, model_id, description=""):
        dataset_ref = self.client.dataset(dataset_id)
        model_ref = dataset_ref.model(model_id)
        model = self.client.get_model(model_ref)

        model.description = description
        self.client.update_model(model, ["description"])
        print(f"Update model description: {description}")

    def fetch_table_description(self, dataset_id, table_id):
        dataset_ref = self.client.dataset(dataset_id)
        table_ref = dataset_ref.table(table_id)
        table = self.client.get_table(table_ref)

        return table.description

    def fetch(self, query):
        df = self.client.query(query).to_dataframe()
        return df

    def insert_df_to_table(self, df, dataset_id, table_id):
        table_id_ = f"{dataset_id}.{table_id}"

        job = self.client.load_table_from_dataframe(df, table_id_)
        job.result()
        print(f"Insert table {dataset_id}.{table_id}")
