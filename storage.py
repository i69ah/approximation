from pyspark.sql import SparkSession, DataFrame, SQLContext


class Storage:
    def __init__(
            self,
            jdbc_url: str,
            username: str,
            password: str,
            driver: str,
            table: str,
            **kwargs
    ):
        spark = SparkSession.builder.config("spark.jars", "./postgresql-42.6.0.jar") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true").config("spark.driver.memory", "1g") \
            .getOrCreate()
        self.sql_context: SQLContext = SQLContext(sparkContext=spark.sparkContext, sparkSession=spark)
        self.jdbc_url: str = jdbc_url
        self.username: str = username
        self.password: str = password
        self.driver: str = driver
        self.table = table
        self.other_properties: dict = kwargs

    def get_data_frame(self) -> DataFrame:
        return self.sql_context.read.format("jdbc").options(
            url=self.jdbc_url,
            user=self.username,
            password=self.password,
            driver=self.driver,
            dbtable=self.table,
            **self.other_properties
        ).load()
