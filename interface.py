from pyspark.sql import SparkSession, DataFrame, SQLContext
from .approximation import approx


class Approximator:
    def __init__(self, connection_data: dict):
        spark = SparkSession.builder.config("spark.jars", "./postgresql-42.6.0.jar") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true").config("spark.driver.memory", "1g") \
            .getOrCreate()
        self.sql_context: SQLContext = SQLContext(sparkContext=spark.sparkContext, sparkSession=spark)
        self.connection_data = connection_data

    def approx_sum(self, col_name: str, preferred_error_bound: float) -> tuple[float, float]:
        if preferred_error_bound < 0 or preferred_error_bound > 1:
            raise ValueError('preferred_error_bound must be between 0 and 1')

        data_frame: DataFrame = self.sql_context.read.format("jdbc").options(**self.connection_data).load()
        sample, blocks_count, sbsa, error_bound = approx(data_frame.select(col_name),
                                                         preferred_error_bound=preferred_error_bound)
        frames_count = sample.count()
        sample_sum = blocks_count * sbsa * sample.sum() / frames_count

        return sample_sum, sample_sum * error_bound

    def approx_avg(self, col_name: str, preferred_error_bound: float) -> tuple[float, float]:
        if preferred_error_bound < 0 or preferred_error_bound > 1:
            raise ValueError('preferred_error_bound must be between 0 and 1')

        data_frame: DataFrame = self.sql_context.read.format("jdbc").options(**self.connection_data).load()
        sample, _, _, error_bound = approx(data_frame.select(col_name), preferred_error_bound=preferred_error_bound)

        return sample.mean(), sample.mean() * error_bound

    def approx_prob(self, col_name: str, value, preferred_error_bound: float) -> tuple[float, float]:
        if preferred_error_bound < 0 or preferred_error_bound > 1:
            raise ValueError('preferred_error_bound must be between 0 and 1')

        data_frame: DataFrame = self.sql_context.read.format("jdbc").options(**self.connection_data).load()
        sample, _, _, error_bound = approx(data_frame.select(col_name), preferred_error_bound=preferred_error_bound)
        filtered = sample.filter(lambda x: x == value)
        result = filtered / sample.count()

        return result, result * error_bound
