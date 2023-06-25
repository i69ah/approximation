from pyspark.sql import DataFrame
from .approximation import approx
from .storage import Storage


class Approximator:
    def __init__(self, connection_data: dict):
        self.storage: Storage = Storage(
            jdbc_url=connection_data["jdbc_url"],
            username=connection_data["username"],
            password=connection_data["password"],
            driver=connection_data["driver"],
            table=connection_data["table"],
            **connection_data["other_properties"]
        )

    def approx_sum(self, col_name: str, preferred_error_bound: float) -> tuple[float, float]:
        if preferred_error_bound < 0 or preferred_error_bound > 1:
            raise ValueError('preferred_error_bound must be between 0 and 1')

        data_frame: DataFrame = self.storage.get_data_frame()
        sample, blocks_count, sbsa, error_bound = approx(data_frame.select(col_name),
                                                         preferred_error_bound=preferred_error_bound)
        frames_count = sample.count()
        sample_sum = blocks_count * sbsa * sample.sum() / frames_count

        return sample_sum, sample_sum * error_bound

    def approx_avg(self, col_name: str, preferred_error_bound: float) -> tuple[float, float]:
        if preferred_error_bound < 0 or preferred_error_bound > 1:
            raise ValueError('preferred_error_bound must be between 0 and 1')

        data_frame: DataFrame = self.storage.get_data_frame()
        sample, _, _, error_bound = approx(data_frame.select(col_name), preferred_error_bound=preferred_error_bound)

        return sample.mean(), sample.mean() * error_bound

    def approx_prob(self, col_name: str, value, preferred_error_bound: float) -> tuple[float, float]:
        if preferred_error_bound < 0 or preferred_error_bound > 1:
            raise ValueError('preferred_error_bound must be between 0 and 1')

        data_frame: DataFrame = self.storage.get_data_frame()
        sample, _, _, error_bound = approx(data_frame.select(col_name), preferred_error_bound=preferred_error_bound)
        filtered = sample.filter(lambda x: x == value)
        result = filtered / sample.count()

        return result, result * error_bound
