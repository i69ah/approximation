import scipy.stats as st
from pyspark import RDD
from pyspark.sql import DataFrame
from typing import Optional


def approx(data: DataFrame, preferred_error_bound: float) -> tuple:
    data_rdd = data.rdd.map(lambda x: x.overall)
    input_data_size = data_rdd.count()
    sbsa = determine_sbsa(data_rdd, input_data_size, preferred_error_bound)
    final_rdd: Optional[RDD] = None

    blocks_count = input_data_size // sbsa
    blocks_processed_count = 1
    for block in data_rdd.randomSplit([sbsa / input_data_size] * (input_data_size // sbsa)):
        if final_rdd is None:
            final_rdd = block.sample(False, .05)
        else:
            final_rdd.union(block.sample(False, .05))

        blocks_processed_count += 1

    error_bound = estimate_error_bound(
        input_data_size=input_data_size,
        frames_rdd=final_rdd,
        sbsa=sbsa,
        frame_size=final_rdd.count(),
        frames_count=blocks_count,
        blocks_count=blocks_count
    )

    return final_rdd, blocks_count, sbsa, error_bound


def get_sample_size_by_cohran(N: int, preferred_error_bound: float) -> int:
    p = .5  # целевая доля (худший сценарий - 50%)
    a = .95  # доверительный интервал
    z = st.norm.ppf(1 - (1 - a) / 2)
    n_0 = int(z**2*p*(1-p) / preferred_error_bound**2) + 1

    return int(n_0 / (1 + (n_0 - 1) / N))


def determine_sbsa(data: RDD, input_data_size: int, preferred_error_bound: float) -> int:
    cohran_sample_size = get_sample_size_by_cohran(input_data_size, preferred_error_bound)
    srs = data.sample(False, cohran_sample_size / input_data_size)
    DEFF = 1 + .5 * (cohran_sample_size - 1)
    varr_srs = srs.variance() * DEFF
    evarr = get_Evarr(input_data_size=input_data_size, preferred_error_bound=preferred_error_bound)
    divider = 2
    block_size = input_data_size
    while varr_srs > evarr:
        block_size = cohran_sample_size // divider
        srs = srs.sample(False, block_size // cohran_sample_size)
        cohran_sample_size = block_size
        DEFF = 1 + .5 * (cohran_sample_size - 1)
        varr_srs = srs.variance() * DEFF
        divider *= 3

    return block_size


def get_Evarr(input_data_size: int, preferred_error_bound: float) -> float:
    t = st.t.ppf((1 + 0.95) / 2, input_data_size - 1)

    return (preferred_error_bound/t)**2


def estimate_error_bound(input_data_size: int, frames_rdd: RDD,
                         sbsa: int, frame_size: int,
                         blocks_count: int,
                         frames_count: int) -> float:
    t = st.t.ppf((1 + 0.95) / 2, input_data_size - 1)
    var = frames_rdd.variance() / (frames_count * frame_size)
    var = (blocks_count / frames_count) * var

    return t * var ** (1 / 2)
