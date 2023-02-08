import pyfolio as pf


def analytics(rets, benchmark_rets):
    """
    Generate a number of tear sheets that are useful for analyzing a strategy’s performance.
    :param rets: Daily returns of the strategy, noncumulative
    :param benchmark_rets: Daily noncumulative returns of the benchmark
    :return: None
    """
    pf.create_full_tear_sheet(returns=rets, benchmark_rets=benchmark_rets)
