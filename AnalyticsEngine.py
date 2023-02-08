import pyfolio as pf


def analytics(rets, benchmark_rets):
    """
    Generate a number of tear sheets that are useful for analyzing a strategy’s performance
    :param rets: daily returns of the strategy, noncumulative
    :param benchmark_rets: daily returns of the benchmark, noncumulative
    :return: None
    """
    pf.create_full_tear_sheet(returns=rets, benchmark_rets=benchmark_rets)
