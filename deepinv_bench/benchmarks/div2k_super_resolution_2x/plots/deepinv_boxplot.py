from deepinv_bench.benchmarks.plots.deepinv_boxplot import Plot as _Plot


class Plot(_Plot):
    # benchopt reads `name` by AST scan, so it must stay a literal here.
    name = "Deepinv Boxplot"
