from benchopt.plotting.base import BasePlot

UNITS = {
    "PSNR": "dB",
}


class Plot(BasePlot):
    name = "Boxplot Deepinv"
    type = "boxplot"
    dropdown = {
        "metric": [
            "PSNR",
            "NIQE",
        ],
    }

    def get_metadata(self, df, metric):
        unit = UNITS.get(metric, None)
        if unit is not None:
            unit = f" ({unit})"
        return {
            "title": f"Deblurring on Div2K\nMetric: {metric}",
            "xlabel": "Solver",
            "ylabel": f"{metric}{unit}",
        }

    def plot(self, df, metric):

        plot_data = []
        for solver, df_solver in df.groupby('solver_name'):
            mean = df_solver[f'objective_{metric}'].mean()
            std = df_solver[f'objective_{metric}_std'].mean()
            y = [[mean - std, mean, mean + std]]
            x = [solver]

            plot_data.append({
                "x": x,
                "y": y,
                "label": solver,
                "color": self.get_style(solver)["color"],
            })

        return plot_data
