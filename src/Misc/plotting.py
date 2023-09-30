import matplotlib.pyplot as plot
from scipy.signal import savgol_filter


"""
 Plots between episodes (only for quantitatively purposes)
 
 author(s): Moritz T.
"""


class Plotter:
    def __init__(self, params):
        self.params = params

    def plot_experiment(self, parameter1, parameter2):
        x = range(parameter1)
        filter_window_size = 49
        if len(parameter2) >= filter_window_size:
            if self.params.plot_filter:
                y = savgol_filter(parameter2, filter_window_size, 3)
        else:
            y = parameter2
        plot.plot(x, y)
        plot.title("Progress")
        plot.xlabel("episode")
        plot.ylabel("undiscounted return")
        plot.show()
    
    def occasionally_plot(self, nr_episode, returns):
        if self.params.plot and nr_episode > 0 and nr_episode % self.params.plotting_frequency == 0:
            self.plot_experiment(nr_episode, returns)