import unittest

from plotanalysis import PlotAnalysis

class PlotAnalysisTest(unittest.TestCase):

    def test_plot_analysis(self):
        # Create a plot analysis object
        plot_analysis = PlotAnalysis()

        # Set the data
        plot_analysis.data = [
            (1, 2),
            (3, 4),
            (5, 6)
        ]

        # Set the x-axis label
        plot_analysis.x_axis_label = "X-Axis"

        # Set the y-axis label
        plot_analysis.y_axis_label = "Y-Axis"

        # Plot the data
        plot_analysis.plot()

        # Check that the plot was created
        self.assertTrue(plot_analysis.plot_exists())

        # Check that the x-axis label is correct
        self.assertEqual(plot_analysis.x_axis_label, "X-Axis")

        # Check that the y-axis label is correct
        self.assertEqual(plot_analysis.y_axis_label, "Y-Axis")

