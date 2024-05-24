from metrics import *
import numpy as np
import unittest

class TestSiteMetrics(unittest.TestCase):

    def create_test_grid(self, data):

        test_grid = np.zeros((110, 210))
        for point in data:
            test_grid[point[0], point[1]] = point[2]

        return test_grid
    def get_test_grid_1(self):

        return self.create_test_grid([(55, 45, 10), (100, 50, 15), (55, 34, 5)])

    def get_test_grid_2(self):
        return self.create_test_grid([(55, 45, 5), (100, 50, 20), (55, 34, 2)])

    def get_test_grid_3(self):
        return self.create_test_grid([(55, 45, 7), (100, 50, 12), (55, 34, 3)])

    def get_test_grid_4(self):
        return self.create_test_grid([(55, 45, 18), (100, 50, 0), (55, 34, 0)])

    def get_test_grid_5(self):

        return self.create_test_grid([(55, 45, 11), (100, 50, 12), (55, 34, 6)])

    def get_test_grid_6(self):
        return self.create_test_grid([(55, 45, 10), (100, 50, 17), (55, 34, 2)])

    def get_test_grid_7(self):
        return self.create_test_grid([(55, 45, 10), (100, 50, 13), (55, 34, 4)])

    def get_test_grid_8(self):
        return self.create_test_grid([(55, 45, 17), (100, 50, 2), (55, 34, 1)])

    def test_get_grid_cell(self):
        # test that the get grid cell function works properly for two different resolutions

        self.assertEquals(getGridCell(-60,135, 0.5), (0, 0))
        self.assertEquals(getGridCell(-60, 136.1, 0.5), (0, 2))
        self.assertEquals(getGridCell(-60, 136.1, 1), (0, 1))

    def test_get_site_values(self):

        # test that the get_site_values function only returns non-zero values for non-zero cells

        test_grid = np.zeros((210,110))

        site_vals = get_site_values(test_grid)

        self.assertEquals(site_vals.tolist(), [0 for i in range(len(site_vals))])

        cell = getGridCell(-9.81386294, 160.1563795, 0.5)
        print(cell)
        test_grid[*cell] = 1
        site_vals = get_site_values(test_grid)

        self.assertEquals(site_vals[0], 1)

        for i in range(1, len(site_vals)):
            if site_vals[i] != 0:
                self.assertEquals(site_vals[i], 1)
                self.assertEquals(getGridCell(*sites[i], 0.5), cell)


    def test_ensemble_statistics_no_sites(self):

        ## Generate test data

        # outputs
        test_grid_1 = self.get_test_grid_1()
        test_grid_2 = self.get_test_grid_2()
        test_grid_3 = self.get_test_grid_3()
        test_grid_4 = self.get_test_grid_4()
        outputs = [test_grid_1, test_grid_3, test_grid_4, test_grid_2]

        # predictions
        test_grid_5 = self.get_test_grid_5()
        test_grid_6 = self.get_test_grid_6()
        test_grid_7 = self.get_test_grid_7()
        test_grid_8 = self.get_test_grid_8()

        predictions= [test_grid_8, test_grid_5, test_grid_6, test_grid_7]

        storm_statistics = compute_ensemble_statistics(outputs)
        unet_statistics = compute_ensemble_statistics(predictions)

        self.assertEquals(storm_statistics["Quantiles"].shape, (5, 110, 210))
        self.assertEquals(unet_statistics["Quantiles"].shape, (5, 110, 210))

        self.assertEquals(storm_statistics["Quantiles"][:, 55, 45].tolist(), [5, 6.5, 8.5, 12, 18])
        self.assertEquals(unet_statistics["Quantiles"][:, 55, 45].tolist(), [10, 10, 10.5, 12.5, 17])

    def get_outputs_and_predictions(self):
        # outputs
        test_grid_1 = self.get_test_grid_1()
        test_grid_2 = self.get_test_grid_2()
        test_grid_3 = self.get_test_grid_3()
        test_grid_4 = self.get_test_grid_4()
        outputs = [test_grid_1, test_grid_3, test_grid_4, test_grid_2]

        # predictions
        test_grid_5 = self.get_test_grid_5()
        test_grid_6 = self.get_test_grid_6()
        test_grid_7 = self.get_test_grid_7()
        test_grid_8 = self.get_test_grid_8()

        predictions = [test_grid_8, test_grid_5, test_grid_6, test_grid_7]

        return outputs, predictions
    def get_test_statistics_and_metrics(self):

        outputs, predictions = self.get_outputs_and_predictions()
        storm_statistics = compute_ensemble_statistics(outputs)
        unet_statistics = compute_ensemble_statistics(predictions)

        all_metrics = compute_metrics(outputs, predictions, storm_statistics, unet_statistics, "Custom-UNet")

        return outputs, predictions, storm_statistics, unet_statistics, all_metrics

    def test_ensemble_figures(self):
        ## Generate test data

        outputs, predictions, storm_statistics, unet_statistics, all_metrics = self.get_test_statistics_and_metrics()

        make_figures(outputs, predictions)

    def test_ks_statistics_map(self):

        outputs, predictions, storm_statistics, unet_statistics, all_metrics = self.get_test_statistics_and_metrics()

        ks_statistic_map(all_metrics)

    def test_quantile_maps(self):
        outputs, predictions, storm_statistics, unet_statistics, all_metrics = self.get_test_statistics_and_metrics()
        plot_quantile_maps(storm_statistics, unet_statistics)

    def test_relative_change_map(self):
        outputs, predictions = self.get_outputs_and_predictions()
        change_map = relative_change(outputs[0], outputs[1])

        self.assertEquals(change_map[100, 50], .2)
        self.assertEquals(change_map[100, 51], 0)

    def test_relative_change_error_map(self):
        outputs, predictions = self.get_outputs_and_predictions()
        error_map, total_error = compute_changes_between_2_samples(outputs, predictions, 0, 1)
        display_example_relative_change_error(error_map)


if __name__ == "__main__":
    unittest.main()

