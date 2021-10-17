# Unit tests for validating datasets and analysis

import unittest
import paths
import numpy
from uuid import uuid4
from os.path import isfile


class TestingPathFunctions(unittest.TestCase):
    """
    Tests paths.py functions and makes sure that all files exist in the correct directories
    """
    
    def test_trefht_members(self):
        all_ds, xghg_ds, xaer_ds = paths.trefht_members()
        self.assertEqual(len(all_ds), 20)
        self.assertEqual(len(xghg_ds), 20)
        self.assertEqual(len(xaer_ds), 20)

        
    def test_trefht_members_raw(self):
        all_ds, xghg_ds, xaer_ds = paths.trefht_members_raw()
        self.assertEqual(len(all_ds), 40)
        self.assertEqual(len(xghg_ds), 40)
        self.assertEqual(len(xaer_ds), 40)
    
    
    def test_trefhtmn_members(self):
        all_ds, xghg_ds, xaer_ds = paths.trefhtmn_members()
        self.assertEqual(len(all_ds), 20)
        self.assertEqual(len(xghg_ds), 20)
        self.assertEqual(len(xaer_ds), 20)
    
    
    def test_trefhtmn_members_raw(self):
        all_ds, xghg_ds, xaer_ds = paths.trefhtmn_members_raw()
        self.assertEqual(len(all_ds), 40)
        self.assertEqual(len(xghg_ds), 40)
        self.assertEqual(len(xaer_ds), 40)
        
        
    def test_trefhtmx_members(self):
        all_ds, xghg_ds, xaer_ds = paths.trefhtmx_members()
        self.assertEqual(len(all_ds), 20)
        self.assertEqual(len(xghg_ds), 20)
        self.assertEqual(len(xaer_ds), 20)
        
        
    def test_trefhtmx_members_raw(self):
        all_ds, xghg_ds, xaer_ds = paths.trefhtmx_members_raw()
        self.assertEqual(len(all_ds), 40)
        self.assertEqual(len(xghg_ds), 40)
        self.assertEqual(len(xaer_ds), 40)
        
        
    def test_thresholds_members_1920_1950_ALL(self):
        all_ds, xghg_ds, xaer_ds = paths.thresholds_members_1920_1950_ALL()
        self.assertEqual(len(all_ds), 20)
        self.assertEqual(len(xghg_ds), 20)
        self.assertEqual(len(xaer_ds), 20)
        
    
    def test_control_download(self):
        controls = paths.control_downloads()
        self.assertEqual(len(controls), 18)
        
        
    def assert_experiment_definition_to_output_ratios(self, all_ds, xghg_ds, xaer_ds):
        all_exp_nums = list(set([num.split("_rNone_")[1][0:4] for num in all_ds]))
        xghg_exp_nums = list(set([num.split("_rNone_")[1][0:4] for num in xghg_ds]))
        xaer_exp_nums = list(set([num.split("_rNone_")[1][0:4] for num in xaer_ds]))
        self.assertEqual(len(all_ds) / len(all_exp_nums), 20, 
                         msg=f"{len(all_ds)} ALL datasets w/ {len(all_exp_nums)} definitions")
        self.assertEqual(len(xghg_ds) / len(xghg_exp_nums), 20, 
                         msg=f"{len(xghg_ds)} XGHG datasets w/ {len(xghg_exp_nums)} definitions")
        self.assertEqual(len(xaer_ds) / len(xaer_exp_nums), 20, 
                         msg=f"{len(xaer_ds)} XAER datasets w/ {len(xaer_exp_nums)} definitions")
    
    
    def test_heat_out_trefht_tmin_members_1920_1950_ALL(self):
        all_ds, xghg_ds, xaer_ds = paths.heat_out_trefht_tmin_members_1920_1950_ALL()
        self.assert_experiment_definition_to_output_ratios(all_ds, xghg_ds, xaer_ds)
        
    
    def test_heat_out_trefht_tmax_members_1920_1950_ALL(self):
        all_ds, xghg_ds, xaer_ds = paths.heat_out_trefht_tmax_members_1920_1950_ALL()
        self.assert_experiment_definition_to_output_ratios(all_ds, xghg_ds, xaer_ds)
        
    
    def equate_heat_out_trefht_min_max_averages_1920_1950(self):
        all_dsx, xghg_dsx, xaer_dsx = paths.heat_out_trefht_tmax_averages_1920_1950()
        all_dsn, xghg_dsn, xaer_dsn = paths.heat_out_trefht_tmin_averages_1920_1950()
        self.assertEqual(len(all_dsx), len(all_dsn), msg=f"{len(all_dsx)} ALL max average datasets w/ {len(all_dsn)} ALL min average datasets")
        self.assertEqual(len(xghg_dsx), len(xghg_dsn), msg=f"{len(xghg_dsx)} ALL max average datasets w/ {len(xghg_dsn)} ALL min average datasets")
        self.assertEqual(len(xaer_dsx), len(xaer_dsn), msg=f"{len(xaer_dsx)} ALL max average datasets w/ {len(xaer_dsn)} ALL min average datasets")
        
        
    def check_ds_files_exist(self, all_ds, xghg_ds=None, xaer_ds=None) -> bool:
        for path in all_ds:
            if not isfile(path):
                return False
        if xghg_ds:
            for path in xghg_ds:
                if not isfile(path):
                    return False
        if xaer_ds:
            for path in xaer_ds:
                if not isfile(path):
                    return False
        return True
        
        
    def test_trefht_members_exist(self):
        all_ds, xghg_ds, xaer_ds = paths.trefht_members()
        self.assertEqual(self.check_ds_files_exist(all_ds, xghg_ds, xaer_ds), True)

        
    def test_trefht_members_raw_exist(self):
        all_ds, xghg_ds, xaer_ds = paths.trefht_members_raw()
        self.assertEqual(self.check_ds_files_exist(all_ds, xghg_ds, xaer_ds), True)
    
    
    def test_trefhtmn_members_exist(self):
        all_ds, xghg_ds, xaer_ds = paths.trefhtmn_members()
        self.assertEqual(self.check_ds_files_exist(all_ds, xghg_ds, xaer_ds), True)
    
    
    def test_trefhtmn_members_raw_exist(self):
        all_ds, xghg_ds, xaer_ds = paths.trefhtmn_members_raw()
        self.assertEqual(self.check_ds_files_exist(all_ds, xghg_ds, xaer_ds), True)
        
        
    def test_trefhtmx_members_exist(self):
        all_ds, xghg_ds, xaer_ds = paths.trefhtmx_members()
        self.assertEqual(self.check_ds_files_exist(all_ds, xghg_ds, xaer_ds), True)
        
        
    def test_trefhtmx_members_raw_exist(self):
        all_ds, xghg_ds, xaer_ds = paths.trefhtmx_members_raw()
        self.assertEqual(self.check_ds_files_exist(all_ds, xghg_ds, xaer_ds), True)
        
        
    def test_thresholds_members_1920_1950_ALL_exist(self):
        all_ds, xghg_ds, xaer_ds = paths.thresholds_members_1920_1950_ALL()
        self.assertEqual(self.check_ds_files_exist(all_ds, xghg_ds, xaer_ds), True)
    
    
    def test_control_download_exist(self):
        controls = paths.control_downloads()
        self.assertEqual(self.check_ds_files_exist(controls), True)
    
    def test_heat_out_trefht_tmin_members_1920_1950_exist(self):
        all_ds, xghg_ds, xaer_ds = paths.heat_out_trefht_tmin_members_1920_1950_ALL()
        self.assertEqual(self.check_ds_files_exist(all_ds, xghg_ds, xaer_ds), True)
        
    
    def test_heat_out_trefht_tmax_members_1920_1950_exist(self):
        all_ds, xghg_ds, xaer_ds = paths.heat_out_trefht_tmax_members_1920_1950_ALL()
        self.assertEqual(self.check_ds_files_exist(all_ds, xghg_ds, xaer_ds), True)
        
        
    def test_heat_out_trefht_tmin_members_1920_1950_control_exist(self):
        all_ds, xghg_ds, xaer_ds = paths.heat_out_trefht_tmin_members_1920_1950_CONTROL()
        self.assertEqual(self.check_ds_files_exist(all_ds, xghg_ds, xaer_ds), True)
        
        
    def test_heat_out_trefht_tmax_members_1920_1950_control_exist(self):
        all_ds, xghg_ds, xaer_ds = paths.heat_out_trefht_tmax_members_1920_1950_CONTROL()
        self.assertEqual(self.check_ds_files_exist(all_ds, xghg_ds, xaer_ds), True)
        
        
class TestingAnalysisFunctionality(unittest.TestCase):
    
    
    def test_dask_xarray_averaging(self):
        # Create 3 random time, lat, lon arrays
        # Average them together to get control comparison
        # Output each to a uuid.nc
        # Import all of them using new open_mfdataset averaging technique
        # Delete all uuid.nc files
        # Compare to control (should be equal)
        pass
        
    
    def test_xarray_time_slicing(self):
        # Create time, lat, lon array (non random, using just 1s maybe)
        # Assert average equal constant
        # Slice
        # Assert sliced average equals constant
        pass
    
        
if __name__ == '__main__':
    unittest.main()

