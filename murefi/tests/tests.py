import unittest
import numpy
import pandas
import pathlib
import bletl
from murefi import ParameterMapping, Timeseries, Replicate, Dataset


class ParameterMapTest(unittest.TestCase):
    def test_fitpars_bounds(self):
        df = pandas.read_csv(pathlib.Path('ParTest.csv'), sep=';')
        actual = ParameterMapping(df).fitpars_bounds
        expected = {'test1A': (1,2), 'test1B': (1,2), 'test2C': (1,2), 'test2D': (3,4)}
        self.assertDictEqual(actual, expected)
        return
    
    def test_fitarray(self):
        df = pandas.read_csv(pathlib.Path('ParTest.csv'), sep=';')
        actual = ParameterMapping(df).fitpars_array
        expected = ['test1A', 'test1B', 'test2C', 'test2D']
        self.assertEqual(actual.tolist(), expected)
        return
    
    def test_parameters_dic(self):
        df = pandas.read_csv(pathlib.Path('ParTest.csv'), sep=';')
        actual = ParameterMapping(df).parameters_dic
        expected = {'A01': ['test1A', 'test1B', '1', '1', 1, 1, 1], 'B02': ['2', 'test1B', 'test2C', 'test2D', 2, 2, 2]} 
        self.assertDictEqual(actual, expected)
        return
    
    def test_bounds_list(self):
        df = pandas.read_csv(pathlib.Path('ParTest.csv'), sep=';')
        actual = ParameterMapping(df).bounds_list
        expected = [(1,2), (1,2), (1,2), (3,4)]
        self.assertEqual(actual, expected)
        return

    
def create_dataset_object(bletl_data, par_dic):
    '''Function to create a dataset object for all replicates
    Args:
        bletl_data:     dictionary with 'FilterTimeSeries' objects (technicially 
                        calibrated data) from bletl
        par_dic: dictionary containing well IDs as keys and the
                        parameters for fitting provided by the user
    Returns: Dataset object (containing Replicate objects for each well)
    '''
    dataset = Dataset()
    for well_id in par_dic.keys():
        bs_x = bletl_data['BS10'].time[well_id]
        bs_y = bletl_data['BS10'].value[well_id].multiply(0.1187).add(0.5866)
        bs_ts = Timeseries('BS', list(bs_x), list(bs_y))
        rep = Replicate(well_id)
        rep['BS'] = bs_ts
        dataset[well_id] = rep
    return dataset


class DatatypesTest(unittest.TestCase):
    def test_dataset(self):
        datafile = pathlib.Path('SiLA_Coryne_Standard_20181026_150350.csv')
        bldata = bletl.parse(datafile)
        calibration_parameters = {
            'cal_0': 65.91,
            'cal_100': 40.60,
            'phi_min': 57.45,
            'phi_max': 18.99,
            'pH_0': 6.46,
            'dpH': 0.56,
        }
        bldata.calibrate(calibration_parameters)
        data = bldata.calibrated_data
        pardic = ParameterMapping(pandas.read_csv(pathlib.Path('ParTest.csv'), sep=';')).parameters_dic
        actual = create_dataset_object(data, pardic)
        self.assertEqual(list(actual.keys()), ['A01', 'B02'])
        self.assertEqual(len(actual['A01']['BS']), len(data['BS10'].value['A01'])) #all data from bletl to Timeseriesin dataset
        return
        
        
if __name__ == '__main__':
    unittest.main(exit=False)