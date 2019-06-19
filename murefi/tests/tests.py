import collections
import unittest
import numpy
import pandas
import pathlib
import scipy.stats as stats

import murefi


dir_testfiles = pathlib.Path(pathlib.Path(__file__).absolute().parent, 'testfiles')


class ParameterMapTest(unittest.TestCase):
    def test_init(self):
        map_df = pandas.read_csv(pathlib.Path(dir_testfiles, 'ParTest.csv'), sep=';')
        map_df.set_index(map_df.columns[0])
        bounds = dict(
            S_0=(1,2),
            X_0=(3,4),
            mue_max=(5,6),
            K_S=(7,8),
            t_lag=(9,10),
            t_acc=(11,12),
            test1A=(1,3)
        )
        initial_guesses = dict(
            S_0=0.1,
            X_0=0.2,
            mue_max=0.3,
            K_S=0.4,
            t_lag=0.5,
            t_acc=0.6
        )
        parmap = murefi.ParameterMapping(map_df, bounds=bounds, guesses=initial_guesses)
        self.assertEqual(parmap.order, ('S_0', 'X_0', 'mue_max', 'K_S', 'Y_XS', 't_lag', 't_acc'))
        self.assertDictEqual(parmap.parameters, collections.OrderedDict([
            ('test1A', 'S_0'),
            ('test1B', 'X_0'),
            ('test2C', 'mue_max'),
            ('test2D', 'K_S')
            ]))
        self.assertEqual(parmap.ndim, 4)
        self.assertEqual(parmap.bounds, ((1,3), (3,4), (5,6), (7,8)))
        self.assertEqual(parmap.guesses, (0.1, 0.2, 0.3, 0.4))
        self.assertEqual(parmap.mapping, {
            'A01':('test1A', 'test1B', 3.0, 4.0, 5.0, 6.0, 7.0),
            'B02':(11.0, 'test1B', 'test2C', 'test2D', 15.0, 16.0, 17.0)
            })
        return
    
    def test_invalid_init(self):
        map_df = pandas.read_csv(pathlib.Path(dir_testfiles, 'ParTest.csv'), sep=';')
        map_df.set_index(map_df.columns[0])
        mapfail_df = pandas.read_csv(pathlib.Path(dir_testfiles, 'ParTestFail.csv'), sep=';')
        mapfail_df.set_index(mapfail_df.columns[0])
        bounds = dict(
            S_0=(1,2),
            X_0=(3,4),
            mue_max=(5,6),
            K_S=(7,8),
            t_lag=(9,10),
            t_acc=(11,12),
            test1A=(1,3)
        )
        initial_guesses = dict(
            S_0=0.1,
            X_0=0.2,
            mue_max=0.3,
            K_S=0.4,
            t_lag=0.5,
            t_acc=0.6
        )
        with self.assertRaises(TypeError):
            _ = murefi.ParameterMapping(map_df, bounds, initial_guesses)
        with self.assertRaises(ValueError):
            _ = murefi.ParameterMapping(mapfail_df, bounds=bounds, guesses=initial_guesses)

    def test_repmap(self):
        map_df = pandas.read_csv(pathlib.Path(dir_testfiles, 'ParTest.csv'), sep=';')
        map_df.set_index(map_df.columns[0])
        bounds = dict(
            S_0=(1,2),
            X_0=(3,4),
            mue_max=(5,6),
            K_S=(7,8),
            t_lag=(9,10),
            t_acc=(11,12),
            test1A=(1,3)
        )
        initial_guesses = dict(
            S_0=0.1,
            X_0=0.2,
            mue_max=0.3,
            K_S=0.4,
            t_lag=0.5,
            t_acc=0.6
        )
        parmap = murefi.ParameterMapping(map_df, bounds=bounds, guesses=initial_guesses)
        theta_fitted = [1,2,13,14]
        expected = {
            'A01': numpy.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
            'B02': numpy.array([11.0, 2.0, 13.0, 14.0, 15.0, 16.0, 17.0])
            }
        self.assertEqual(parmap.repmap(theta_fitted).keys(), expected.keys())
        for key in expected.keys():
            self.assertTrue(numpy.array_equal(parmap.repmap(theta_fitted)[key], expected[key]))
        return


class TestDataset(unittest.TestCase):
    def test_dataset(self):
        rep = murefi.Replicate('A01')
        rep['S_observed'] = murefi.Timeseries([0,2,3,  5,  8  ], [1,2,3,4,5], independent_key='S', timeseries_key='S_observed')
        ds = murefi.Dataset()
        ds['A01'] = rep
        self.assertTrue('A01' in ds)
        self.assertEqual(list(ds.keys()), ['A01'])
        return


class TestReplicate(unittest.TestCase):
    def test_x_any(self):
        rep = murefi.Replicate('A01')
        rep['S_observed'] = murefi.Timeseries([0,2,3,  5,  8  ], [1,2,3,4,5], independent_key='S', timeseries_key='S_observed')
        rep['X_observed'] = murefi.Timeseries([  2,3,4,  6,8,9], [1,2,3,4,5,6], independent_key='X', timeseries_key='X_observed')
        self.assertTrue(numpy.array_equal(rep.x_any, [0,2,3,4,5,6,8,9]))
        return

    def test_observation_booleans(self):
        rep = murefi.Replicate('A01')
        rep['S_observed'] = murefi.Timeseries([0,2,3,  5,  8  ], [1,2,3,4,5], independent_key='S', timeseries_key='S_observed')
        rep['X_observed'] = murefi.Timeseries([  2,3,4,  6,8,9], [1,2,3,4,5,6], independent_key='X', timeseries_key='X_observed')
        result = rep.get_observation_booleans(['S_observed', 'X_observed', 'P_observed'])
        self.assertTrue(numpy.array_equal(result['S_observed'], [True,True,True,False,True,False,True,False]))
        self.assertTrue(numpy.array_equal(result['X_observed'], [False,True,True,True,False,True,True,True]))
        self.assertTrue(numpy.array_equal(result['P_observed'], [False,False,False,False,False,False,False,False]))
        return

    def test_make_template(self):
        template = murefi.Replicate.make_template(0.5, 3.5, independent_keys=['A', 'B', 'C'], N=20)
        self.assertIn('A', template)
        self.assertIn('B', template)
        self.assertIn('C', template)
        self.assertTrue(template['A'].x[0] == 0.5)
        self.assertTrue(template['A'].x[-1] == 3.5)
        self.assertTrue(len(template['A'].x) == 20)
        return


if __name__ == '__main__':
    unittest.main(exit=False)