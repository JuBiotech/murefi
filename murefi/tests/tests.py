import collections
import unittest
import numpy
import pandas
import pathlib
import scipy.stats as stats

import calibr8
import murefi


dir_testfiles = pathlib.Path(pathlib.Path(__file__).absolute().parent, 'testfiles')


def _mini_model():
    class MiniModel(murefi.BaseODEModel):
        def dydt(self, y, t, theta):
            A, B, C = y
            alpha, beta = theta
            dCdt = alpha*A + beta*B**2
            dAdt = -dCdt
            dBdt = -2*dCdt
            return [dAdt, dBdt, dCdt]
    return MiniModel(independent_keys=['A', 'B', 'C'])


def _mini_error_model(independent:str, dependent:str):
    class EM(calibr8.ErrorModel):
        def loglikelihood(self, *, y_obs,  y_hat, theta=None):
            # assume Normal with sd=1
            return numpy.sum(stats.norm.logpdf(y_obs-y_hat))
    return EM(independent_key=independent, dependent_key=dependent)


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
        rep['S_observed'] = murefi.Timeseries([0,2,3,  5,  8  ], [1,2,3,4,5], independent_key='S', dependent_key='S_observed')
        ds = murefi.Dataset()
        ds['A01'] = rep
        self.assertTrue('A01' in ds)
        self.assertEqual(list(ds.keys()), ['A01'])
        return


class TestReplicate(unittest.TestCase):
    def test_x_any(self):
        rep = murefi.Replicate('A01')
        rep['S_observed'] = murefi.Timeseries([0,2,3,  5,  8  ], [1,2,3,4,5], independent_key='S', dependent_key='S_observed')
        rep['X_observed'] = murefi.Timeseries([  2,3,4,  6,8,9], [1,2,3,4,5,6], independent_key='X', dependent_key='X_observed')
        self.assertTrue(numpy.array_equal(rep.x_any, [0,2,3,4,5,6,8,9]))
        return

    def test_observation_booleans(self):
        rep = murefi.Replicate('A01')
        rep['S_observed'] = murefi.Timeseries([0,2,3,  5,  8  ], [1,2,3,4,5], independent_key='S', dependent_key='S_observed')
        rep['X_observed'] = murefi.Timeseries([  2,3,4,  6,8,9], [1,2,3,4,5,6], independent_key='X', dependent_key='X_observed')
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


class TestBaseODEModel(unittest.TestCase):
    def test_attributes(self):
        model = _mini_model()

        self.assertIsInstance(model, murefi.BaseODEModel)
        self.assertEqual(model.n_y, 3)
        self.assertSequenceEqual(model.independent_keys, ['A', 'B', 'C'])
        return

    def test_solver(self):
        theta = [0.23, 0.85]
        y0 = [2., 2., 0.]
        x = numpy.linspace(0, 1, 5)
        model = _mini_model()      

        y_hat = model.solver(y0, x, theta)
        self.assertIsInstance(y_hat, dict)
        self.assertIn('A', y_hat)
        self.assertIn('B', y_hat)
        self.assertIn('C', y_hat)
        self.assertTrue(numpy.allclose(y_hat['A'], [2.0, 1.4819299, 1.28322046, 1.16995677, 1.09060199]))
        self.assertTrue(numpy.allclose(y_hat['B'], [2.0, 0.9638598, 0.56644092, 0.33991354, 0.18120399]))
        self.assertTrue(numpy.allclose(y_hat['C'], [0.0, 0.5180701, 0.71677954, 0.83004323, 0.90939801]))
        return

    def test_solver_no_zero_time(self):
        theta = [0.23, 0.85]
        y0 = [2., 2., 0.]
        x = numpy.linspace(0, 1, 5)[1:]
        model = _mini_model()      

        y_hat = model.solver(y0, x, theta)
        self.assertTrue(numpy.allclose(y_hat['A'], [1.4819299, 1.28322046, 1.16995677, 1.09060199]))
        self.assertTrue(numpy.allclose(y_hat['B'], [0.9638598, 0.56644092, 0.33991354, 0.18120399]))
        self.assertTrue(numpy.allclose(y_hat['C'], [0.5180701, 0.71677954, 0.83004323, 0.90939801]))
        return

    def test_predict_replicate(self):
        theta = [0.23, 0.85]
        y0 = [2., 2., 0.]
        x = numpy.linspace(0, 1, 5)
        model = _mini_model()
        
        template = murefi.Replicate('TestRep')
        # one observation of A, two observations of C
        template['A'] = murefi.Timeseries(x[:3], [0]*3, independent_key='A', dependent_key='A')
        template['C1'] = murefi.Timeseries(x[2:4], [0]*2, independent_key='C', dependent_key='C1')
        template['C2'] = murefi.Timeseries(x[1:4], [0]*3, independent_key='C', dependent_key='C2')
        prediction = model.predict_replicate(y0 + theta, template)

        self.assertIsInstance(prediction, murefi.Replicate)
        self.assertEqual(prediction.iid, 'TestRep')
        self.assertIn('A', prediction)
        self.assertFalse('B' in prediction)
        self.assertIn('C1', prediction)
        self.assertIn('C2', prediction)

        self.assertTrue(numpy.allclose(prediction['A'].y, [2.0, 1.4819299, 1.28322046]))
        self.assertTrue(numpy.allclose(prediction['C1'].y, [0.71677954, 0.83004323]))
        self.assertTrue(numpy.allclose(prediction['C2'].y, [0.5180701, 0.71677954, 0.83004323]))
        return

    def test_predict_dataset(self):
        model = _mini_model()

        # create a template dataset
        dataset = murefi.Dataset()
        dataset['R1'] = murefi.Replicate.make_template(0, 1, 'AB', N=60, iid='R1')
        dataset['R2'] = murefi.Replicate.make_template(0.2, 1, 'BC', N=20, iid='R2')

        # create a parameter mapping that uses replicate-wise alpha parameters (6 dims)
        mapping = pandas.DataFrame(columns=['id,A0,B0,C0,alpha,beta'.split(',')]).set_index('id')
        mapping.loc['R1'] = 'A0,B0,C0,alpha_1,beta'.split(',')
        mapping.loc['R2'] = 'A0,B0,C0,alpha_2,beta'.split(',')
        mapping = mapping.reset_index()
        pm = murefi.ParameterMapping(mapping, bounds=dict(), guesses=dict())
        self.assertEqual(pm.ndim, 6)

        
        # set a global parameter vector with alpha_1=0.22, alpha_2=0.24
        self.assertSequenceEqual(tuple(pm.parameters.keys()), 'A0,B0,C0,alpha_1,alpha_2,beta'.split(','))
        theta = [2., 2., 0.] + [0.22, 0.24, 0.85]
        prediction = model.predict_dataset(template=dataset, par_map=pm, theta_fit=theta)

        self.assertIsInstance(prediction, murefi.Dataset)
        self.assertIn('R1', prediction)
        self.assertIn('R2', prediction)
        self.assertTrue('A' in prediction['R1'])
        self.assertTrue('B' in prediction['R1'])
        self.assertFalse('C' in prediction['R1'])
        self.assertFalse('A' in prediction['R2'])
        self.assertTrue('B' in prediction['R2'])
        self.assertTrue('C' in prediction['R2'])
        self.assertEqual(len(prediction['R1'].x_any), 60)
        self.assertEqual(len(prediction['R2'].x_any), 20)
        return


class TestObjectives(unittest.TestCase):
    def test_for_dataset(self):
        model = _mini_model()

        # create a template dataset (uses numpy.empty to create y-values!)
        dataset = murefi.Dataset()
        dataset['R1'] = murefi.Replicate.make_template(0, 1, 'AB', iid='R1')
        dataset['R2'] = murefi.Replicate.make_template(0.2, 1, 'BC', N=20, iid='R2')
        # set all y-values to 0.5 to avoid NaNs in the loglikelihood
        for _, rep in dataset.items():
            for _, ts in rep.items():
                ts.y = numpy.repeat(0.5, len(ts))

        # create a parameter mapping that uses replicate-wise alpha parameters (6 dims)
        mapping = pandas.DataFrame(columns=['id,A0,B0,C0,alpha,beta'.split(',')]).set_index('id')
        mapping.loc['R1'] = 'A0,B0,C0,alpha_1,beta'.split(',')
        mapping.loc['R2'] = 'A0,B0,C0,alpha_2,beta'.split(',')
        mapping = mapping.reset_index()
        pm = murefi.ParameterMapping(mapping, bounds=dict(), guesses=dict())
        self.assertEqual(pm.ndim, 6)

        obj = murefi.objectives.for_dataset(dataset, model, pm, error_models=[
            _mini_error_model('A', 'A'),
            _mini_error_model('B', 'B'),
            _mini_error_model('C', 'C'),
        ])
        
        self.assertTrue(callable(obj))
        theta = [2., 2., 0.] + [0.22, 0.24, 0.85]
        L = obj(theta)
        self.assertIsInstance(L, float)
        self.assertNotEqual(L, float('nan'))
        return


if __name__ == '__main__':
    unittest.main(exit=False)