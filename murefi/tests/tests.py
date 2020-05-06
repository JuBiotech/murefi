import collections
import h5py
import unittest
import numpy
import pandas
import pathlib
import scipy.integrate
import scipy.stats as stats
import tempfile

import calibr8
import murefi


try:
    import pymc3
    import theano
    import theano.tensor as tt
    HAVE_PYMC3 = True
except ModuleNotFoundError:
    HAVE_PYMC3 = False


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
    return MiniModel(theta_names=['A0', 'B0', 'C0', 'alpha', 'beta'], independent_keys=['A', 'B', 'C'])


def _mini_error_model(independent:str, dependent:str):
    class EM(calibr8.BasePolynomialModelT):
        def __init__(self):
            super().__init__(independent_key=independent, dependent_key=dependent, mu_degree=1, scale_degree=0)
    em = EM()
    em.theta_fitted = [0, 1, 1, 100]
    assert len(em.theta_fitted) == len(em.theta_names)
    return em


class ParameterMapTest(unittest.TestCase):
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

    def test_init(self):
        map_df = pandas.read_csv(pathlib.Path(dir_testfiles, 'ParTest.csv'), sep=';')
        map_df.set_index(map_df.columns[0])
        parmap = murefi.ParameterMapping(map_df, bounds=self.bounds, guesses=self.initial_guesses)
        self.assertEqual(parmap.order, ('S_0', 'X_0', 'mue_max', 'K_S', 'Y_XS', 't_lag', 't_acc'))
        self.assertDictEqual(parmap.parameters, collections.OrderedDict([
            ('test1A', 'S_0'),
            ('test1B', 'X_0'),
            ('test2C', 'mue_max'),
            ('test2D', 'K_S')
            ]))
        self.assertSequenceEqual(parmap.theta_names, tuple(parmap.parameters.keys()))
        self.assertEqual(parmap.ndim, 4)
        self.assertEqual(parmap.bounds, ((1,3), (3,4), (5,6), (7,8)))
        self.assertEqual(parmap.guesses, (0.1, 0.2, 0.3, 0.4))
        self.assertEqual(parmap.mapping, {
            'A01':('test1A', 'test1B', 3.0, 4.0, 5.0, 6.0, 7.0),
            'B02':(11.0, 'test1B', 'test2C', 'test2D', 15.0, 16.0, 17.0)
            })
        parmap = murefi.ParameterMapping(map_df, bounds=None, guesses=None)
        self.assertEqual(parmap.order, ('S_0', 'X_0', 'mue_max', 'K_S', 'Y_XS', 't_lag', 't_acc'))
        self.assertDictEqual(parmap.parameters, collections.OrderedDict([
            ('test1A', 'S_0'),
            ('test1B', 'X_0'),
            ('test2C', 'mue_max'),
            ('test2D', 'K_S')
            ]))
        self.assertEqual(parmap.ndim, 4)
        self.assertEqual(parmap.bounds,((None, None), (None, None), (None, None), (None, None)))
        self.assertEqual(parmap.guesses, (None, None, None, None))
        self.assertEqual(parmap.mapping, {
            'A01':('test1A', 'test1B', 3.0, 4.0, 5.0, 6.0, 7.0),
            'B02':(11.0, 'test1B', 'test2C', 'test2D', 15.0, 16.0, 17.0)
            })
        pass

    def test_invalid_init(self):
        map_df = pandas.read_csv(pathlib.Path(dir_testfiles, 'ParTest.csv'), sep=';')
        map_df.set_index(map_df.columns[0])
        mapfail_df = pandas.read_csv(pathlib.Path(dir_testfiles, 'ParTestFail.csv'), sep=';')
        mapfail_df.set_index(mapfail_df.columns[0])
        with self.assertRaises(TypeError):
            murefi.ParameterMapping(map_df, self.bounds, self.initial_guesses)
        with self.assertRaises(ValueError):
            murefi.ParameterMapping(mapfail_df, bounds=self.bounds, guesses=self.initial_guesses)
        pass

    def test_repmap_array(self):
        map_df = pandas.read_csv(pathlib.Path(dir_testfiles, 'ParTest.csv'), sep=';')
        map_df.set_index(map_df.columns[0])
        parmap = murefi.ParameterMapping(map_df, bounds=self.bounds, guesses=self.initial_guesses)
        theta_fitted = [1,2,13,14]
        expected = {
            'A01': numpy.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
            'B02': numpy.array([11.0, 2.0, 13.0, 14.0, 15.0, 16.0, 17.0])
        }
        self.assertEqual(parmap.repmap(theta_fitted).keys(), expected.keys())
        for key in expected.keys():
            self.assertTrue(numpy.array_equal(parmap.repmap(theta_fitted)[key], expected[key]))
        pass

    def test_repmap_dict(self):
        map_df = pandas.read_csv(pathlib.Path(dir_testfiles, 'ParTest.csv'), sep=';')
        map_df.set_index(map_df.columns[0])
        parmap = murefi.ParameterMapping(map_df, bounds=self.bounds, guesses=self.initial_guesses)
        theta_fitted = dict(test1A=1.0, test1B=2.0, test2C=13, test2D=14)
        expected = {
            'A01': numpy.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
            'B02': numpy.array([11.0, 2.0, 13.0, 14.0, 15.0, 16.0, 17.0])
        }
        self.assertEqual(parmap.repmap(theta_fitted).keys(), expected.keys())
        for key in expected.keys():
            self.assertTrue(numpy.array_equal(parmap.repmap(theta_fitted)[key], expected[key]))
        pass

    def test_repmap_2d_array(self):
        map_df = pandas.read_csv(pathlib.Path(dir_testfiles, 'ParTest.csv'), sep=';')
        map_df.set_index(map_df.columns[0])
        parmap = murefi.ParameterMapping(map_df, bounds=self.bounds, guesses=self.initial_guesses)
        P = len(parmap.parameters)

        # test with (P, S) array
        S = 11
        theta = numpy.random.uniform(size=(P, S))
        mapped = parmap.repmap(theta)
        for rid, parameters in mapped.items():
            assert isinstance(parameters, tuple)
            assert len(parameters) == 7
            for p, pval in enumerate(parameters):
                assert numpy.shape(pval) in {(), (S,)}
        pass

    def test_repmap_2d_mixed_dict(self):
        map_df = pandas.read_csv(pathlib.Path(dir_testfiles, 'ParTest.csv'), sep=';')
        map_df.set_index(map_df.columns[0])
        parmap = murefi.ParameterMapping(map_df, bounds=self.bounds, guesses=self.initial_guesses)
        P = len(parmap.parameters)

        # test with dictionary of mixed (S,) and scalars
        S = 23
        theta = dict(
            test1A=numpy.arange(S) + 1.0,
            test1B=numpy.arange(S) + 2.0,
            test2C=13,
            test2D=numpy.arange(S) + 14,
        )
        mapped = parmap.repmap(theta)
        mapped_shapes = {
            rid : [numpy.shape(pval) for pval in parameters]
            for rid, parameters in mapped.items()
        }
        expected = {
            'A01': numpy.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
            'B02': numpy.array([11.0, 2.0, 13.0, 14.0, 15.0, 16.0, 17.0])
        }
        for rid, parmeters in mapped.items():
            parameters = mapped[rid]
            assert isinstance(parameters, tuple)
            assert len(parameters) == 7
            for p, pval in enumerate(parameters):
                pshape = numpy.shape(pval)
                if pshape == ():
                    assert pval == expected[rid][p]
                else:
                    assert pshape == (S,)
                    numpy.testing.assert_array_equal(
                        pval,
                        numpy.arange(S) + expected[rid][p]
                    )
        pass


class TestDataset(unittest.TestCase):
    def test_dataset(self):
        rep = murefi.Replicate('A01')
        rep['S_observed'] = murefi.Timeseries([0,2,3,  5,  8  ], [1,2,3,4,5], independent_key='S', dependent_key='S_observed')
        ds = murefi.Dataset()
        ds['A01'] = rep
        self.assertTrue('A01' in ds)
        self.assertEqual(list(ds.keys()), ['A01'])
        return

    def test_make_template(self):
        template = murefi.Dataset.make_template(0.5, 3.5, independent_keys='ABC', rids='R1,R2,R3,R4'.split(','), N=20)
        self.assertIsInstance(template, murefi.Dataset)
        self.assertIn('R1', template)
        self.assertIn('R2', template)
        self.assertIn('R3', template)
        self.assertTrue(template['R1']['A'].t[0] == 0.5)
        self.assertTrue(template['R2']['B'].t[-1] == 3.5)
        self.assertTrue(len(template['R3']['C'].t) == 20)
        return

    def test_make_template_like(self):
        ds = murefi.Dataset()
        ds['A01'] = murefi.Replicate('A01')
        ds['A01']['A_obs'] = murefi.Timeseries([0,2,3], [0.2,0.4,0.1], independent_key='A', dependent_key='A_obs')
        ds['B01'] = murefi.Replicate('B01')
        ds['B01']['B_obs'] = murefi.Timeseries([2,3,5], [0.2,0.4,0.1], independent_key='B', dependent_key='B_obs')

        template = murefi.Dataset.make_template_like(ds, independent_keys=['A', 'B', 'C'], N=20)
        assert isinstance(template, murefi.Dataset)
        assert 'A01' in template
        assert 'B01' in template
        for ikey in 'ABC':
            assert ikey in template['A01']
            assert ikey in template['B01']
            numpy.testing.assert_array_equal(template['A01'][ikey].t, numpy.linspace(0, 3, 20))
            numpy.testing.assert_array_equal(template['B01'][ikey].t, numpy.linspace(2, 5, 20))
        return


class TestReplicate(unittest.TestCase):
    def test_t_any(self):
        rep = murefi.Replicate('A01')
        rep['S_observed'] = murefi.Timeseries([0,2,3,  5,  8  ], [1,2,3,4,5], independent_key='S', dependent_key='S_observed')
        rep['X_observed'] = murefi.Timeseries([  2,3,4,  6,8,9], [1,2,3,4,5,6], independent_key='X', dependent_key='X_observed')
        self.assertTrue(numpy.array_equal(rep.t_any, [0,2,3,4,5,6,8,9]))
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

    def test_observation_indices(self):
        rep = murefi.Replicate('A01')
        rep['S_observed'] = murefi.Timeseries([0,2,3,  5,  8  ], [1,2,3,4,5], independent_key='S', dependent_key='S_observed')
        rep['X_observed'] = murefi.Timeseries([  2,3,4,  6,8,9], [1,2,3,4,5,6], independent_key='X', dependent_key='X_observed')
        result = rep.get_observation_indices(['S_observed', 'X_observed', 'P_observed'])
        self.assertTrue(numpy.array_equal(result['S_observed'], [0,1,2,4,6]))
        self.assertTrue(numpy.array_equal(result['X_observed'], [1,2,3,5,6,7]))
        self.assertTrue(numpy.array_equal(result['P_observed'], []))
        return

    def test_make_template(self):
        template = murefi.Replicate.make_template(0.5, 3.5, independent_keys=['A', 'B', 'C'], N=20)
        self.assertIn('A', template)
        self.assertIn('B', template)
        self.assertIn('C', template)
        self.assertTrue(template['A'].t[0] == 0.5)
        self.assertTrue(template['A'].t[-1] == 3.5)
        self.assertTrue(len(template['A'].t) == 20)
        return


class TestTimeseries(unittest.TestCase):
    def test_t_monotonic(self):
        with self.assertRaises(ValueError):
            murefi.Timeseries([1,2,0,4], [1,2,3,4], independent_key='A', dependent_key='A_obs')
        pass

    def test_y_1d(self):
        N = 45
        t = numpy.linspace(0, 60, N)
        y = numpy.random.normal(t)
        assert t.shape == (N,)
        assert y.shape == (N,)

        ts = murefi.Timeseries(t, y, independent_key='T', dependent_key='T_obs')
        numpy.testing.assert_array_equal(ts.t, t)
        numpy.testing.assert_array_equal(ts.y, y)
        assert ts.independent_key == 'T'
        assert ts.dependent_key == 'T_obs'
        assert not ts.is_distribution

        with self.assertRaises(murefi.ShapeError):
            murefi.Timeseries([1,2,3], [1,2,3,4], independent_key='T', dependent_key='T_obs')
        pass

    def test_y_2d(self):
        N = 45
        S = 500
        t = numpy.linspace(0, 60, N)
        y = numpy.random.normal(t, size=(S, N))
        assert t.shape == (N,)
        assert y.shape == (S, N)

        ts = murefi.Timeseries(t, y, independent_key='T', dependent_key='T_obs')
        assert ts.is_distribution
        numpy.testing.assert_array_equal(ts.t, t)
        numpy.testing.assert_array_equal(ts.y, y)
        assert ts.independent_key == 'T'
        assert ts.dependent_key == 'T_obs'

        ts = murefi.Timeseries(
            [1,2,3],
            [[4,5,6]]*5,
            independent_key='T',
            dependent_key='T_obs'
        )
        assert ts.is_distribution
        numpy.testing.assert_array_equal(ts.t, [1,2,3])
        numpy.testing.assert_array_equal(ts.y.shape, (5, 3))
        numpy.testing.assert_array_equal(ts.y[1,:], [4,5,6])

        with self.assertRaises(murefi.ShapeError):
            murefi.Timeseries(
                [1,2,3],
                [[4,5,6,7]]*5,
                independent_key='T',
                dependent_key='T_obs'
            )
        pass

    def test_to_from_dataset_1d(self):
        ts = murefi.Timeseries(
            t=numpy.linspace(0, 10, 50),
            y=numpy.random.normal(size=(50,)),
            independent_key='A', dependent_key='A_obs'
        )
        assert ts.is_distribution == False

        # test with an in-memory HDF5 file
        with h5py.File('testfile.h5', 'w', driver='core') as file:
            group = file.create_group('A02')
            ts._to_dataset(group)
            assert 'A_obs' in group
            tsds = group['A_obs']
            assert tsds.attrs['independent_key'] == 'A'
            assert tsds.attrs['dependent_key'] == 'A_obs'
            assert tsds.shape == (1 + 1, 50)

            # load from the dataset
            ts_loaded = murefi.Timeseries._from_dataset(tsds)
            assert ts.is_distribution == False
            assert ts_loaded.independent_key == ts.independent_key
            assert ts_loaded.dependent_key == ts.dependent_key
            numpy.testing.assert_array_equal(ts_loaded.t, ts.t)
            numpy.testing.assert_array_equal(ts_loaded.y, ts.y)
        pass

    def test_to_from_dataset_2d(self):
        ts = murefi.Timeseries(
            t=numpy.linspace(0, 10, 50),
            y=numpy.random.normal(size=(1000,50)),
            independent_key='A', dependent_key='A_obs'
        )
        assert ts.is_distribution == True

        # test with an in-memory HDF5 file
        with h5py.File('testfile.h5', 'w', driver='core') as file:
            group = file.create_group('A02')
            ts._to_dataset(group)
            assert 'A_obs' in group
            tsds = group['A_obs']
            assert tsds.attrs['independent_key'] == 'A'
            assert tsds.attrs['dependent_key'] == 'A_obs'
            assert tsds.shape == (1 + 1000, 50)

            # load from the dataset
            ts_loaded = murefi.Timeseries._from_dataset(tsds)
            assert ts.is_distribution == True
            assert ts_loaded.independent_key == ts.independent_key
            assert ts_loaded.dependent_key == ts.dependent_key
            numpy.testing.assert_array_equal(ts_loaded.t, ts.t)
            numpy.testing.assert_array_equal(ts_loaded.y, ts.y)
        pass


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
        T = 5
        t = numpy.linspace(0, 1, T)
        model = _mini_model()      

        y_hat = model.solver(y0, t, theta)
        self.assertIsInstance(y_hat, dict)
        self.assertIn('A', y_hat)
        self.assertIn('B', y_hat)
        self.assertIn('C', y_hat)
        for ikey in model.independent_keys:
            assert isinstance(y_hat[ikey], numpy.ndarray)
            assert y_hat[ikey].shape == (T,)
        self.assertTrue(numpy.allclose(y_hat['A'], [2.0, 1.4819299, 1.28322046, 1.16995677, 1.09060199]))
        self.assertTrue(numpy.allclose(y_hat['B'], [2.0, 0.9638598, 0.56644092, 0.33991354, 0.18120399]))
        self.assertTrue(numpy.allclose(y_hat['C'], [0.0, 0.5180701, 0.71677954, 0.83004323, 0.90939801]))
        return

    def test_solver_no_zero_time(self):
        theta = [0.23, 0.85]
        y0 = [2., 2., 0.]
        t = numpy.linspace(0, 1, 5)[1:]
        model = _mini_model()      

        y_hat = model.solver(y0, t, theta)
        self.assertTrue(numpy.allclose(y_hat['A'], [1.4819299, 1.28322046, 1.16995677, 1.09060199]))
        self.assertTrue(numpy.allclose(y_hat['B'], [0.9638598, 0.56644092, 0.33991354, 0.18120399]))
        self.assertTrue(numpy.allclose(y_hat['C'], [0.5180701, 0.71677954, 0.83004323, 0.90939801]))
        return

    def test_solver_vectorized(self):
        S = 300
        T = 25

        model = _mini_model()      
        y0 = numpy.array([[2., 2., 0.]] * S).T
        theta = numpy.array([[0.23, 0.85]] * S).T
        t = numpy.linspace(0.2, 2, T)
        assert y0.shape == (3, S)
        assert theta.shape == (2, S)
        assert t.shape == (T,)

        y_hat = model.solver_vectorized(y0, t, theta)
        assert isinstance(y_hat, dict)
        for ikey in model.independent_keys:
            assert ikey in y_hat
            assert isinstance(y_hat[ikey], numpy.ndarray)
            assert y_hat[ikey].shape == (T, S)
        pass

    def test_predict_replicate(self):
        theta = [0.23, 0.85]
        y0 = [2., 2., 0.]
        t = numpy.linspace(0, 1, 5)
        model = _mini_model()
        
        template = murefi.Replicate('TestRep')
        # one observation of A, two observations of C
        template['A'] = murefi.Timeseries(t[:3], [0]*3, independent_key='A', dependent_key='A')
        template['C1'] = murefi.Timeseries(t[2:4], [0]*2, independent_key='C', dependent_key='C1')
        template['C2'] = murefi.Timeseries(t[1:4], [0]*3, independent_key='C', dependent_key='C2')
        prediction = model.predict_replicate(y0 + theta, template)

        self.assertIsInstance(prediction, murefi.Replicate)
        self.assertEqual(prediction.rid, 'TestRep')
        self.assertIn('A', prediction)
        self.assertFalse('B' in prediction)
        self.assertIn('C1', prediction)
        self.assertIn('C2', prediction)

        self.assertTrue(numpy.allclose(prediction['A'].y, [2.0, 1.4819299, 1.28322046]))
        self.assertTrue(numpy.allclose(prediction['C1'].y, [0.71677954, 0.83004323]))
        self.assertTrue(numpy.allclose(prediction['C2'].y, [0.5180701, 0.71677954, 0.83004323]))
        pass

    def test_predict_replicate_distribution(self):
        t = numpy.linspace(0, 1, 5)
        model = _mini_model()
        template = murefi.Replicate('TestRep')
        # one observation of A, two observations of C
        template['A'] = murefi.Timeseries(t[:3], [0]*3, independent_key='A', dependent_key='A')
        template['C1'] = murefi.Timeseries(t[2:4], [0]*2, independent_key='C', dependent_key='C1')
        template['C2'] = murefi.Timeseries(t[1:4], [0]*3, independent_key='C', dependent_key='C2')

        P = len(model.theta_names)
        S = 300
        pred = model.predict_replicate(template=template, parameters=numpy.ones((P, S)))
        for dkey, ts_pred in pred.items():
            ts_template = template[dkey]
            numpy.testing.assert_array_equal(ts_pred.t, ts_template.t)
            assert numpy.shape(ts_pred.y) == (S, len(ts_template))
        pass

    def test_predict_replicate_inputchecks(self):
        t = numpy.linspace(0, 1, 5)
        model = _mini_model()
        template = murefi.Replicate('TestRep')
        # one observation of A, two observations of C
        template['A'] = murefi.Timeseries(t[:3], [0]*3, independent_key='A', dependent_key='A')
        template['C1'] = murefi.Timeseries(t[2:4], [0]*2, independent_key='C', dependent_key='C1')
        template['C2'] = murefi.Timeseries(t[1:4], [0]*3, independent_key='C', dependent_key='C2')

        P = len(model.theta_names)

        # wrong parameter shapes
        with self.assertRaises(murefi.ShapeError):
            # wrong number of parameters
            model.predict_replicate(template=template, parameters=numpy.ones((P+2,)))
        with self.assertRaises(murefi.ShapeError):
            # 3D parameters
            model.predict_replicate(template=template, parameters=numpy.ones((P, 300, 1)))
        pass

    def test_predict_dataset(self):
        model = _mini_model()
        
        # create a template dataset
        dataset = murefi.Dataset()
        dataset['R1'] = murefi.Replicate.make_template(0, 1, 'AB', N=60, rid='R1')
        dataset['R2'] = murefi.Replicate.make_template(0.2, 1, 'BC', N=20, rid='R2')

        # create a parameter mapping that uses replicate-wise alpha parameters (6 dims)
        mapping = pandas.DataFrame(columns='id,A0,B0,C0,alpha,beta'.split(',')).set_index('id')
        mapping.loc['R1'] = 'A0,B0,C0,alpha_1,beta'.split(',')
        mapping.loc['R2'] = 'A0,B0,C0,alpha_2,beta'.split(',')
        mapping = mapping.reset_index()
        pm = murefi.ParameterMapping(mapping, bounds=dict(), guesses=dict())
        self.assertEqual(pm.ndim, 6)

        
        # set a global parameter vector with alpha_1=0.22, alpha_2=0.24
        self.assertSequenceEqual(tuple(pm.parameters.keys()), 'A0,B0,C0,alpha_1,alpha_2,beta'.split(','))
        theta = [2., 2., 0.] + [0.22, 0.24, 0.85]
        prediction = model.predict_dataset(template=dataset, theta_mapping=pm, parameters=theta)

        self.assertIsInstance(prediction, murefi.Dataset)
        self.assertIn('R1', prediction)
        self.assertIn('R2', prediction)
        self.assertTrue('A' in prediction['R1'])
        self.assertTrue('B' in prediction['R1'])
        self.assertFalse('C' in prediction['R1'])
        self.assertFalse('A' in prediction['R2'])
        self.assertTrue('B' in prediction['R2'])
        self.assertTrue('C' in prediction['R2'])
        self.assertEqual(len(prediction['R1'].t_any), 60)
        self.assertEqual(len(prediction['R2'].t_any), 20)
        return

    def test_predict_dataset_distribution(self):
        model = _mini_model()
        
        # create a template dataset
        dataset = murefi.Dataset()
        dataset['R1'] = murefi.Replicate.make_template(0, 1, 'AB', N=60, rid='R1')
        dataset['R2'] = murefi.Replicate.make_template(0.2, 1, 'BC', N=20, rid='R2')

        # create a parameter mapping that uses replicate-wise alpha parameters (6 dims)
        mapping = pandas.DataFrame(columns='id,A0,B0,C0,alpha,beta'.split(',')).set_index('id')
        mapping.loc['R1'] = 'A0,B0,C0,alpha_1,beta'.split(',')
        mapping.loc['R2'] = 'A0,B0,C0,alpha_2,beta'.split(',')
        mapping = mapping.reset_index()
        pm = murefi.ParameterMapping(mapping, bounds=dict(), guesses=dict())
        self.assertEqual(pm.ndim, 6)

        # set a global parameter vector with alpha_1=0.22, alpha_2=0.24
        self.assertSequenceEqual(tuple(pm.parameters.keys()), 'A0,B0,C0,alpha_1,alpha_2,beta'.split(','))
        theta = [2., 2., 0.] + [0.22, 0.24, 0.85]
        # randomize the parameter vector into a matrix
        P = len(theta)
        S = 40
        theta = numpy.array(theta)[:,numpy.newaxis] + numpy.random.uniform(0, 0.1, size=(P, S))
        assert theta.shape == (P, S)

        prediction = model.predict_dataset(template=dataset, theta_mapping=pm, parameters=theta)
        self.assertIsInstance(prediction, murefi.Dataset)
        self.assertIn('R1', prediction)
        self.assertIn('R2', prediction)
        self.assertTrue('A' in prediction['R1'])
        self.assertTrue('B' in prediction['R1'])
        self.assertFalse('C' in prediction['R1'])
        self.assertFalse('A' in prediction['R2'])
        self.assertTrue('B' in prediction['R2'])
        self.assertTrue('C' in prediction['R2'])
        self.assertEqual(len(prediction['R1'].t_any), 60)
        self.assertEqual(len(prediction['R2'].t_any), 20)
        for rid, rep in prediction.items():
            for dkey, ts in rep.items():
                assert numpy.shape(ts.y) == (S, len(dataset[rid][dkey]))
        pass

    def test_attributes(self):
        monod = murefi.MonodModel()
        self.assertIsInstance(monod, murefi.BaseODEModel)
        self.assertIsInstance(monod, murefi.MonodModel)
        self.assertEqual(monod.n_y, 2)
        self.assertSequenceEqual(monod.independent_keys, ['S', 'X'])
    
    def test_dydt(self):
        monod = murefi.MonodModel()
        y = numpy.array([0.1, 10])
        t = 0
        theta = numpy.array([0.5, 0.1, 0.5])
        true = monod.dydt(y, t, theta)
        expected = [-0.5, 0.25]


class TestObjectives(unittest.TestCase):
    def test_for_dataset(self):
        model = _mini_model()

        # create a template dataset (uses numpy.empty to create y-values!)
        dataset = murefi.Dataset()
        dataset['R1'] = murefi.Replicate.make_template(0, 1, 'AB', rid='R1')
        dataset['R2'] = murefi.Replicate.make_template(0.2, 1, 'BC', N=20, rid='R2')
        # set all y-values to 0.5 to avoid NaNs in the loglikelihood
        for _, rep in dataset.items():
            for _, ts in rep.items():
                ts.y = numpy.repeat(0.5, len(ts))

        # create a parameter mapping that uses replicate-wise alpha parameters (6 dims)
        mapping = pandas.DataFrame(columns='id,A0,B0,C0,alpha,beta'.split(',')).set_index('id')
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


class TestSymbolicComputation(unittest.TestCase):
    @unittest.skipUnless(HAVE_PYMC3, 'requires PyMC3')
    def test_timeseries_support(self):
        t = numpy.linspace(0, 10, 10)
        with theano.configparser.change_flags(compute_test_value='off'):
            y = tt.scalar('TestY', dtype=theano.config.floatX)
            assert isinstance(y, tt.TensorVariable)
            ts = murefi.Timeseries(t, y, independent_key='Test', dependent_key='Test')
        return

    @unittest.skipUnless(HAVE_PYMC3, 'requires PyMC3')
    def test_symbolic_parameter_mapping(self):
        map_df = pandas.read_csv(pathlib.Path(dir_testfiles, 'ParTest.csv'), sep=';')
        map_df.set_index(map_df.columns[0])
        parmap = murefi.ParameterMapping(map_df, bounds=dict(
                S_0=(1,2),
                X_0=(3,4),
                mue_max=(5,6),
                K_S=(7,8),
                t_lag=(9,10),
                t_acc=(11,12)
            ), guesses=dict(
                S_0=0.1,
                X_0=0.2,
                mue_max=0.3,
                K_S=0.4,
                t_lag=0.5,
                t_acc=0.6
            )
        )

        # create a theta that is a mix of constant and symbolic variables
        with theano.configparser.change_flags(compute_test_value='off'):
            theta_fitted = [1, tt.scalar('mu_max', dtype=theano.config.floatX), 13, tt.scalar('t_acc', dtype=theano.config.floatX)]

            # map it to the two replicates
            expected = {
                'A01': numpy.array([1.0, None, 3.0, 4.0, 5.0, 6.0, 7.0]),
                'B02': numpy.array([11.0, None, 13.0, None, 15.0, 16.0, 17.0])
            }
            mapped = parmap.repmap(theta_fitted)
            self.assertEqual(mapped.keys(), expected.keys())
            for rid in expected.keys():
                for exp, act in zip(expected[rid], mapped[rid]):
                    if exp is not None:
                        self.assertTrue(exp, act)
                    else:
                        assert isinstance(act, tt.TensorVariable)
        return

    @unittest.skipUnless(HAVE_PYMC3, 'requires PyMC3')
    def test_symbolic_predict_replicate(self):
        with theano.configparser.change_flags(compute_test_value='off'):
            inputs = [
                tt.scalar('beta', dtype=theano.config.floatX),
                tt.scalar('A', dtype=theano.config.floatX)
            ]
            theta = [0.23, inputs[0]]
            y0 = [inputs[1], 2., 0.]
            t = numpy.linspace(0, 1, 5)
            model = _mini_model()
            
            template = murefi.Replicate('TestRep')
            # one observation of A, two observations of C
            template['A'] = murefi.Timeseries(t[:3], [0]*3, independent_key='A', dependent_key='A')
            template['C1'] = murefi.Timeseries(t[2:4], [0]*2, independent_key='C', dependent_key='C1')
            template['C2'] = murefi.Timeseries(t[1:4], [0]*3, independent_key='C', dependent_key='C2')

            # construct the symbolic computation graph
            prediction = model.predict_replicate(y0 + theta, template)

            self.assertIsInstance(prediction, murefi.Replicate)
            self.assertEqual(prediction.rid, 'TestRep')
            self.assertIn('A', prediction)
            self.assertFalse('B' in prediction)
            self.assertIn('C1', prediction)
            self.assertIn('C2', prediction)
            
            self.assertIsInstance(prediction['A'].y, tt.TensorVariable)
            self.assertIsInstance(prediction['C1'].y, tt.TensorVariable)
            self.assertIsInstance(prediction['C2'].y, tt.TensorVariable)

            outputs = [
                prediction['A'].y,
                prediction['C1'].y,
                prediction['C2'].y
            ]

            # compile a theano function for performing the computation
            f = theano.function(inputs, outputs)

            # compute the model outcome
            actual = f(0.85, 2.0)

            self.assertTrue(numpy.allclose(actual[0], [2.0, 1.4819299, 1.28322046]))
            self.assertTrue(numpy.allclose(actual[1], [0.71677954, 0.83004323]))
            self.assertTrue(numpy.allclose(actual[2], [0.5180701, 0.71677954, 0.83004323]))
        return
    
    @unittest.skipUnless(HAVE_PYMC3, 'requires PyMC3')
    def test_symbolic_predict_dataset(self):
        with theano.configparser.change_flags(compute_test_value='off'):
            inputs = [
                tt.scalar('beta', dtype=theano.config.floatX),
                tt.scalar('A', dtype=theano.config.floatX)
            ]
            theta = [0.23, inputs[0]]
            y0 = [inputs[1], 2., 0.]
            t = numpy.linspace(0, 1, 5)
            model = _mini_model()
            
            # create a parameter mapping
            mapping = pandas.DataFrame(columns='id,A0,B0,C0,alpha,beta'.split(',')).set_index('id')
            mapping.loc['TestRep'] = 'A0,B0,C0,alpha,beta'.split(',')
            mapping = mapping.reset_index()
            pm = murefi.ParameterMapping(mapping, bounds=dict(), guesses=dict())
            self.assertEqual(pm.ndim, 5)
            self.assertSequenceEqual(tuple(pm.parameters.keys()), 'A0,B0,C0,alpha,beta'.split(','))

            # create a dataset
            ds_template = murefi.Dataset()

            # One replicate with one observation of A, two observations of C
            template = murefi.Replicate('TestRep')
            template['A'] = murefi.Timeseries(t[:3], [0]*3, independent_key='A', dependent_key='A')
            template['C1'] = murefi.Timeseries(t[2:4], [0]*2, independent_key='C', dependent_key='C1')
            template['C2'] = murefi.Timeseries(t[1:4], [0]*3, independent_key='C', dependent_key='C2')
            ds_template['TestRep'] = template

            # construct the symbolic computation graph
            prediction = model.predict_dataset(ds_template, pm, parameters = y0 + theta)

            self.assertIsInstance(prediction, murefi.Dataset)
            self.assertIn('A', prediction['TestRep'])
            self.assertFalse('B' in prediction['TestRep'])
            self.assertIn('C1', prediction['TestRep'])
            self.assertIn('C2', prediction['TestRep'])
            
            self.assertIsInstance(prediction['TestRep']['A'].y, tt.TensorVariable)
            self.assertIsInstance(prediction['TestRep']['C1'].y, tt.TensorVariable)
            self.assertIsInstance(prediction['TestRep']['C2'].y, tt.TensorVariable)

            outputs = [
                prediction['TestRep']['A'].y,
                prediction['TestRep']['C1'].y,
                prediction['TestRep']['C2'].y
            ]

            # compile a theano function for performing the computation
            f = theano.function(inputs, outputs)

            # compute the model outcome
            actual = f(0.85, 2.0)

            self.assertTrue(numpy.allclose(actual[0], [2.0, 1.4819299, 1.28322046]))
            self.assertTrue(numpy.allclose(actual[1], [0.71677954, 0.83004323]))
            self.assertTrue(numpy.allclose(actual[2], [0.5180701, 0.71677954, 0.83004323]))
        return

    @unittest.skipUnless(HAVE_PYMC3, 'requires PyMC3')
    def test_integration_op(self):
        model = _mini_model()

        with pymc3.Model() as pmodel:
            inputs = [
                pymc3.Uniform('beta', 0, 1),
                pymc3.Uniform('A', 1, 3)
            ]
            theta = [0.23, inputs[0]]
            y0 = [inputs[1], 2., 0.]
            t = numpy.linspace(0, 1, 5)

            op = murefi.symbolic.IntegrationOp(model.solver, model.independent_keys)
            outputs = op(y0, t, theta)

            self.assertIsInstance(outputs, tt.TensorVariable)

            # compile a theano function for performing the computation
            f = theano.function(inputs, outputs)

            # compute the model outcome
            actual = f(0.83, 2.0)
            expected = model.solver([2., 2., 0.], t, [0.23, 0.83])
            
            self.assertTrue(numpy.allclose(actual[0], expected['A']))
            self.assertTrue(numpy.allclose(actual[1], expected['B']))
            self.assertTrue(numpy.allclose(actual[2], expected['C']))        
        return
    
    
    @unittest.skipUnless(HAVE_PYMC3, 'requires PyMC3')
    def test_computation_graph_for_dataset(self):
        with pymc3.Model() as pmodel:
            inputs = [
                pymc3.Uniform('beta', 0, 1),
                pymc3.Uniform('A', 1, 3)
            ]
            theta = [0.23, inputs[0]]
            y0 = [inputs[1], 2., 0.]
            t = numpy.linspace(0, 1, 5)
            model = _mini_model()
            
            # create a parameter mapping
            mapping = pandas.DataFrame(columns='id,A0,B0,C0,alpha,beta'.split(',')).set_index('id')
            mapping.loc['TestRep'] = 'A0,B0,C0,alpha,beta'.split(',')
            mapping.loc['TestRep2'] = 'A0,B0,C0,alpha,beta'.split(',')
            mapping = mapping.reset_index()
            pm = murefi.ParameterMapping(mapping, bounds=dict(), guesses=dict())
            self.assertEqual(pm.ndim, 5)
            self.assertSequenceEqual(tuple(pm.parameters.keys()), 'A0,B0,C0,alpha,beta'.split(','))

            # create a dataset
            ds_template = murefi.Dataset()

            # One replicate with one observation of A, two observations of C
            template = murefi.Replicate('TestRep')
            template['A'] = murefi.Timeseries(t[:3], [0]*3, independent_key='A', dependent_key='A')
            template['C1'] = murefi.Timeseries(t[2:4], [0]*2, independent_key='C', dependent_key='C1')
            template['C2'] = murefi.Timeseries(t[1:4], [0]*3, independent_key='C', dependent_key='C2')
            ds_template['TestRep'] = template
            template2 = murefi.Replicate('TestRep2')
            template2['A'] = murefi.Timeseries(t[:3], [0]*3, independent_key='A', dependent_key='A')
            template2['C1'] = murefi.Timeseries(t[2:4], [0]*2, independent_key='C', dependent_key='C1')
            ds_template['TestRep2'] = template2
            
            objective = murefi.objectives.for_dataset(ds_template, model, pm, error_models=[
                    _mini_error_model('A', 'A'),
                    _mini_error_model('C', 'C2'),
                    _mini_error_model('C', 'C1'),
            ])
            L = objective(y0 + theta)
            self.assertTrue(len(L) == 5)
            self.assertTrue(calibr8.istensor(L))

        return


class TestHDF5storage(unittest.TestCase):
    def _test_save_and_load(self, ds_original):
        # use a temporary directory, because a tempfile.NamedTemporaryFile can not be opened
        # multiple times on all platforms (https://docs.python.org/3/library/tempfile.html#tempfile.NamedTemporaryFile)
        with tempfile.TemporaryDirectory() as dir:
            fp = pathlib.Path(dir, 'testing.h5')
            murefi.save_dataset(ds_original, fp)
            ds_loaded = murefi.load_dataset(fp)

        self.assertIsInstance(ds_loaded, murefi.Dataset)
        self.assertEqual(set(ds_original.keys()), set(ds_loaded.keys()))

        for rid, rep_orig in ds_original.items():
            for dkey, ts_orig in rep_orig.items():
                ts_loaded = ds_loaded[rid][dkey]
                self.assertIsInstance(ts_loaded, murefi.Timeseries)
                self.assertEqual(ts_orig.independent_key, ts_loaded.independent_key)
                self.assertEqual(ts_orig.dependent_key, ts_loaded.dependent_key)
                numpy.testing.assert_array_equal(ts_orig.t, ts_loaded.t)
                numpy.testing.assert_array_equal(ts_orig.y, ts_loaded.y)
        return

    def test_empty_dataset(self):
        ds = murefi.Dataset()
        self._test_save_and_load(ds)
        
    def test_standard_dataset(self):
        ds = murefi.Dataset.make_template(tmin=0, tmax=5, independent_keys='SXP', rids=['R1', 'R2', 'R3'])
        self._test_save_and_load(ds)
        return

    def test_empty_replicate(self):
        ds = murefi.Dataset.make_template(tmin=0, tmax=5, independent_keys='SXP', rids=['R1', 'R2', 'R3'])
        ds['R2'].pop('S')
        ds['R2'].pop('X')
        ds['R2'].pop('P')
        self.assertEqual(len(ds['R2']), 0)
        self._test_save_and_load(ds)
        return


if __name__ == '__main__':
    unittest.main(exit=False)