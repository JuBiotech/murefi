import collections
import h5py
import numpy
import pandas
import pathlib
import pytest
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
        def dydt(self, y, t, ode_parameters):
            A, B, C = y
            alpha, beta = ode_parameters
            dCdt = alpha*A + beta*B**2
            dAdt = -dCdt
            dBdt = -2*dCdt
            return [dAdt, dBdt, dCdt]
    return MiniModel(parameter_names=['A0', 'B0', 'C0', 'alpha', 'beta'], independent_keys=['A', 'B', 'C'])


def _mini_calibration_model(independent:str, dependent:str):
    class CM(calibr8.BasePolynomialModelT):
        def __init__(self):
            super().__init__(independent_key=independent, dependent_key=dependent, mu_degree=1, scale_degree=0)
    cm = CM()
    cm.theta_fitted = [0, 1, 1, 100]
    assert len(cm.theta_fitted) == len(cm.theta_names)
    return cm


@pytest.fixture
def df_mapping():
    map_df = pandas.DataFrame(columns="rid;S_0;X_0;mue_max;K_S;Y_XS;t_lag;t_acc".split(";")).set_index("rid")
    map_df.loc["A01"] = ("test1A", "test1B", 3, 4, 5, 6, 7)
    map_df.loc["B02"] = (11, "test1B", "test2C", "test2D", 15, 16, 17)
    return map_df


class TestExceptions:
    def test_dtype_error(self):
        with pytest.raises(murefi.DtypeError):
            raise murefi.DtypeError('Just the message.')
        with pytest.raises(murefi.DtypeError):
            raise murefi.DtypeError('With types.', actual=str)
        with pytest.raises(murefi.DtypeError):
            raise murefi.DtypeError('With types.', expected=str)
        with pytest.raises(murefi.DtypeError):
            raise murefi.DtypeError('With types.', actual=int, expected=str)
        pass

    def test_shape_error(self):
        with pytest.raises(murefi.ShapeError):
            raise murefi.ShapeError('Just the message.')
        with pytest.raises(murefi.ShapeError):
            raise murefi.ShapeError('With shapes.', actual=(2,3))
        with pytest.raises(murefi.ShapeError):
            raise murefi.ShapeError('With shapes.', expected='(2,3) or (5,6')
        with pytest.raises(murefi.ShapeError):
            raise murefi.ShapeError('With shapes.', actual=(), expected='(5,4) or (?,?,6)')
        pass


class TestParameterMapping:
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

    def test_init(self, df_mapping):
        parmap = murefi.ParameterMapping(df_mapping, bounds=self.bounds, guesses=self.initial_guesses)
        assert parmap.order == ('S_0', 'X_0', 'mue_max', 'K_S', 'Y_XS', 't_lag', 't_acc')
        assert parmap.parameters == collections.OrderedDict([
            ('test1A', 'S_0'),
            ('test1B', 'X_0'),
            ('test2C', 'mue_max'),
            ('test2D', 'K_S')
        ])
        assert parmap.theta_names == tuple(parmap.parameters.keys())
        numpy.testing.assert_array_equal(parmap.theta_names, tuple(parmap.parameters.keys()))
        assert parmap.ndim == 4
        assert parmap.bounds == ((1,3), (3,4), (5,6), (7,8))
        assert parmap.guesses == (0.1, 0.2, 0.3, 0.4)
        assert parmap.mapping == {
            'A01':('test1A', 'test1B', 3.0, 4.0, 5.0, 6.0, 7.0),
            'B02':(11.0, 'test1B', 'test2C', 'test2D', 15.0, 16.0, 17.0)
        }
        numpy.testing.assert_array_equal(
            parmap.merge_vectors(parmap.coords),
            tuple(parmap.parameters.keys())
        )
        parmap = murefi.ParameterMapping(df_mapping, bounds=None, guesses=None)
        assert parmap.order == ('S_0', 'X_0', 'mue_max', 'K_S', 'Y_XS', 't_lag', 't_acc')
        assert parmap.parameters == collections.OrderedDict([
            ('test1A', 'S_0'),
            ('test1B', 'X_0'),
            ('test2C', 'mue_max'),
            ('test2D', 'K_S')
        ])
        assert parmap.ndim == 4
        assert parmap.bounds == ((None, None), (None, None), (None, None), (None, None))
        assert parmap.guesses == (None, None, None, None)
        assert parmap.mapping == {
            'A01':('test1A', 'test1B', 3.0, 4.0, 5.0, 6.0, 7.0),
            'B02':(11.0, 'test1B', 'test2C', 'test2D', 15.0, 16.0, 17.0)
        }
        
        with pytest.warns(UserWarning, match="should be named 'rid'"):
            dfcopy = df_mapping.copy()
            dfcopy.index.name = "id"
            murefi.ParameterMapping(dfcopy, bounds=None, guesses=None)
        pass

    def test_invalid_init(self):
        mapfail_df = pandas.DataFrame(columns="rid;S_0;X_0;mue_max;K_S;Y_XS;t_lag;t_acc".split(";")).set_index("rid")
        # the "test1B" parameter is used in two columns:
        mapfail_df.loc["A01"] = ("test1A", "test1B", "test1B", 4, 5, 6, 7)
        mapfail_df.loc["B02"] = (11, "test1B", "test2C", "test2D", 15, 16, 17)
        with pytest.raises(TypeError):
            murefi.ParameterMapping(mapfail_df, self.bounds, self.initial_guesses)
        with pytest.raises(ValueError):
            murefi.ParameterMapping(mapfail_df, bounds=self.bounds, guesses=self.initial_guesses)
        pass

    def test_as_dataframe(self, df_mapping):
        parmap = murefi.ParameterMapping(df_mapping, bounds=self.bounds, guesses=self.initial_guesses)
        df = parmap.as_dataframe()
        assert isinstance(df, pandas.DataFrame)
        assert df.index.name == "rid"
        numpy.testing.assert_array_equal(df.index, df_mapping.index)
        numpy.testing.assert_array_equal(df.columns, df_mapping.columns)
        numpy.testing.assert_array_equal(df.values, df_mapping.values)
        pass

    def test_repmap_dict_missing_one(self, df_mapping):
        parmap = murefi.ParameterMapping(df_mapping, bounds=self.bounds, guesses=self.initial_guesses)

        p_kick = 'test1A'
        with pytest.raises(KeyError, match="Parameters {'" + p_kick + "'} are missing"):
            parameters = {
                pname : pguess
                for pname, pguess in zip(parmap.parameters.keys(), parmap.guesses)
            }
            parameters.pop(p_kick)
            parmap.repmap(parameters)
        pass

    def test_repmap_array_missing_one(self, df_mapping):
        parmap = murefi.ParameterMapping(df_mapping, bounds=self.bounds, guesses=self.initial_guesses)
        with pytest.raises(murefi.ShapeError):
            parmap.repmap(parmap.guesses[:-1])
        pass

    def test_repmap_array(self, df_mapping):
        parmap = murefi.ParameterMapping(df_mapping, bounds=self.bounds, guesses=self.initial_guesses)
        theta_fitted = [1,2,13,14]
        expected = {
            'A01': numpy.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
            'B02': numpy.array([11.0, 2.0, 13.0, 14.0, 15.0, 16.0, 17.0])
        }
        assert parmap.repmap(theta_fitted).keys() == expected.keys()
        for key in expected.keys():
            numpy.testing.assert_array_equal(parmap.repmap(theta_fitted)[key], expected[key])
        pass

    def test_repmap_dict(self, df_mapping):
        parmap = murefi.ParameterMapping(df_mapping, bounds=self.bounds, guesses=self.initial_guesses)
        theta_fitted = dict(test1A=1.0, test1B=2.0, test2C=13, test2D=14)
        expected = {
            'A01': numpy.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
            'B02': numpy.array([11.0, 2.0, 13.0, 14.0, 15.0, 16.0, 17.0])
        }
        assert parmap.repmap(theta_fitted).keys() == expected.keys()
        for key in expected.keys():
            numpy.testing.assert_array_equal(parmap.repmap(theta_fitted)[key], expected[key])
        pass

    def test_repmap_2d_array(self, df_mapping):
        parmap = murefi.ParameterMapping(df_mapping, bounds=self.bounds, guesses=self.initial_guesses)
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

    def test_repmap_2d_mixed_dict(self, df_mapping):
        parmap = murefi.ParameterMapping(df_mapping, bounds=self.bounds, guesses=self.initial_guesses)
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


class TestDataset:
    def test_dataset(self):
        rep = murefi.Replicate('A01')
        rep['S_observed'] = murefi.Timeseries([0,2,3,  5,  8  ], [1,2,3,4,5], independent_key='S', dependent_key='S_observed')
        ds = murefi.Dataset()
        ds['A01'] = rep
        assert 'A01' in ds
        assert list(ds.keys()) == ['A01']
        return

    def test_make_template(self):
        template = murefi.Dataset.make_template(0.5, 3.5, independent_keys='ABC', rids='R1,R2,R3,R4'.split(','), N=20)
        assert isinstance(template, murefi.Dataset)
        assert 'R1' in template
        assert 'R2' in template
        assert 'R3' in template
        assert template['R1']['A'].t[0] == 0.5
        assert template['R2']['B'].t[-1] == 3.5
        assert len(template['R3']['C'].t) == 20
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

    def test_assign_wrong_key(self):
        ds = murefi.Dataset()
        rep = murefi.Replicate('A01')
        with pytest.raises(KeyError, match="match"):
            ds['bla'] = rep
        pass


class TestReplicate:
    def test_t_any(self):
        rep = murefi.Replicate('A01')
        assert rep.t_any is None
        rep['S_observed'] = murefi.Timeseries([0,2,3,  5,  8  ], [1,2,3,4,5], independent_key='S', dependent_key='S_observed')
        rep['X_observed'] = murefi.Timeseries([  2,3,4,  6,8,9], [1,2,3,4,5,6], independent_key='X', dependent_key='X_observed')
        numpy.testing.assert_array_equal(rep.t_any, [0,2,3,4,5,6,8,9])
        return

    def test_observation_booleans(self):
        rep = murefi.Replicate('A01')
        rep['S_observed'] = murefi.Timeseries([0,2,3,  5,  8  ], [1,2,3,4,5], independent_key='S', dependent_key='S_observed')
        rep['X_observed'] = murefi.Timeseries([  2,3,4,  6,8,9], [1,2,3,4,5,6], independent_key='X', dependent_key='X_observed')
        result = rep.get_observation_booleans(['S_observed', 'X_observed', 'P_observed'])
        numpy.testing.assert_array_equal(result['S_observed'], [True,True,True,False,True,False,True,False])
        numpy.testing.assert_array_equal(result['X_observed'], [False,True,True,True,False,True,True,True])
        numpy.testing.assert_array_equal(result['P_observed'], [False,False,False,False,False,False,False,False])
        return

    def test_observation_indices(self):
        rep = murefi.Replicate('A01')
        rep['S_observed'] = murefi.Timeseries([0,2,3,  5,  8  ], [1,2,3,4,5], independent_key='S', dependent_key='S_observed')
        rep['X_observed'] = murefi.Timeseries([  2,3,4,  6,8,9], [1,2,3,4,5,6], independent_key='X', dependent_key='X_observed')
        result = rep.get_observation_indices(['S_observed', 'X_observed', 'P_observed'])
        numpy.testing.assert_array_equal(result['S_observed'], [0,1,2,4,6])
        numpy.testing.assert_array_equal(result['X_observed'], [1,2,3,5,6,7])
        numpy.testing.assert_array_equal(result['P_observed'], [])
        return

    def test_make_template(self):
        template = murefi.Replicate.make_template(0.5, 3.5, independent_keys=['A', 'B', 'C'], N=20)
        assert 'A' in template
        assert 'B' in template
        assert 'C' in template
        assert template['A'].t[0] == 0.5
        assert template['A'].t[-1] == 3.5
        assert len(template['A'].t) == 20
        return

    def test_no_duplicate_time_in_templates(self):
        template = murefi.Replicate.make_template(tmin=2, tmax=2, independent_keys="ABC")
        assert len(template["B"].t) == 1
        pass

    def test_str_repr(self):
        template = murefi.Replicate.make_template(0.5, 3.5, independent_keys=['A', 'B', 'C'], N=20)
        expected = 'Replicate(A[:20], B[:20], C[:20])'
        assert template.__str__() == expected
        assert template.__repr__() == expected
        pass


class TestTimeseries:
    def test_t_monotonic(self):
        with pytest.raises(ValueError):
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

        with pytest.raises(murefi.ShapeError):
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

        with pytest.raises(murefi.ShapeError):
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

    def test_inputchecks(self):
        with pytest.raises(murefi.DtypeError):
            murefi.Timeseries(t=4, y=[1], independent_key='I', dependent_key='D')
        with pytest.raises(murefi.DtypeError):
            murefi.Timeseries(t=[4,5], y='ab', independent_key='I', dependent_key='D')
        with pytest.raises(murefi.DtypeError):
            murefi.Timeseries(t=[4,5], y=(1,2), independent_key=15, dependent_key='D')
        with pytest.raises(murefi.DtypeError):
            murefi.Timeseries(t=[4,5], y=(1,2), independent_key='I', dependent_key=0.7)
        pass

    def test_str_repr(self):
        ts = murefi.Timeseries(t=[4,5], y=(1,2), independent_key='I', dependent_key='D')
        expected = 'D[:2]'
        assert ts.__str__() == expected
        assert ts.__repr__() == expected
        pass


class TestBaseODEModel:
    def test_attributes(self):
        model = _mini_model()

        assert isinstance(model, murefi.BaseODEModel)
        assert model.n_y0 == 3
        numpy.testing.assert_array_equal(model.independent_keys, ['A', 'B', 'C'])
        return

    def test_solver(self):
        ode_parameters = [0.23, 0.85]
        y0 = [2., 2., 0.]
        T = 5
        t = numpy.linspace(0, 1, T)
        model = _mini_model()      

        y_hat = model.solver(y0, t, ode_parameters)
        assert isinstance(y_hat, dict)
        assert 'A' in y_hat
        assert 'B' in y_hat
        assert 'C' in y_hat
        for ikey in model.independent_keys:
            assert isinstance(y_hat[ikey], numpy.ndarray)
            assert y_hat[ikey].shape == (T,)
        assert numpy.allclose(y_hat['A'], [2.0, 1.4819299, 1.28322046, 1.16995677, 1.09060199])
        assert numpy.allclose(y_hat['B'], [2.0, 0.9638598, 0.56644092, 0.33991354, 0.18120399])
        assert numpy.allclose(y_hat['C'], [0.0, 0.5180701, 0.71677954, 0.83004323, 0.90939801])
        return

    def test_solver_no_zero_time(self):
        ode_parameters = [0.23, 0.85]
        y0 = [2., 2., 0.]
        t = numpy.linspace(0, 1, 5)[1:]
        model = _mini_model()      

        y_hat = model.solver(y0, t, ode_parameters)
        assert numpy.allclose(y_hat['A'], [1.4819299, 1.28322046, 1.16995677, 1.09060199])
        assert numpy.allclose(y_hat['B'], [0.9638598, 0.56644092, 0.33991354, 0.18120399])
        assert numpy.allclose(y_hat['C'], [0.5180701, 0.71677954, 0.83004323, 0.90939801])
        return

    def test_solver_vectorized(self):
        S = 300
        T = 25

        model = _mini_model()      
        y0 = numpy.array([[2., 2., 0.]] * S).T
        ode_parameters = numpy.array([[0.23, 0.85]] * S).T
        t = numpy.linspace(0.2, 2, T)
        assert y0.shape == (3, S)
        assert ode_parameters.shape == (2, S)
        assert t.shape == (T,)

        y_hat = model.solver_vectorized(y0, t, ode_parameters)
        assert isinstance(y_hat, dict)
        for ikey in model.independent_keys:
            assert ikey in y_hat
            assert isinstance(y_hat[ikey], numpy.ndarray)
            assert y_hat[ikey].shape == (T, S)

        with pytest.raises(murefi.ShapeError, match="y0"):
            model.solver_vectorized([1,2,3,5,6], t, ode_parameters)

        with pytest.raises(murefi.ShapeError, match="ode_parameters"):
            model.solver_vectorized(y0, t, ode_parameters[:-1])
        pass

    def test_predict_replicate(self):
        ode_parameters = [0.23, 0.85]
        y0 = [2., 2., 0.]
        t = numpy.linspace(0, 1, 5)
        model = _mini_model()
        
        template = murefi.Replicate('TestRep')
        # one observation of A, two observations of C
        template['A'] = murefi.Timeseries(t[:3], [0]*3, independent_key='A', dependent_key='A')
        template['C1'] = murefi.Timeseries(t[2:4], [0]*2, independent_key='C', dependent_key='C1')
        template['C2'] = murefi.Timeseries(t[1:4], [0]*3, independent_key='C', dependent_key='C2')
        prediction = model.predict_replicate(y0 + ode_parameters, template)

        assert isinstance(prediction, murefi.Replicate)
        assert prediction.rid == 'TestRep'
        assert 'A' in prediction
        assert 'B' not in prediction
        assert 'C1' in prediction
        'C2' in prediction

        assert numpy.allclose(prediction['A'].y, [2.0, 1.4819299, 1.28322046])
        assert numpy.allclose(prediction['C1'].y, [0.71677954, 0.83004323])
        assert numpy.allclose(prediction['C2'].y, [0.5180701, 0.71677954, 0.83004323])
        pass

    def test_predict_replicate_distribution(self):
        t = numpy.linspace(0, 1, 5)
        model = _mini_model()
        template = murefi.Replicate('TestRep')
        # one observation of A, two observations of C
        template['A'] = murefi.Timeseries(t[:3], [0]*3, independent_key='A', dependent_key='A')
        template['C1'] = murefi.Timeseries(t[2:4], [0]*2, independent_key='C', dependent_key='C1')
        template['C2'] = murefi.Timeseries(t[1:4], [0]*3, independent_key='C', dependent_key='C2')

        P = len(model.parameter_names)
        S = 300
        pred = model.predict_replicate(template=template, parameters=numpy.ones((P, S)))
        for dkey, ts_pred in pred.items():
            ts_template = template[dkey]
            numpy.testing.assert_array_equal(ts_pred.t, ts_template.t)
            assert numpy.shape(ts_pred.y) == (S, len(ts_template))

        with pytest.raises(murefi.ShapeError, match="inconsistent"):
            parameters = [1] * P
            parameters[0] = numpy.ones(shape=(12,))
            parameters[1] = numpy.ones(shape=(13,))
            model.predict_replicate(template=template, parameters=parameters)
        pass

    def test_predict_replicate_inputchecks(self):
        t = numpy.linspace(0, 1, 5)
        model = _mini_model()
        template = murefi.Replicate('TestRep')
        # one observation of A, two observations of C
        template['A'] = murefi.Timeseries(t[:3], [0]*3, independent_key='A', dependent_key='A')
        template['C1'] = murefi.Timeseries(t[2:4], [0]*2, independent_key='C', dependent_key='C1')
        template['C2'] = murefi.Timeseries(t[1:4], [0]*3, independent_key='C', dependent_key='C2')

        P = len(model.parameter_names)

        # wrong template
        with pytest.raises(ValueError, match="template"):
            model.predict_replicate(template='A01', parameters=numpy.ones((P,)))

        # wrong parameter types
        with pytest.raises(murefi.DtypeError, match="parameters"):
            model.predict_replicate(template=template, parameters={
                pname : 1
                for pname in model.parameter_names
            })

        # wrong parameter shapes
        with pytest.raises(murefi.ShapeError):
            # wrong number of parameters
            model.predict_replicate(template=template, parameters=numpy.ones((P+2,)))
        with pytest.raises(murefi.ShapeError):
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
        mapping = pandas.DataFrame(columns='rid,A0,B0,C0,alpha,beta'.split(',')).set_index('rid')
        mapping.loc['R1'] = 'A0,B0,C0,alpha_1,beta'.split(',')
        mapping.loc['R2'] = 'A0,B0,C0,alpha_2,beta'.split(',')
        pm = murefi.ParameterMapping(mapping, bounds=dict(), guesses=dict())
        assert pm.ndim == 6

        
        # set a global parameter vector with alpha_1=0.22, alpha_2=0.24
        numpy.testing.assert_array_equal(tuple(pm.parameters.keys()), 'A0,B0,C0,alpha_1,alpha_2,beta'.split(','))
        theta = [2., 2., 0.] + [0.22, 0.24, 0.85]
        prediction = model.predict_dataset(template=dataset, parameter_mapping=pm, parameters=theta)

        assert isinstance(prediction, murefi.Dataset)
        assert 'R1' in prediction
        assert 'R2' in prediction
        assert 'A' in prediction['R1']
        assert 'B' in prediction['R1']
        assert 'C' not in prediction['R1']
        assert 'A' not in prediction['R2']
        assert 'B' in prediction['R2']
        assert 'C' in prediction['R2']
        assert len(prediction['R1'].t_any) == 60
        assert len(prediction['R2'].t_any) == 20

        # test that it checks the order
        model.parameter_names = model.parameter_names[::-1]
        with pytest.raises(ValueError, match="order"):
            model.predict_dataset(dataset, pm, theta)
        return

    def test_predict_dataset_distribution(self):
        model = _mini_model()
        
        # create a template dataset
        dataset = murefi.Dataset()
        dataset['R1'] = murefi.Replicate.make_template(0, 1, 'AB', N=60, rid='R1')
        dataset['R2'] = murefi.Replicate.make_template(0.2, 1, 'BC', N=20, rid='R2')

        # create a parameter mapping that uses replicate-wise alpha parameters (6 dims)
        mapping = pandas.DataFrame(columns='rid,A0,B0,C0,alpha,beta'.split(',')).set_index('rid')
        mapping.loc['R1'] = 'A0,B0,C0,alpha_1,beta'.split(',')
        mapping.loc['R2'] = 'A0,B0,C0,alpha_2,beta'.split(',')
        pm = murefi.ParameterMapping(mapping, bounds=dict(), guesses=dict())
        assert pm.ndim == 6

        # set a global parameter vector with alpha_1=0.22, alpha_2=0.24
        numpy.testing.assert_array_equal(tuple(pm.parameters.keys()), 'A0,B0,C0,alpha_1,alpha_2,beta'.split(','))
        theta = [2., 2., 0.] + [0.22, 0.24, 0.85]
        # randomize the parameter vector into a matrix
        P = len(theta)
        S = 40
        theta = numpy.array(theta)[:,numpy.newaxis] + numpy.random.uniform(0, 0.1, size=(P, S))
        assert theta.shape == (P, S)

        prediction = model.predict_dataset(template=dataset, parameter_mapping=pm, parameters=theta)
        assert isinstance(prediction, murefi.Dataset)
        assert 'R1' in prediction
        assert 'R2' in prediction
        assert 'A' in prediction['R1']
        assert 'B' in prediction['R1']
        assert 'C' not in prediction['R1']
        assert 'A' not in prediction['R2']
        assert 'B' in prediction['R2']
        assert 'C' in prediction['R2']
        assert len(prediction['R1'].t_any) == 60
        assert len(prediction['R2'].t_any) == 20
        for rid, rep in prediction.items():
            for dkey, ts in rep.items():
                assert numpy.shape(ts.y) == (S, len(dataset[rid][dkey]))
        pass


class TestObjectives:
    def _prepare(self):
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
        mapping = pandas.DataFrame(columns='rid,A0,B0,C0,alpha,beta'.split(',')).set_index('rid')
        mapping.loc['R1'] = 'A0,B0,C0,alpha_1,beta'.split(',')
        mapping.loc['R2'] = 'A0,B0,C0,alpha_2,beta'.split(',')
        pm = murefi.ParameterMapping(mapping, bounds=dict(), guesses=dict())
        assert pm.ndim == 6
        return model, dataset, pm

    def test_for_dataset_creation(self):
        model, dataset, pm = self._prepare()
        assert pm.ndim == 6

        obj = murefi.objectives.for_dataset(dataset, model, pm, calibration_models=[
            _mini_calibration_model('A', 'A'),
            _mini_calibration_model('B', 'B'),
            _mini_calibration_model('C', 'C'),
        ])
        
        assert callable(obj)
        theta = [2., 2., 0.] + [0.22, 0.24, 0.85]
        L = obj(theta)
        assert isinstance(L, float)
        assert numpy.isfinite(L)
        pass

    def test_for_dataset_checks_theta_order(self):
        model, dataset, pm = self._prepare()
        
        # manipulate the order of parameters the model expects
        model.parameter_names = model.parameter_names[::-1]
        with pytest.raises(ValueError):
            obj = murefi.objectives.for_dataset(dataset, model, pm, calibration_models=[
                _mini_calibration_model('A', 'A'),
                _mini_calibration_model('B', 'B'),
                _mini_calibration_model('C', 'C'),
            ])
        pass

    def test_for_dataset_inf_on_nan(self):
        model, dataset, pm = self._prepare()
        assert pm.ndim == 6

        # manipulate an observation into NaN
        dataset['R1']['A'].y[0] = numpy.nan

        obj = murefi.objectives.for_dataset(dataset, model, pm, calibration_models=[
            _mini_calibration_model('A', 'A'),
            _mini_calibration_model('B', 'B'),
            _mini_calibration_model('C', 'C'),
        ])

        assert callable(obj)
        theta = [2., 2., 0.] + [0.22, 0.24, 0.85]
        L = obj(theta)
        assert isinstance(L, float)
        assert numpy.isinf(L)
        pass


class TestSymbolicComputation:
    @pytest.mark.skipif(not HAVE_PYMC3, reason='requires PyMC3')
    def test_timeseries_support(self):
        t = numpy.linspace(0, 10, 10)
        with theano.config.change_flags(compute_test_value='off'):
            y = tt.scalar('TestY', dtype=theano.config.floatX)
            assert isinstance(y, tt.TensorVariable)
            ts = murefi.Timeseries(t, y, independent_key='Test', dependent_key='Test')
        return

    @pytest.mark.skipif(not HAVE_PYMC3, reason='requires PyMC3')
    def test_symbolic_parameter_mapping(self, df_mapping):
        parmap = murefi.ParameterMapping(df_mapping, bounds=dict(
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
        with theano.config.change_flags(compute_test_value='off'):
            theta_fitted = [1, tt.scalar('mu_max', dtype=theano.config.floatX), 13, tt.scalar('t_acc', dtype=theano.config.floatX)]

            # map it to the two replicates
            expected = {
                'A01': numpy.array([1.0, None, 3.0, 4.0, 5.0, 6.0, 7.0]),
                'B02': numpy.array([11.0, None, 13.0, None, 15.0, 16.0, 17.0])
            }
            mapped = parmap.repmap(theta_fitted)
            assert mapped.keys() == expected.keys()
            for rid in expected.keys():
                for exp, act in zip(expected[rid], mapped[rid]):
                    if exp is not None:
                        assert exp == act
                    else:
                        assert isinstance(act, tt.TensorVariable)
        return

    @pytest.mark.skipif(not HAVE_PYMC3, reason='requires PyMC3')
    def test_symbolic_predict_replicate(self):
        with theano.config.change_flags(compute_test_value='off'):
            inputs = [
                tt.scalar('beta', dtype=theano.config.floatX),
                tt.scalar('A', dtype=theano.config.floatX)
            ]
            ode_parameters = [0.23, inputs[0]]
            y0 = [inputs[1], 2., 0.]
            t = numpy.linspace(0, 1, 5)
            model = _mini_model()
            
            template = murefi.Replicate('TestRep')
            # one observation of A, two observations of C
            template['A'] = murefi.Timeseries(t[:3], [0]*3, independent_key='A', dependent_key='A')
            template['C1'] = murefi.Timeseries(t[2:4], [0]*2, independent_key='C', dependent_key='C1')
            template['C2'] = murefi.Timeseries(t[1:4], [0]*3, independent_key='C', dependent_key='C2')

            # construct the symbolic computation graph
            prediction = model.predict_replicate(y0 + ode_parameters, template)

            assert isinstance(prediction, murefi.Replicate)
            assert prediction.rid == 'TestRep'
            assert 'A' in prediction
            assert 'B' not in prediction
            assert 'C1' in prediction
            assert 'C2' in prediction
            
            assert isinstance(prediction['A'].y, tt.TensorVariable)
            assert isinstance(prediction['C1'].y, tt.TensorVariable)
            assert isinstance(prediction['C2'].y, tt.TensorVariable)

            outputs = [
                prediction['A'].y,
                prediction['C1'].y,
                prediction['C2'].y
            ]

            # compile a theano function for performing the computation
            f = theano.function(inputs, outputs)

            # compute the model outcome
            actual = f(0.85, 2.0)

            assert numpy.allclose(actual[0], [2.0, 1.4819299, 1.28322046])
            assert numpy.allclose(actual[1], [0.71677954, 0.83004323])
            assert numpy.allclose(actual[2], [0.5180701, 0.71677954, 0.83004323])
        return
    
    @pytest.mark.skipif(not HAVE_PYMC3, reason='requires PyMC3')
    def test_symbolic_predict_dataset(self):
        with theano.config.change_flags(compute_test_value='off'):
            inputs = [
                tt.scalar('beta', dtype=theano.config.floatX),
                tt.scalar('A', dtype=theano.config.floatX)
            ]
            ode_parameters = [0.23, inputs[0]]
            y0 = [inputs[1], 2., 0.]
            t = numpy.linspace(0, 1, 5)
            model = _mini_model()
            
            # create a parameter mapping
            mapping = pandas.DataFrame(columns='rid,A0,B0,C0,alpha,beta'.split(',')).set_index('rid')
            mapping.loc['TestRep'] = 'A0,B0,C0,alpha,beta'.split(',')
            pm = murefi.ParameterMapping(mapping, bounds=dict(), guesses=dict())
            assert pm.ndim == 5
            numpy.testing.assert_array_equal(tuple(pm.parameters.keys()), 'A0,B0,C0,alpha,beta'.split(','))

            # create a dataset
            ds_template = murefi.Dataset()

            # One replicate with one observation of A, two observations of C
            template = murefi.Replicate('TestRep')
            template['A'] = murefi.Timeseries(t[:3], [0]*3, independent_key='A', dependent_key='A')
            template['C1'] = murefi.Timeseries(t[2:4], [0]*2, independent_key='C', dependent_key='C1')
            template['C2'] = murefi.Timeseries(t[1:4], [0]*3, independent_key='C', dependent_key='C2')
            ds_template['TestRep'] = template

            # construct the symbolic computation graph
            prediction = model.predict_dataset(ds_template, pm, parameters = y0 + ode_parameters)

            assert isinstance(prediction, murefi.Dataset)
            assert 'A' in prediction['TestRep']
            assert 'B' not in prediction['TestRep']
            assert 'C1' in prediction['TestRep']
            assert 'C2' in prediction['TestRep']
            
            assert isinstance(prediction['TestRep']['A'].y, tt.TensorVariable)
            assert isinstance(prediction['TestRep']['C1'].y, tt.TensorVariable)
            assert isinstance(prediction['TestRep']['C2'].y, tt.TensorVariable)

            outputs = [
                prediction['TestRep']['A'].y,
                prediction['TestRep']['C1'].y,
                prediction['TestRep']['C2'].y
            ]

            # compile a theano function for performing the computation
            f = theano.function(inputs, outputs)

            # compute the model outcome
            actual = f(0.85, 2.0)

            assert numpy.allclose(actual[0], [2.0, 1.4819299, 1.28322046])
            assert numpy.allclose(actual[1], [0.71677954, 0.83004323])
            assert numpy.allclose(actual[2], [0.5180701, 0.71677954, 0.83004323])
        return

    @pytest.mark.skipif(not HAVE_PYMC3, reason='requires PyMC3')
    def test_integration_op(self):
        model = _mini_model()

        with pymc3.Model() as pmodel:
            inputs = [
                pymc3.Uniform('beta', 0, 1),
                pymc3.Uniform('A', 1, 3)
            ]
            ode_parameters = [0.23, inputs[0]]
            y0 = [inputs[1], 2., 0.]
            t = numpy.linspace(0, 1, 5)

            op = murefi.symbolic.IntegrationOp(model.solver, model.independent_keys)
            outputs = op(y0, t, ode_parameters)

            assert isinstance(outputs, tt.TensorVariable)

            # compile a theano function for performing the computation
            f = theano.function(inputs, outputs)

            # compute the model outcome
            actual = f(0.83, 2.0)
            expected = model.solver([2., 2., 0.], t, [0.23, 0.83])
            
            assert numpy.allclose(actual[0], expected['A'])
            assert numpy.allclose(actual[1], expected['B'])
            assert numpy.allclose(actual[2], expected['C'])
        return

    @pytest.mark.skipif(not HAVE_PYMC3, reason='requires PyMC3')
    def test_computation_graph_for_dataset(self):
        with pymc3.Model() as pmodel:
            inputs = [
                pymc3.Uniform('beta', 0, 1),
                pymc3.Uniform('A', 1, 3)
            ]
            ode_parameters = [0.23, inputs[0]]
            y0 = [inputs[1], 2., 0.]
            t = numpy.linspace(0, 1, 5)
            model = _mini_model()
            
            # create a parameter mapping
            mapping = pandas.DataFrame(columns='rid,A0,B0,C0,alpha,beta'.split(',')).set_index('rid')
            mapping.loc['TestRep'] = 'A0,B0,C0,alpha,beta'.split(',')
            mapping.loc['TestRep2'] = 'A0,B0,C0,alpha,beta'.split(',')
            pm = murefi.ParameterMapping(mapping, bounds=dict(), guesses=dict())
            assert pm.ndim == 5
            numpy.testing.assert_array_equal(tuple(pm.parameters.keys()), 'A0,B0,C0,alpha,beta'.split(','))

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
            
            objective = murefi.objectives.for_dataset(ds_template, model, pm, calibration_models=[
                    _mini_calibration_model('A', 'A'),
                    _mini_calibration_model('C', 'C2'),
                    _mini_calibration_model('C', 'C1'),
            ])
            L = objective(y0 + ode_parameters)
            assert calibr8.istensor(L)
            assert L.ndim == 0

        return

    @pytest.mark.skipif(not HAVE_PYMC3, reason='requires PyMC3')
    def test_predict_replicate(self):
        t = numpy.linspace(0, 1, 5)
        model = _mini_model()
        template = murefi.Replicate('TestRep')
        # one observation of A, two observations of C
        template['A'] = murefi.Timeseries(t[:3], [0]*3, independent_key='A', dependent_key='A')
        template['C1'] = murefi.Timeseries(t[2:4], [0]*2, independent_key='C', dependent_key='C1')
        template['C2'] = murefi.Timeseries(t[1:4], [0]*3, independent_key='C', dependent_key='C2')

        P = len(model.parameter_names)

        # mix of scalars, vectors, tensors
        with pytest.raises(murefi.DtypeError, match="incompatible"):
            with pymc3.Model():
                parameters = [1] * P
                parameters[0] = pymc3.Uniform('u')
                parameters[1] = numpy.ones(shape=(13,))
                model.predict_replicate(template=template, parameters=parameters)

        # mix of tensor and scalars
        with pymc3.Model():
            parameters = [1] * P
            parameters[0] = pymc3.Uniform('u')
            if murefi.symbolic.HAVE_SUNODE:
                # Workaround for https://github.com/aseyboldt/sunode/issues/16
                # is to include a free parameter in the ODE parameters:
                parameters[-1] = pymc3.Normal("n", mu=1)

            rep = model.predict_replicate(template=template, parameters=parameters)
            for dkey in template.keys():
                assert dkey in rep
                numpy.testing.assert_array_equal(rep[dkey].t, template[dkey].t)
                assert calibr8.istensor(rep[dkey].y)
        pass


class TestHDF5storage:
    def _test_save_and_load(self, ds_original):
        # use a temporary directory, because a tempfile.NamedTemporaryFile can not be opened
        # multiple times on all platforms (https://docs.python.org/3/library/tempfile.html#tempfile.NamedTemporaryFile)
        with tempfile.TemporaryDirectory() as dir:
            fp = pathlib.Path(dir, 'testing.h5')
            murefi.save_dataset(ds_original, fp)
            ds_loaded = murefi.load_dataset(fp)

        assert isinstance(ds_loaded, murefi.Dataset)
        assert set(ds_original.keys()) == set(ds_loaded.keys())

        for rid, rep_orig in ds_original.items():
            for dkey, ts_orig in rep_orig.items():
                ts_loaded = ds_loaded[rid][dkey]
                assert isinstance(ts_loaded, murefi.Timeseries)
                assert ts_orig.independent_key == ts_loaded.independent_key
                assert ts_orig.dependent_key == ts_loaded.dependent_key
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
        assert len(ds['R2']) == 0
        self._test_save_and_load(ds)
        return
