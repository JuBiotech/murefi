import collections
import unittest
import numpy
import pandas
import pathlib
import scipy.stats as stats

import calibr8


dir_testfiles = pathlib.Path(pathlib.Path(__file__).absolute().parent, 'testfiles')
       

class ErrorModelTest(unittest.TestCase):
    def test_init(self):
        independent = 'X'
        dependent = 'BS'
        key = 'X'
        errormodel = calibr8.ErrorModel(independent, dependent, key)
        self.assertEqual(errormodel.independent, independent)
        self.assertEqual(errormodel.dependent, dependent)
        self.assertEqual(errormodel.key, key)
        self.assertEqual(errormodel.theta_fitted, None)
    
    def test_exceptions(self):
        independent = 'X'
        dependent = 'BS'
        key = 'X'
        y_hat = numpy.array([1,2,3])
        y_obs = numpy.array([4,5,6])
        errormodel = calibr8.ErrorModel(independent, dependent, key)
        with self.assertRaises(NotImplementedError):
            _ = errormodel.predict_dependent(y_hat)
        with self.assertRaises(NotImplementedError):
            _ = errormodel.predict_independent(y_hat)
        with self.assertRaises(NotImplementedError):
            _ = errormodel.infer_independent(y_obs)
        with self.assertRaises(NotImplementedError):
            _ = errormodel.loglikelihood(y_obs=y_obs, y_hat=y_hat, theta=[1,2,3])
        with self.assertRaises(NotImplementedError):
            _ = errormodel.fit(independent=y_hat, dependent=y_obs, theta_guessed=None)
        return
    
    def test_fit(self):
        independent = 'X'
        dependent = 'BS'
        key = 'X'
        y_hat = numpy.array([1,2,3])
        y_obs = numpy.array([4,5,6])
        theta_guessed= [0]
        errormodel = calibr8.ErrorModel(independent, dependent, key)
        with self.assertRaises(TypeError):
            _ = errormodel.fit(y_hat, y_obs, theta_guessed)


class GlucoseErrorModelTest(unittest.TestCase):
    def test_predict_dependent(self):
        independent = 'Glu'
        dependent = 'OD'
        key = 'S'
        y_hat = numpy.array([1,2,3])
        theta = [0,0,0]
        errormodel = calibr8.GlucoseErrorModel(independent, dependent, key)
        errormodel.theta_fitted = [0,1,0.1]
        with self.assertRaises(TypeError):
            _ = errormodel.predict_dependent(y_hat, theta)
        mu, sigma, df = errormodel.predict_dependent(y_hat)
        self.assertTrue(numpy.array_equal(mu, numpy.array([1,2,3])))
        self.assertTrue(numpy.array_equal(sigma, numpy.array([0.1,0.1,0.1])))
        self.assertEqual(df, 1)
        return
    
    def test_predict_independent(self):
        errormodel = calibr8.GlucoseErrorModel('Glu', 'OD', 'S')
        errormodel.theta_fitted = [0, 2, 0.1]
        
        x_original = numpy.array([4, 5, 6])
        mu, sd, df = errormodel.predict_dependent(x_original)
        x_predicted = errormodel.predict_independent(y_obs=mu)
        
        self.assertTrue(numpy.array_equal(mu, [8, 10, 12]))
        self.assertTrue(numpy.array_equal(sd, [0.1, 0.1, 0.1]))
        self.assertTrue(numpy.allclose(x_predicted, x_original))
        return

    def test_loglikelihood(self):
        independent = 'Glu'
        dependent = 'OD'
        key = 'S'
        y_hat = numpy.array([1,2,3])
        y_obs = numpy.array([1,2,3])
        errormodel = calibr8.GlucoseErrorModel(independent, dependent, key)
        errormodel.theta_fitted = [0,1,0.1]
        with self.assertRaises(TypeError):
            _ = errormodel.loglikelihood(y_obs, y_hat=y_hat)
        true = errormodel.loglikelihood(y_obs=y_obs, y_hat=y_hat)
        mu, sigma, df = errormodel.predict_dependent(y_hat, theta=errormodel.theta_fitted)
        expected = numpy.sum(numpy.log(stats.t.pdf(x=y_obs, loc=mu, scale=sigma, df=1)))
        self.assertEqual(expected, true)
        true = errormodel.loglikelihood(y_obs=y_obs, y_hat=y_hat)
        mu, sigma, df = errormodel.predict_dependent(y_hat, theta=errormodel.theta_fitted)
        expected = numpy.sum(numpy.log(stats.t.pdf(x=y_obs, loc=mu, scale=sigma, df=1)))
        self.assertEqual(expected, true)
        return
    
    def test_loglikelihood_without_fit(self):
        independent = 'Glu'
        dependent = 'OD'
        key = 'S'
        y_hat = numpy.array([1,2,3])
        y_obs = numpy.array([1,2,3])
        errormodel = calibr8.GlucoseErrorModel(independent, dependent, key)
        with self.assertRaises(Exception):
            _= errormodel.loglikelihood(y_obs=y_obs, y_hat=y_hat)
        return


class BiomassErrorModelTest(unittest.TestCase):
    def test_predict_dependent(self):
        independent = 'BTM'
        dependent = 'BS'
        key = 'X'
        y_hat = numpy.array([1,10])
        errormodel = calibr8.BiomassErrorModel(independent, dependent, key)
        errormodel.theta_fitted = numpy.array([5,5,10,0.5,1,0])
        theta = numpy.array([0,0,0,0,0])
        with self.assertRaises(TypeError):
            _ = errormodel.predict_dependent(y_hat, theta)
        mu, sigma, df = errormodel.predict_dependent(y_hat)
        expected = numpy.exp(2*5-10+(2*(10-5))/(1+numpy.exp(-2*0.5/(10-5)*(numpy.log(y_hat)-5))))
        self.assertTrue(numpy.allclose(mu, expected))
        self.assertTrue(numpy.allclose(sigma, numpy.array([1,1])))
        self.assertEqual(df, 1)
        return
    
    def test_predict_independent(self):
        errormodel = calibr8.BiomassErrorModel('BTM', 'BS', 'X')
        errormodel.theta_fitted = numpy.array([5, 5, 10, 0.5, 0, 1])
        
        x_original = numpy.linspace(0.01, 30, 20)
        mu, sd, df = errormodel.predict_dependent(x_original)
        x_predicted = errormodel.predict_independent(y_obs=mu)
        
        self.assertTrue(numpy.allclose(x_predicted, x_original))
        return

    def test_loglikelihood(self):
        independent = 'BTM'
        dependent = 'BS'
        key = 'X'
        y_hat = numpy.array([1,2,3])
        y_obs = numpy.array([1,2,3])
        errormodel = calibr8.BiomassErrorModel(independent, dependent, key)
        errormodel.theta_fitted = numpy.array([5,5,10,0.5,0,1])
        with self.assertRaises(TypeError):
            _ = errormodel.loglikelihood(y_obs, y_hat=y_hat, theta = errormodel.theta_fitted)
        theta = errormodel.theta_fitted
        true = errormodel.loglikelihood(y_obs=y_obs, y_hat=y_hat, theta=theta)
        mu, sigma, df = errormodel.predict_dependent(y_hat, theta=theta)
        expected = numpy.sum(numpy.log(stats.t.pdf(x=y_obs, loc=mu, scale=sigma, df=1)))
        self.assertEqual(expected, true)
        true = errormodel.loglikelihood(y_obs=y_obs, y_hat=y_hat)
        mu, sigma, df = errormodel.predict_dependent(y_hat, theta=theta)
        expected = numpy.sum(numpy.log(stats.t.pdf(x=y_obs, loc=mu, scale=sigma, df=1)))
        self.assertEqual(expected, true)
        return
    
    def test_loglikelihood_without_fit(self):
        independent = 'BTM'
        dependent = 'BS'
        key = 'X'
        y_hat = numpy.array([1,2,3])
        y_obs = numpy.array([1,2,3])
        errormodel = calibr8.BiomassErrorModel(independent, dependent, key)
        with self.assertRaises(Exception):
            _= errormodel.loglikelihood(y_obs=y_obs, y_hat=y_hat)
        return


if __name__ == '__main__':
    unittest.main(exit=False)