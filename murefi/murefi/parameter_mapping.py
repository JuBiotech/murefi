import pandas
import pathlib
import numpy

class ParameterMapping(object):
    def __init__(self, path: pathlib.Path):
        """ Function to map all the parameters provided by the user as csv file
        to dictionaries and arrays with the user input, the extracted fitting parameters
        and the bounds.
        The column names (first row) of the csv files are fixed to 'Parameter', 'S_0', 'X_0', 'mue_max', 'K_S',
        'Y_XS', 't_lag' and 't_acc'. The user can either type in a number if the value should be
        fixed or an arbitrary name if the parameter should be fitted.
        In the 'Parameter' column, 'lower_bound' and 'upper_bound' specify the bounds for the fit.
        The other entries in the columns specify the wells the user wishes to use for the parameter estimation (format: 'A01').
                
        Return:
        self.fitpars_bounds:  dictionary with only fitting parameters as keys and tuples (lower bound, upper bound)
                              as value
        self.parameters_dic:  dictionary with the values or names for all parameters provided by the user
                              (floats and strings)
        self.bounds_list:     list of tuples with (lower bound, upper bound) for each fitting paramter
        self.fitpars_array:   numpy array with only the parameters for fitting
        """
        
        parameters = pandas.read_csv(path, delimiter = ';')
        parameters = parameters.set_index('Parameter')
        parameters = parameters.T
        
        fitpars_bounds = {}
        for name, column in parameters.iteritems():
            for i, element in enumerate(column):
                try:
                    float(element)
                except ValueError:
                    fitpars_bounds[element] = (parameters.lower_bound[i], parameters.upper_bound[i])
        
        bounds_list = [
            fitpars_bounds[parameter]
            for parameter in fitpars_bounds.keys()
        ]
        
        parameters_dic = {
            column : parameters[column].tolist()
            for column in parameters.columns 
            if not 'bound' in column
        }
        
        bounds_dic = {
            column: parameters[column].tolist()
            for column in parameters.columns
            if 'bound' in column
        }
        
        fitpars_list = [
            parameter
            for parameter in fitpars_bounds.keys()
        ]
            
        self.fitpars_bounds = fitpars_bounds
        self.parameters_dic = parameters_dic
        self.bounds_list = bounds_list
        self.fitpars_array = numpy.array(fitpars_list)