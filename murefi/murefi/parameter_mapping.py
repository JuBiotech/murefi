import numpy
import pandas
import pathlib


class ParameterMapping(object):
    def __init__(self, parameters: pandas.DataFrame):
        """ Class to map process parameters and bounds from a csv file.
        
        The purpose of this class is to map all the parameters provided by the user as a csv file
        to dictionaries and arrays with the user input, the extracted fitting parameters
        and the bounds.
        The column names (first row) of the csv files must be 'Parameter', 'S_0', 'X_0', 'mue_max', 'K_S',
        'Y_XS', 't_lag' and 't_acc' to allow for the correct readout. For each parameter and well, the user 
        can either type in a number if the value should be fixed or an arbitrary name if the parameter should
        be fitted. In the column named 'Parameter', 'lower_bound' and 'upper_bound' specify the bounds for the fit.
        The other entries in the columns specify the wells the user wishes to use for the parameter estimation (format: 'A01').
           
        Args:
            parameters (pandas.DataFrame): dataframe of parameter settings.

        self.fitpars_bounds:  dictionary with only fitting parameters as keys and tuples (lower bound, upper bound)
                                as value
        self.parameters_dic:  dictionary with the values or names for all parameters provided by the user
                                (floats and strings)
        self.bounds_list:     list of tuples with (lower bound, upper bound) for each fitting paramter
        self.fitpars_array:   numpy array with only the parameters for fitting
        """
        
        if not 'Parameter' in parameters.columns:
            raise ValueError('The parameter mapping CSV must call its first column "Parameter"')
        parameters = parameters.set_index('Parameter')
        parameters = parameters.T
        
        fitpars_bounds = {}
        for name, column in parameters.iteritems():
            for i, element in enumerate(column):
                try:
                    float(element)
                except ValueError:
                    fitpars_bounds[element] = (numpy.float(parameters.lower_bound[i]), numpy.float(parameters.upper_bound[i]))
        
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

    def repmap(self, theta_full):
        """Remaps a full parameter vector to a dictionary of replicate-wise parameter vectors.

        Args:
            theta_full (array): full parameter vector

        Returns:
            theta_dict (dict): dictionary of replicate-wise parameter vectors
        """
        pname_to_pvalue = {
            pname : pvalue
            for pname, pvalue in zip(self.fitpars_array, theta_full)
        }
        theta_dict = {
            rkey : [
                pname_to_pvalue[pname] if isinstance(pname, str) else pname
                for pname in pnames
            ]
            for rkey, pnames in self.parameters_dic.items()
        }
        return theta_dict
