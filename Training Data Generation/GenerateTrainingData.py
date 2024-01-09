
from SampleSTORM import sampleStorm
from RiskFactors import averageLandfallsPerMonth

def generateTrainingData(total_years):
    '''
    Generate ML training data

    The ML inputs, X, are randomly generated inputs to the STORM model
    The output targets, Y, are grids of average landfall per month

    NOTE: need to add input parameter generation to this,
    currently just to test production of the output targets from observed data

    :param total_years: number of years to run the STORM simulations over
    :return:
    '''
    input_parameters = None # replace with generated parameters

    tc_data = sampleStorm(total_years)

    avgLandfallsPerMonth = averageLandfallsPerMonth(tc_data, 'EP', total_years, 1)

    return input_parameters, avgLandfallsPerMonth

