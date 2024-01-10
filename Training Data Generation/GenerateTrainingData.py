
from SampleSTORM import sampleStorm
from RiskFactors import averageLandfallsPerMonth
from GenerateInputParameters import generateInputParameters
import numpy as np
from matplotlib import pyplot as plt


monthsall=[[6,7,8,9,10,11],[6,7,8,9,10,11],[4,5,6,9,10,11],[1,2,3,4,11,12],[1,2,3,4,11,12],[5,6,7,8,9,10,11]]

def generateTrainingData(total_years, basin='EP'):
    '''
    Generate ML training data

    The ML inputs, X, are randomly generated inputs to the STORM model
    The output targets, Y, are grids of average landfall per month

    :param total_years: number of years to run the STORM simulations over
    :return:
    '''

    basins = ['EP', 'NA', 'NI', 'SI', 'SP', 'WP']

    monthlist = monthsall[basins.index(basin)]

    genesis_matrices, movement_coefficients = generateInputParameters(basin, monthlist) # replace with generated parameters

    tc_data = sampleStorm(total_years, genesis_matrices, movement_coefficients)

    avg_landfalls_per_month = averageLandfallsPerMonth(tc_data, 'EP', total_years, 1)

    ## split up input, output data for each month
    inputs = [(genesis_matrices[month], movement_coefficients) for month in monthlist]
    outputs = [avg_landfalls_per_month[:,:,month] for month in monthlist]

    return inputs, outputs