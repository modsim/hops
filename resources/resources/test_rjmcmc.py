import numpy as np
import scipy.linalg.blas
from scipy.stats import gamma
from scipy.integrate import nquad
import random
from time import time

# There are 4 pVjossible models:
# 1) scale = 1, loc = 0, called model 00, because both parameters are deactivated
# 2) loc = 0, called model 01, because scale parameter is active
# 3) scale = 1, called model 10 because location parameter is active
# 4) shape, scale and loc variable, called model 11 because both parameters are activated

# boundaries:
# shape is [0.1, 10]
# scale is [0.1, 10]
# loc is [0, 0.9]
# 1 measurement at 1

prob_dict = {("shape", "scale", "loc"): 0.18335007, ("shape", "loc"): 0.34859468, ("shape", "scale"): 0.15325739,
             ("shape",): 0.31479786}


def get_par_range_dict():
    return {"shape": [0.1, 10], "scale": [0.1, 10], "loc": [0, 0.9]}


def get_volume(prop_list):
    range_dict = get_par_range_dict()
    vol = 1
    for prop in prop_list:
        vol *= range_dict[prop][1] - range_dict[prop][0]
    return vol


def evaluate_gamma_model_11(shape, scale, loc):
    x = 1
    return gamma.pdf(x, shape, scale=scale, loc=loc)


def evaluate_gamma_model_10(shape, loc):
    x = 1
    return gamma.pdf(x, shape, scale=1, loc=loc)


def evaluate_gamma_model_01(shape, scale):
    x = 1
    return gamma.pdf(x, shape, scale=scale, loc=0)


def evaluate_gamma_model_00(shape):
    x = 1
    return gamma.pdf(x, shape, scale=1, loc=0)


def get_model_probabilities():
    pair_ranges = get_par_range_dict()
    opts = {"epsabs": 1e-4, "epsrel": 1e-4}
    m1 = nquad(evaluate_gamma_model_11, [pair_ranges['shape'], pair_ranges['scale'], pair_ranges['loc']], opts=opts)[0]
    m1 /= get_volume(["shape", "scale", "loc"])
    m2 = nquad(evaluate_gamma_model_10, [pair_ranges['shape'], pair_ranges['loc']], opts=opts)[0]
    m2 /= get_volume(["shape", "loc"])
    m3 = nquad(evaluate_gamma_model_01, [pair_ranges['shape'], pair_ranges['scale']], opts=opts)[0]
    m3 /= get_volume(["shape", "scale"])
    m4 = nquad(evaluate_gamma_model_00, [pair_ranges['shape']], opts=opts)[0]
    m4 /= get_volume(["shape"])
    prob_list = np.array([m4, m3, m2, m1])
    prob_list = prob_list / prob_list.sum()
    prob_list = [('00', prob_list[0]), ('01', prob_list[1]), ('10', prob_list[2]), ('11', prob_list[3])]
    return prob_list


def sample_means(prob_list):
    locationSamples_00 = [0.]
    scaleSamples_00 = [1]
    shapeSamples_00 = [1]

    locationSamples_01 = [0.]
    scaleSamples_01 = [1]
    shapeSamples_01 = [1]

    locationSamples_10 = [0.]
    scaleSamples_10 = [1]
    shapeSamples_10 = [1]

    locationSamples_11 = [0.]
    scaleSamples_11 = [1]
    shapeSamples_11 = [1]

    for i in range(100_000):
        locationProposal = random.uniform(0, 0.9)
        scaleProposal = random.uniform(0.1, 10)
        shapeProposal = random.uniform(0.1, 10)

        ap_00 = gamma.pdf(1, shapeProposal, scale=1, loc=0) / gamma.pdf(1, shapeSamples_00[-1], scale=1, loc=0)
        ap_01 = gamma.pdf(1, shapeProposal, scale=scaleProposal, loc=0) / gamma.pdf(1, shapeSamples_01[-1],
                                                                                    scale=scaleSamples_01[-1], loc=0)
        ap_10 = gamma.pdf(1, shapeProposal, scale=1, loc=locationProposal) / gamma.pdf(1, shapeSamples_10[-1], scale=1,
                                                                                       loc=locationSamples_10[-1])
        ap_11 = gamma.pdf(1, shapeProposal, scale=scaleProposal, loc=locationProposal) / gamma.pdf(1,
                                                                                                   shapeSamples_11[-1],
                                                                                                   scale=
                                                                                                   scaleSamples_11[-1],
                                                                                                   loc=
                                                                                                   locationSamples_11[
                                                                                                       -1])

        chance = random.uniform(0, 1)
        if chance < ap_00:
            locationSamples_00.append(0)
            scaleSamples_00.append(1)
            shapeSamples_00.append(shapeProposal)
        else:
            locationSamples_00.append(0)
            scaleSamples_00.append(1)
            shapeSamples_00.append(shapeSamples_00[-1])

        chance = random.uniform(0, 1)
        if chance < ap_01:
            locationSamples_01.append(0)
            scaleSamples_01.append(scaleProposal)
            shapeSamples_01.append(shapeProposal)
        else:
            locationSamples_01.append(0)
            scaleSamples_01.append(scaleSamples_01[-1])
            shapeSamples_01.append(shapeSamples_01[-1])

        chance = random.uniform(0, 1)
        if chance < ap_10:
            locationSamples_10.append(locationProposal)
            scaleSamples_10.append(1)
            shapeSamples_10.append(shapeProposal)
        else:
            locationSamples_10.append(locationSamples_10[-1])
            scaleSamples_10.append(1)
            shapeSamples_10.append(shapeSamples_10[-1])

        chance = random.uniform(0, 1)
        if chance < ap_11:
            locationSamples_11.append(locationProposal)
            scaleSamples_11.append(scaleProposal)
            shapeSamples_11.append(shapeProposal)
        else:
            locationSamples_11.append(locationSamples_11[-1])
            scaleSamples_11.append(scaleSamples_11[-1])
            shapeSamples_11.append(shapeSamples_11[-1])

    location_mean = prob_list[0][1] * np.mean(locationSamples_00[5000:]) + prob_list[1][1] * np.mean(locationSamples_01[5000:]) + \
                    prob_list[2][1] * np.mean(locationSamples_10[5000:]) + prob_list[3][1] * np.mean(locationSamples_11[5000:])

    scale_mean = prob_list[0][1] * np.mean(scaleSamples_00[5000:]) + prob_list[1][1] * np.mean(scaleSamples_01[5000:]) + \
                 prob_list[2][1] * np.mean(scaleSamples_10[5000:]) + prob_list[3][1] * np.mean(scaleSamples_11[5000:])

    shape_mean = prob_list[0][1] * np.mean(shapeSamples_00[5000:]) + prob_list[1][1] * np.mean(shapeSamples_01[5000:]) + \
                 prob_list[2][1] * np.mean(shapeSamples_10[5000:]) + prob_list[3][1] * np.mean(shapeSamples_11[5000:])

    return location_mean, scale_mean, shape_mean


if __name__ == "__main__":
    start = time()
    model_probabilities = get_model_probabilities()
    for m in model_probabilities:
        print('model ' + str(m[0]) + ' has prob ' + str(m[1]))

    print(model_probabilities)
    mean_location, mean_scale, mean_shape = sample_means(model_probabilities)
    print('mean_location: ' + str(mean_location))
    print('mean_scale: ' + str(mean_scale))
    print('mean_shape: ' + str(mean_shape))
    end = time()
    print('runtime ' + str(end-start))
