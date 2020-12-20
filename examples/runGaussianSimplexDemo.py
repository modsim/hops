import subprocess
import multiprocessing


def run(params):
    print('running params')
    print(params)
    num_samples = '10000'
    num_thinning = '10'
    subprocess.check_output((
        ["./../cmake-build-release/examples/SamplingGaussianSimplexDemo",
         params['dims'],
         params['mean'],
         params['cov'],
         num_samples,
         num_thinning,
         params['algo'].split(" ")[0],
         params['algo'].split(" ")[1],
         ]
    ))


def main():
    dims = [32, 128]
    cov_scales = [0.001, 0.1, 0.5]
    means = ['corner', 'chebyshev']
    algos = [
             "CSmMALANoGradient 0",
             "CSmMALANoGradient 0.01",
             "CSmMALANoGradient 0.25",
             "CSmMALANoGradient 0.5",
             "CSmMALANoGradient 0.75",
             "CSmMALANoGradient 0.99",
             "CSmMALANoGradient 1",
        ]
    parameters = []
    for d in dims:
        for m in means:
            for c in cov_scales:
                for a in algos:
                    parameters.append({
                        'dims': str(d),
                        'mean': str(m),
                        'cov': str(c),
                        'algo': str(a),
                    })

    num_procs = 7
    pool = multiprocessing.Pool(num_procs)
    pool.map(run, parameters)


if __name__ == "__main__":
    main()
