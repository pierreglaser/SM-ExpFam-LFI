import itertools
import shutil
from pathlib import Path

from smnle import smnle

TASKS = (
    "two_moons",
    "gaussian_linear_uniform",
    "slcp",
    "lotka_volterra",
)

# talk about neural net architecture

OBSERVATIONS = list(range(1, 11))

LRS = [0.01, 0.005, 0.001]

if __name__ == "__main__":


    close_after_simulation = True

    Path("dask-simulation-logs").mkdir(exist_ok=True)

    if shutil.which("srun"):
        from dask_jobqueue import SLURMCluster

        cluster = SLURMCluster(  # type: ignore
            n_workers=0,  # create the workers "lazily" (upon cluster.scale)
            memory="4GB",  # amount of RAM per worker
            processes=1,  # number of execution units per worker (threads and processes)
            cores=1,  # among those execution units, number of processes
            job_extra=[
                # '--export=ALL', # default behavior.
                "--output=dask-simulation-logs/R-%x.%j.out",
                "--error=dask-simulation-logs/R-%x.%j.err",
                *(),
                # '--export=OMP_NUM_THREADS=1,MKL_NUM_THREADS=1',
            ],
            # extra = ["--no-nanny"],
            # scheduler_options={'dashboard_address': ':8787', 'port': 45987, 'allowed_failures': 10},
            scheduler_options={
                "dashboard_address": ":8787",
                "allowed_failures": 10,
            },
            job_cpu=8,
            walltime="24:0:0",
        )
        print(cluster.job_script())
        cluster.scale(50)
    else:
        from dask.distributed import LocalCluster
        cluster = LocalCluster(n_workers=4)

    from distributed import Client

    print('testing cluster')
    client = Client(cluster)
    print(client.submit(lambda x: x + 1, 10).result())
    print('cluster is working')

    futures = []
    args = []

    grid = list(
        itertools.product(
            TASKS,
            OBSERVATIONS,
            LRS
        )
    )

    for arg in grid:
        task, num_observation, lr = arg
        fut = client.submit(
            smnle,
            model=task,
            num_observation=num_observation,
            SM_lr=lr,
            SM_lr_theta=lr,
            epochs=500,
            mcmc_num_chains=50,
            mcmc_num_warmup_steps=1000,
            num_posterior_samples=10000,
        )
        futures.append(fut)
        args.append(arg)

    # wait till all dask futures are done
    client.gather(futures)

    results = [fut.result() for fut in futures]
    print(dict(zip(args, results)))

    with open("results.pkl", "wb") as f:
        import pickle
        pickle.dump(zip(args, results), f)

    client.shutdown()
    cluster.close()

    import time
    time.sleep(10)
