# Dreamer-ensemble
An ensemble of various world models including dreamer v2 and v3
Reimplementation of [DreamerV2](https://danijar.com/project/dreamerv2/) model-based RL algorithm in PyTorch. 

The official DreamerV2 implementation [can be found here](https://danijar.com/project/dreamerv2/).

This is a research project with no guarantees of stability and support. Breaking changes to be expected!

## Quick start 
If you want to use dreamer V2 for atari, try

```python launch.py --config defaults_wis_v3 atari atari_pong --generator_prefill_steps 1000 --wm_type v2```
Also, for dreamer V3

```python launch.py --config defaults_wis_v3 atari atari_pong --generator_prefill_steps 1000 --wm_type v3```

## Mlflow Tracking

PyDreamer relies quite heavily on [Mlflow](https://www.mlflow.org/docs/latest/tracking.html) tracking to log metrics, images, store model checkpoints and even replay buffer. 

That does NOT mean you need to have a Mlflow tracking server installed. By default, `mlflow` is just a pip package, which stores all metrics and files locally under `./mlruns` directory.

That said, if you are running experiments on the cloud, it is *very* convenient to set up a persistent Mlflow [tracking server](https://www.mlflow.org/docs/latest/tracking.html#mlflow-tracking-servers). In that case just set the `MLFLOW_TRACKING_URI` env variable, and all the metrics will be sent to the server instead of written to the local dir.

Note that the replay buffer is just a directory with mlflow artifacts in `*.npz` format, so if you set up an S3 or GCS mlflow artifact store, the replay buffer will be actually stored on the cloud and replayed from there! This makes it easy to persist data across container restarts, but be careful to store data in the same cloud region as the training containers, to avoid data transfer charges.
