from typing import Dict
import tempfile
from pathlib import Path
import numpy as np
import mlflow
import os
from mlflow.tracking import MlflowClient

# FPS = 10
# B, T = 3, 500

def download_artifact_npz(run_id, artifact_path) -> Dict[str, np.ndarray]:
    mlflow.set_tracking_uri('file:///home/chenghan/pydreamer/mlruns')
    client = MlflowClient()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = client.download_artifacts(run_id, artifact_path, tmpdir)
        with Path(path).open('rb') as f:
            data = np.load(f)
            print(data.files)
            # print(data['terminal'])
            # print(data['action_pred'])
            return {k: data[k] for k in data.keys()}  # type: ignore

def encode_gif(frames, fps=10,T=500):
    # Copyright Danijar
    from subprocess import Popen, PIPE
    h, w, c = frames[0].shape
    pxfmt = {1: 'gray', 3: 'rgb24'}[c]
    cmd = ' '.join([
        'ffmpeg -y -f rawvideo -vcodec rawvideo',
        f'-r {fps:.02f} -s {w}x{h} -pix_fmt {pxfmt} -i - -filter_complex',
        '[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse',
        f'-r {fps:.02f} -f gif -'])
    proc = Popen(cmd.split(' '), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    for image in frames:
        proc.stdin.write(image.tobytes())  # type: ignore
    out, err = proc.communicate()
    if proc.returncode:
        raise IOError('\n'.join([' '.join(cmd), err.decode('utf8')]))
    del proc
    return out


def make_gif_wm(env_name,path,index,action_type,fps=10,dream=True):
    T=500
    with Path(path).open('rb') as f:
        tensors = np.load(f,allow_pickle=True)
        print(tensors.files)
        if dream==True:
            img=tensors['image_pred']
        else:
            img=tensors['image']
    # print(img.shape)
    # img=img.transpose(1,0,4,3,2)
    for i in range(img.shape[0]):
        img_cut = img[i:i+1, :T].reshape((-1, 64, 64, 3))
        # print(img_cut.shape)
        gif = encode_gif(img_cut, fps)
        if dream==True:
            folder_path = f'/home/chenghan/pydreamer/wm_results/figures/{env_name}/{action_type}'
            os.makedirs(folder_path, exist_ok=True)
            dest_path = f'{folder_path}/dream_{action_type}_{index}_{i}.gif'
        else:
            folder_path = f'/home/chenghan/pydreamer/wm_results/figures/{env_name}/{action_type}'
            os.makedirs(folder_path, exist_ok=True)
            dest_path = f'{folder_path}/origin_{action_type}_{index}_{i}.gif'
        with Path(dest_path).open('wb') as f:
            f.write(gif)

# path1='/home/chenghan/pydreamer/wm_results/tensors_Atari-Pong_704_data.npz'
# path2='/home/chenghan/pydreamer/wm_results/dream_tensors_Atari-Pong_704_data.npz'
# make_gif_wm('pong',path1,1,dream=False)
# make_gif_wm('pong',path2,1,dream=True)