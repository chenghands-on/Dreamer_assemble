import argparse
import pandas as pd
from pathlib import Path
import os

def parse_args():
    parser = argparse.ArgumentParser(description="argument parser", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--uri", default='99f26e5dc334468ba20cbaa43361a1e9', type=str)
    parser.add_argument("--tidy", default=False, type=bool,required=False)
    parser.add_argument("--index", default='0', type=str)
    parser.add_argument("--env", default='atari_pong', type=str)
    return parser.parse_args()


def plot_results(df_list, env='atari_pong'):
    import numpy as np
    import holoviews as hv
    """
    Given a list of dataframes, plot results on given environment.
    """
    df = pd.concat(df_list)
    # aggregate runs
    df = df.groupby(['method', 'env', 'env_steps'])['return'].agg(['mean', 'std', 'count']).reset_index()
    df['std'] = df['std'].fillna(0)
    df = df.rename(columns={'mean': 'return', 'std': 'return_std'})
    data = df

    df_env = data[data['env'] == env]
    fig = hv.Overlay([
        hv.Curve(df_method, 'env_steps', 'return', group=env, label=method).opts(
            xlim=(0, 20e6),
            ylim=(-22, 22),
        )
        * 
        hv.Spread(df_method, 'env_steps', ['return', 'return_std']).opts(
            alpha=0.2,
        )
        for method, df_method
        in df_env.groupby('method')
    ]).opts(title=env)

    hv.save(fig, f'figures/{env}.png', dpi=144)
    return fig

def load_data_from_csv(csv_files, method_name, env_steps_interval=1e6):
    """
    Load data from csv files, append a method name column, discretize steps and group by certain fields.
    """
    df = pd.concat([pd.read_csv(f) for f in csv_files])
    df['method'] = method_name
    # discretize to 1e6 steps
    df['env_steps'] = (df['env_steps'] / env_steps_interval).apply(np.ceil) * env_steps_interval  
    df = df.groupby(['env', 'method', 'run', 'env_steps'])[['return']].mean().reset_index()
    return df

def main():
    # 解析命令行参数
    args = parse_args()

    folder_path = f'/home/chenghan/pydreamer/mlruns/0/{args.uri}/metrics'
    print(os.listdir(folder_path))

    columns = ['steps', 'return', 'env_steps']
    merged_data = pd.DataFrame()

    for column in columns:
        file = f'{folder_path}/agent/{column}'
        data = pd.read_csv(file,delimiter=' ')
        merged_data[column] = data.iloc[:, 1]

    file=f'{folder_path}/_loss'
    data=pd.read_csv(file,delimiter=' ')
    merged_data['loss'] = data.iloc[:, 1]

    merged_data['env'] = args.env
    merged_data['run'] = args.index

    name=merged_data['env'][0]
    index=merged_data['run'][0]
    folder_path = f'/home/chenghan/pydreamer/results/figures/{name}'
    os.makedirs(folder_path, exist_ok=True)

    if args.tidy:
        merged_data.to_csv(f'{folder_path}/{index}_tidy.csv', index=False)
    else:
        merged_data.to_csv(f'{folder_path}/{index}_def.csv', index=False)

# 如果当前脚本被当作主程序运行，那么执行 main() 函数
if __name__ == '__main__':
    main()
