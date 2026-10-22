import os,sys
from pathlib import Path
for _thread in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[_thread]='1'
sys.dont_write_bytecode=True
import argparse,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg');import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser(description='Compare four saved actual Test evaluations; no model inference or training')
    p.add_argument('--data-root',type=Path);p.add_argument('--verify',action='store_true');p.add_argument('--smoke-test',action='store_true');a=p.parse_args()
    folders=['GRU','LSTM','Transformer','Compact'];frames=[];states=[];times=[];metadata=[]
    for folder in folders:
        source=HERE.parent/folder/'results';frame=pd.read_csv(source/'overall.csv');state=pd.read_csv(source/'per_state.csv');timing=pd.read_csv(source/'inference_time.csv');info=json.loads((source/'evaluation.json').read_text());frames.append(frame);states.append(state);times.append(timing);metadata.append(info)
    assert len({(m['dataset'],m['horizon_steps'],m['origin_stride'],m['smoke_only']) for m in metadata})==1,'Do not combine incompatible evaluation protocols'
    out=HERE/'results';out.mkdir(exist_ok=True);overall=pd.concat(frames,ignore_index=True);perstate=pd.concat(states,ignore_index=True);latency=pd.concat(times,ignore_index=True)
    overall.to_csv(out/'overall.csv',index=False);perstate.to_csv(out/'per_state.csv',index=False);latency.to_csv(out/'inference_time.csv',index=False)
    summary=overall.merge(latency,on=['model','scope']);summary.to_csv(out/'summary.csv',index=False)
    for label,source,column,ylabel in [('accuracy',overall,'NRMSE_train_std','Macro NRMSE / training std'),('latency',latency,'mean_us','Mean native call time [us]')]:
        fig,axes=plt.subplots(1,2,figsize=(11,4.5))
        for ax,scope in zip(axes,['one_step','recursive_horizon']):
            group=source[source.scope==scope];ax.bar(group.model,group[column]);ax.set(title=scope,ylabel=ylabel)
        fig.tight_layout();fig.savefig(out/(label+'.png'),dpi=150);plt.close(fig)
    fig,ax=plt.subplots()
    for _,row in summary[summary.scope=='recursive_horizon'].iterrows():ax.scatter(row.mean_us,row.NRMSE_train_std);ax.annotate(row.model,(row.mean_us,row.NRMSE_train_std))
    ax.set(xlabel='50-step native mean call time [us]',ylabel='50-step macro NRMSE');fig.tight_layout();fig.savefig(out/'accuracy_computation.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots();g=overall.drop_duplicates('model');bars=ax.bar(g.model,g.parameters);ax.bar_label(bars);ax.set_ylabel('Parameters');fig.tight_layout();fig.savefig(out/'parameters.png',dpi=150);plt.close(fig)
    print('Historically reused Test; one raw step10us,50steps0.5ms; equal recording/state means; original and retrained checkpoints labelled separately.')
    print(summary.to_string(index=False,float_format=lambda v:format(v,'.12g')));print(perstate.to_string(index=False,float_format=lambda v:format(v,'.10g')))
if __name__=='__main__':main()
