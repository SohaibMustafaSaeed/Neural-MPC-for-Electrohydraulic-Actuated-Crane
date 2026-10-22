import os,sys
from pathlib import Path
for _thread in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[_thread]='1'
sys.dont_write_bytecode=True
import argparse,importlib.util,time,json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('gru_trainer',HERE/'train.py');trainer=importlib.util.module_from_spec(spec);sys.modules[spec.name]=trainer;spec.loader.exec_module(trainer)
from preprocess import convert_recording,read_chunks,INPUT_COLS,OUTPUT_COLS,COMMAND_INDICES,STATE_INDICES
SI=np.asarray(STATE_INDICES);CI=np.asarray(COMMAND_INDICES)

class Metrics:
    def __init__(self,std):
        self.std=std;self.n=0;self.mean=np.zeros(22);self.m2=np.zeros(22);self.sse=np.zeros(22);self.sae=np.zeros(22);self.maximum=np.zeros(22);self.persistence=np.zeros(22)
    def add(self,actual,prediction,baseline):
        n=len(actual);mean=actual.mean(axis=0);delta=mean-self.mean
        self.m2+=((actual-mean)**2).sum(axis=0)+delta**2*self.n*n/(self.n+n);self.mean+=delta*n/(self.n+n);self.n+=n
        error=prediction-actual;self.sse+=(error**2).sum(axis=0);self.sae+=abs(error).sum(axis=0);self.maximum=np.maximum(self.maximum,abs(error).max(axis=0));self.persistence+=((baseline-actual)**2).sum(axis=0)
    def table(self):
        var=self.m2/self.n;ratio=np.full(22,np.nan);np.divide(self.sse/self.n,var,out=ratio,where=var>np.finfo(float).eps*np.maximum(1.,self.mean**2+var));rmse=np.sqrt(self.sse/self.n)
        return pd.DataFrame(dict(feature=OUTPUT_COLS,MAE=self.sae/self.n,RMSE=rmse,R2=1-ratio,NRMSE_train_std=rmse/self.std,persistence_RMSE=np.sqrt(self.persistence/self.n)))

def array_for(source,prepared,sc):
    target=prepared/'testing'/(source.stem+'.npy');metadata=target.with_suffix('.json');stat=source.stat()
    if target.exists() and metadata.exists():
        d=json.loads(metadata.read_text())
        if d['source_size']==stat.st_size and d['source_mtime_ns']==stat.st_mtime_ns and d['mean']==sc['mean'] and d['std']==sc['std']:
            return np.load(target,mmap_mode='r')
    rows=sum(len(chunk) for chunk in read_chunks(source,int(sc['startup_rows_removed']),100000))
    convert_recording(source,target.parent,rows,int(sc['startup_rows_removed']),100000,sc)
    return np.load(target,mmap_mode='r')

def prediction_plot(data,indices,path,title,lead='prediction'):
    fig,axes=plt.subplots(len(indices),1,figsize=(12,2.2*len(indices)),sharex=True,squeeze=False)
    for ax,j in zip(axes.flat,indices):
        scale=1e-5 if '[Pa]' in OUTPUT_COLS[j] else 1;ax.plot(data['time'] if lead=='prediction' else data['time']-49e-5,data['actual' if lead=='prediction' else 'actual_one'][:,j]*scale,'k',label='Recorded Test43');ax.plot(data['time'] if lead=='prediction' else data['time']-49e-5,data[lead][:,j]*scale,label='gru',lw=.8);ax.set_ylabel(OUTPUT_COLS[j].replace('[Pa]','[bar]'));ax.grid(alpha=.2)
    axes.flat[0].legend();axes.flat[-1].set_xlabel('Source CSV time [s]');fig.suptitle(title);fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)

def main():
    p=argparse.ArgumentParser(description='Frozen or explicitly selected gru checkpoint; all11historically reused Test recordings')
    p.add_argument('--data-root',type=Path,default=Path(os.environ.get('PATU_DATA_ROOT',HERE.parents[2])))
    p.add_argument('--prepared-root',type=Path);p.add_argument('--checkpoint',default='original');p.add_argument('--output-dir',type=Path);p.add_argument('--verify',action='store_true');p.add_argument('--smoke-test',action='store_true')
    a=p.parse_args();path=HERE/'checkpoints/gru_original.pt' if a.checkpoint=='original' else Path(a.checkpoint).resolve()
    model,ck=trainer.load_bundle(path);parameters=sum(v.numel() for v in model.parameters());assert parameters==dict(gru=4098,lstm=4946,transformer=4710,compact=4246)['gru']
    sc=ck['scaler'];mean=np.asarray(sc['mean']);std=np.asarray(sc['std']);weights=trainer.export_weights(model)
    prepared=a.prepared_root or a.data_root/'PATU_prepared_cache';out=a.output_dir or (HERE/'results' if a.checkpoint=='original' and not a.smoke_test else HERE/'results'/path.stem/('smoke' if a.smoke_test else 'test'))
    out.mkdir(parents=True,exist_ok=True);details=[];latency=[];plot={}
    for number,k in enumerate(range(43,54)):
        source=a.data_root/'Test'/f'Simulation_Output_{k}.csv';x=array_for(source,prepared,sc)
        origins=np.unique(np.r_[np.arange(1,len(x)-50,10000),len(x)-51]);origins=origins[:2] if a.smoke_test else origins
        meters={s:Metrics(std[SI]) for s in ['one_step','recursive_horizon']};selected=[];actual=[];first=[];first_truth=[];times=[]
        for t in origins:
            previous=x[t-1].copy();current=x[t].copy();memory=trainer.native_encode(current,weights[0],weights[1]);rows=np.zeros((50,24),np.float32);rows[:,CI]=x[t:t+50][:,CI]
            prediction,*_=trainer.native_chunk(rows,previous,current,memory,True,mean,std,*weights)
            target=x[t+1:t+51][:,SI].astype(float)*std[SI]+mean[SI];anchor=current[SI].astype(float)*std[SI]+mean[SI]
            meters['one_step'].add(target[:1],prediction[:1],anchor[None]);meters['recursive_horizon'].add(target,prediction,np.tile(anchor,(50,1)))
            if number==0:selected.append(prediction[-1]);actual.append(target[-1]);first.append(prediction[0]);first_truth.append(target[0]);times.append(18+(t+50)*1e-5)
        for scope,meter in meters.items():
            frame=meter.table();frame.insert(0,'scope',scope);frame.insert(0,'recording',source.name);frame.insert(0,'model','gru');frame['origins']=len(origins);details.append(frame)
        if number==0:
            plot=dict(time=np.asarray(times),prediction=np.asarray(selected),actual=np.asarray(actual),one=np.asarray(first),actual_one=np.asarray(first_truth))
            for h,scope in [(1,'one_step'),(50,'recursive_horizon')]:
                rows=np.zeros((h,24),np.float32);rows[:,CI]=x[1:h+1][:,CI];previous=x[0].copy();current=x[1].copy();memory=trainer.native_encode(current,weights[0],weights[1])
                def call():return trainer.native_chunk(rows,previous,current,memory,True,mean,std,*weights)
                for _ in range(5):call()
                values=[]
                for _ in range(200):
                    tick=time.perf_counter();call();values.append((time.perf_counter()-tick)*1e6)
                latency.append(dict(model='gru',scope=scope,mean_us=np.mean(values),median_us=np.median(values),p95_us=np.percentile(values,95),p99_us=np.percentile(values,99),max_us=np.max(values)))
        print('gru',source.name,len(origins),'origins',flush=True)
    detail=pd.concat(details,ignore_index=True);detail.to_csv(out/'per_recording_per_state.csv',index=False)
    average=detail.groupby(['model','scope','feature'])[['MAE','RMSE','R2','NRMSE_train_std','persistence_RMSE']].mean().reset_index();average.assign(_order=average.feature.map({n:i for i,n in enumerate(OUTPUT_COLS)})).sort_values(['scope','_order']).drop(columns='_order').to_csv(out/'per_state.csv',index=False)
    overall=average.groupby(['model','scope'])[['NRMSE_train_std','R2']].mean().reset_index();overall['parameters']=parameters;overall.to_csv(out/'overall.csv',index=False)
    detail.groupby(['model','recording','scope'])[['NRMSE_train_std','R2']].mean().reset_index().to_csv(out/'per_recording.csv',index=False);pd.DataFrame(latency).to_csv(out/'inference_time.csv',index=False)
    np.savez_compressed(out/'predictions_Test43.npz',**plot)
    mech=[OUTPUT_COLS.index(n) for n in ['thetalift[rad]','thetajib[rad]','thetaliftdot[rad/s]','thetajibdot[rad/s]']];pressure=[j for j,n in enumerate(OUTPUT_COLS) if '[Pa]' in n]
    prediction_plot(plot,mech,out/'one_step_prediction.png','One step =10us; full retained Test43 span','one')
    prediction_plot(plot,mech,out/'recursive_prediction.png','Last lead of50raw steps =0.5ms; full retained Test43 span')
    prediction_plot(plot,mech,out/'mechanical_states.png','Mechanical states; 50-step recursive last lead')
    prediction_plot(plot,pressure,out/'hydraulic_states.png','Hydraulic states; 50-step recursive last lead')
    metadata=dict(model='gru',checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),checkpoint=path.name,parameters=parameters,dataset='11 historically reused Test recordings',smoke_only=a.smoke_test,horizon_steps=50,horizon_s=.0005,origin_stride=10000,aggregation='Equal recording metrics per state, then equal state mean; undefined R2 omitted',timing_scope='native_chunk only; initial encoding/buffers/JIT/I-O/metrics excluded')
    (out/'evaluation.json').write_text(json.dumps(metadata,indent=2));print(overall.to_string(index=False,float_format=lambda v:format(v,'.14g')));print(average.to_string(index=False,float_format=lambda v:format(v,'.10g')));print(pd.DataFrame(latency).to_string(index=False))
    if a.checkpoint=='original' and not a.smoke_test:
        refs=dict(gru=(5.4196491933785e-6,.00015035229186523),lstm=(5.4138400165737e-6,.00015311336744197),transformer=(5.7867260717943e-6,.00015981830658744),compact=(5.3349289219539e-6,.00014975278816917))
        np.testing.assert_allclose(overall.set_index('scope').loc[['one_step','recursive_horizon'],'NRMSE_train_std'],refs['gru'],rtol=1e-10,atol=1e-15)
        print('Frozen original Test regression PASSED; no retraining.')
if __name__=='__main__':main()
