import os,sys
from pathlib import Path
for _thread in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[_thread]='1'
sys.dont_write_bytecode=True
import argparse,time,json,hashlib,importlib.util
import numpy as np,pandas as pd,torch
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'MPC'));import mpc
def context():return mpc.context()
def feature(*args):return mpc.feature(*args)

def generate(ctx,dataset,destination,smoke=False):
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'protocol.json').write_text(json.dumps(dict(dynamics_sha256=mpc.EXPECTED,settings=ctx['cfg'],warm_start='historical previous second move repeated twice',feature_dimension=210),indent=2))
    entries=mpc.inventory('all-train')
    if smoke:entries=[next(s for s in entries if s['controller_split']==split) for split in ['training','validation']]
    for spec in entries:
        dest=destination/(Path(spec['recording']).stem+'.npz')
        if dest.exists():continue
        record=dataset.record(spec,ctx);plant=mpc.NeuralPlant(ctx['base'],record['previous'],record['current'],True);previous=record['previous_command'].copy();guess=np.tile(previous,(2,1));data=[]
        # Original distill400 rollout convention: unshifted previous second move repeated twice.
        for k in range(min(2,spec['matched_updates']) if smoke else spec['matched_updates']):
            ref=record['reference'][k*10:k*10+41];pa,cu=plant.previous.copy(),plant.current.copy();warm=guess.copy()
            obj=mpc.make_objective(ctx,pa,cu,previous,ref[1:],40);res,moves,seconds=mpc.solve(obj,guess,dict(ctx['cfg'],horizon_steps=40));command=moves[0].copy();guess=np.repeat(moves[1:],2,axis=0)
            generated,_=plant.advance(np.tile(command,(10,1)));f=feature(ctx['base'],pa,cu,previous,ref)
            data.append(dict(x=f,previous_input=pa,current_input=cu,previous_command=previous.copy(),warm=warm,plan=moves.copy(),target=command.copy(),valid=bool(np.isfinite(obj.best_cost) and obj.best_cost<1e12 and np.isfinite(f).all()),reference=ref[:,mpc.MECH].copy()))
            previous=command.copy()
        np.savez_compressed(dest,**{key:np.asarray([r[key] for r in data]) for key in data[0]})
        (dest.with_suffix('.json')).write_text(json.dumps(dict(recording=spec['recording'],split=spec['controller_split'],samples=len(data),teacher='Original400ms maxiter5, two100/300ms moves, historical shift',dynamics_sha256=mpc.EXPECTED),indent=2))
        print('TEACHER',spec['recording'],spec['controller_split'],len(data),'samples',flush=True)
    return entries

def pool(folder,smoke=False):
    ctx=mpc.context();protocol_path=folder/'protocol.json' if (folder/'protocol.json').exists() else folder.parent/'protocol.json'
    if not protocol_path.exists():raise FileNotFoundError('Teacher protocol required to verify historical settings: '+str(protocol_path))
    protocol=json.loads(protocol_path.read_text());assert protocol['dynamics_sha256']==mpc.EXPECTED
    for key in ['horizon_steps','maxiter','control_interval_s','prediction_interval_s','cost_weights','position_scale_m']:assert protocol['settings'][key]==ctx['cfg'][key],key
    parts=[]
    for spec in mpc.inventory('all-train'):
        path=folder/(Path(spec['recording']).stem+'.npz')
        if not path.exists():
            if smoke:continue
            raise FileNotFoundError('Missing teacher demonstration: '+str(path))
        with np.load(path) as z:
            part={k:z[k].copy() for k in ['x','warm','plan','target','valid']};n=len(part['x']);
            np.testing.assert_array_equal(z['warm'][1:],np.repeat(z['plan'][:-1,1:,:],2,axis=1))
            ref=np.zeros((41,22));ref[:,mpc.MECH]=z['reference'][0]
            np.testing.assert_array_equal(z['x'][0],feature(ctx['base'],z['previous_input'][0],z['current_input'][0],z['previous_command'][0],ref))
            part.update(source=np.full(n,spec['recording']),split=np.full(n,spec['controller_split']))
        assert part['x'].shape==(n,210) and part['target'].shape==(n,2) and part['plan'].shape==(n,2,2)
        assert np.isfinite(part['x']).all() and np.isfinite(part['target']).all()
        parts.append(part)
    return {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}

def train(ctx,p,dest,seed,epochs=240,patience=40):
    x=p['x'];y=p['target'];tr=(p['split']=='training')&p['valid'];va=(p['split']=='validation')&p['valid']
    mean=x[tr].astype(float).mean(0).astype(np.float32);std=x[tr].astype(float).std(0).astype(np.float32);std[std<1e-6]=1
    center=(sum(ctx['bounds'])/2).astype(np.float32);half=((ctx['bounds'][1]-ctx['bounds'][0])/2).astype(np.float32)
    tx=torch.from_numpy((x[tr]-mean)/std);ty=torch.from_numpy(((y[tr]-center)/half).astype(np.float32));vx=torch.from_numpy((x[va]-mean)/std);vy=torch.from_numpy(((y[va]-center)/half).astype(np.float32))
    torch.manual_seed(seed);rng=np.random.default_rng(seed);model=mpc.MLP(210,128,64,2);assert sum(v.numel() for v in model.parameters())==35394
    opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-5);best=np.inf;stale=0;history=[];started=time.perf_counter()
    print('NEURAL TRAINING FROM SCRATCH:210->128->64->2;35394parameters; demonstrations=',len(x),'train=',int(tr.sum()),'validation=',int(va.sum()),'Adam lr=.001 wd=1e-5; normalized command MSE; batch512; gradclip5; patience40; epochs=',epochs,flush=True)
    for epoch in range(epochs):
        for b in np.array_split(rng.permutation(len(tx)),int(np.ceil(len(tx)/512))):
            opt.zero_grad();loss=((model(tx[b])-ty[b])**2).mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
        with torch.no_grad():tl=float(((model(tx)-ty)**2).mean());vl=float(((model(vx)-vy)**2).mean())
        history.append(dict(epoch=epoch+1,training_MSE=tl,validation_MSE=vl))
        if vl<best-1e-7:best=vl;weights={k:v.detach().clone() for k,v in model.state_dict().items()};stale=0
        else:stale+=1
        print('Epoch',epoch+1,'training=',tl,'validation=',vl,'best=',best,flush=True)
        if stale>=patience:break
    model.load_state_dict(weights)
    with torch.no_grad():pred=model(vx).numpy()*half+center
    error=pred-y[va];target=y[va];moving=abs(target)>.05
    summary=dict(name=dest.stem,seed=seed,dimension=210,width=128,second=64,outputs=2,parameters=35394,validation_MSE=best,epochs=len(history),lift_MAE_V=float(abs(error[:,0]).mean()),jib_MAE_V=float(abs(error[:,1]).mean()),lift_RMSE_V=float(np.sqrt(np.mean(error[:,0]**2))),jib_RMSE_V=float(np.sqrt(np.mean(error[:,1]**2))),wrong_sign_percent=float(((pred*target<0)&moving).sum()/max(moving.sum(),1)*100),training_seconds=time.perf_counter()-started)
    torch.save(dict(weights=weights,mean=mean,std=std,center=center,half=half,**summary),dest)
    result=HERE/'results'/dest.stem;result.mkdir(parents=True,exist_ok=True);pd.DataFrame(history).to_csv(result/'training_loss.csv',index=False)
    (result/'training_configuration.json').write_text(json.dumps(dict(**summary,learning_rate=.001,weight_decay=1e-5,batch_size=512,patience=patience,gradient_clip=5.,feature_precision='historical float32 cast before scaler application',training_recordings=sorted(set(p['source'][tr])),validation_recordings=sorted(set(p['source'][va]))),indent=2))
    dest.with_suffix('.scaler.json').write_text(json.dumps(dict(mean=mean.tolist(),std=std.tolist(),center=center.tolist(),half=half.tolist()),indent=2))
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots();f=pd.DataFrame(history);ax.plot(f.epoch,f.training_MSE,label='Training');ax.plot(f.epoch,f.validation_MSE,label='Validation');ax.set(xlabel='Epoch',ylabel='Normalized command MSE',yscale='log');ax.legend();fig.tight_layout();fig.savefig(result/'training_loss.png',dpi=150);plt.close(fig)
    policy=mpc.Policy(dest);assert np.isfinite(policy(x[va][0])).all();print('SELECTED CHECKPOINT',dest,'reloaded successfully; seconds=',summary['training_seconds'],flush=True);return summary

def main():
    p=argparse.ArgumentParser(description='Generate original MPC400 teacher demonstrations, then train historical wide policy; optional reuse')
    p.add_argument('--data-root',type=Path,default=Path(os.environ.get('PATU_DATA_ROOT',HERE.parent.parent)))
    p.add_argument('--prepared-root',type=Path);p.add_argument('--control-cache',type=Path);p.add_argument('--reuse-demonstrations',type=Path);p.add_argument('--demonstrations-dir',type=Path);p.add_argument('--seed',type=int,default=44);p.add_argument('--epochs',type=int,default=240);p.add_argument('--smoke-test',action='store_true');p.add_argument('--verify',action='store_true');p.add_argument('--generate-only',action='store_true');a=p.parse_args()
    ctx=context()
    if a.verify:mpc.verify();return
    folder=a.reuse_demonstrations or a.demonstrations_dir or a.data_root/('PATU_teacher_smoke' if a.smoke_test else 'PATU_teacher_demonstrations')
    if not a.reuse_demonstrations:generate(ctx,mpc.Dataset(a.data_root,a.prepared_root,a.control_cache),folder,a.smoke_test)
    if a.generate_only:return
    data=pool(folder,a.smoke_test);dest=HERE/'checkpoints'/('wide_retrained_'+('smoke_' if a.smoke_test else '')+'s'+str(a.seed)+'.pt')
    if dest.exists():raise FileExistsError('Checkpoint already exists; refusing overwrite: '+str(dest))
    summary=train(ctx,data,dest,a.seed,epochs=3 if a.smoke_test else a.epochs)
    result=HERE/'results'/dest.stem
    (result/'demonstrations.json').write_text(json.dumps(dict(path=str(folder.resolve()),reused=bool(a.reuse_demonstrations),samples=len(data['x']),smoke_only=a.smoke_test,teacher='Original400ms maxiter5; historical shifted second move',Test_used=False),indent=2))
    print('A newly trained policy is not assumed to reproduce frozen wide_s44 closed-loop metrics.',flush=True)
if __name__=='__main__':main()
