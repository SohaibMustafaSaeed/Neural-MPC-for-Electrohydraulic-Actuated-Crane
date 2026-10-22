import os,sys
from pathlib import Path
for _thread in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[_thread]='1'
sys.dont_write_bytecode=True
import argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'MPC'));import mpc
HERE=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser(description='Original MPC400 vs conventional3 vs neural-assisted3; shared learned-model simulation')
    p.add_argument('--data-root',type=Path,default=Path(os.environ.get('PATU_DATA_ROOT',HERE.parent.parent)))
    p.add_argument('--prepared-root',type=Path);p.add_argument('--control-cache',type=Path);p.add_argument('--checkpoint',default='original');p.add_argument('--verify',action='store_true');p.add_argument('--smoke-test',action='store_true');p.add_argument('--recompute',action='store_true');a=p.parse_args()
    if a.verify:mpc.verify();return
    path=HERE/'checkpoints/wide_s44.pt' if a.checkpoint=='original' else Path(a.checkpoint).resolve();policy=mpc.Policy(path);ctx=mpc.context();dataset=mpc.Dataset(a.data_root,a.prepared_root,a.control_cache)
    names=['mpc400','conventional3','hybrid'];tag=('smoke' if a.smoke_test else '') if a.checkpoint=='original' else path.stem+('/smoke' if a.smoke_test else '')
    out=HERE/'results'/tag;folders={name:(mpc.ROOT/'results'/('smoke' if a.smoke_test else '')/'mpc400/runs' if name=='mpc400' else out/('conventional_3iter' if name=='conventional3' else 'neural_assisted_3iter')/'runs') for name in names}
    for name in names:
        for spec in mpc.inventory():
            mpc.rollout(ctx,dataset,spec,name,folders[name],2 if a.smoke_test else None,policy_override=policy if name=='hybrid' else None,force=a.recompute and name!='mpc400')
            if a.checkpoint=='original' and not a.smoke_test:mpc.check_saved(name,spec,folders[name])
    mpc.results_summary(names,folders,out/'comparison',dataset,ctx,2 if a.smoke_test else None)
if __name__=='__main__':main()
