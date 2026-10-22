import os,sys
from pathlib import Path
for _thread in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[_thread]='1'
sys.dont_write_bytecode=True
import mpc
import argparse
def main():
    p=argparse.ArgumentParser(description='Frozen9916Compact Cartesian MPC on nine Train-derived validation recordings')
    p.add_argument('--data-root',type=Path,default=Path(os.environ.get('PATU_DATA_ROOT',Path(__file__).resolve().parents[1].parent)))
    p.add_argument('--prepared-root',type=Path);p.add_argument('--control-cache',type=Path);p.add_argument('--controller',choices=['mpc200','mpc400','both'],default='both');p.add_argument('--verify',action='store_true');p.add_argument('--smoke-test',action='store_true');p.add_argument('--recompute',action='store_true');a=p.parse_args()
    if a.verify:mpc.verify();return
    dataset=mpc.Dataset(a.data_root,a.prepared_root,a.control_cache);ctx=mpc.context();names=['mpc200','mpc400'] if a.controller=='both' else [a.controller];limit=2 if a.smoke_test else None
    folders={name:mpc.ROOT/'results'/('smoke' if a.smoke_test else '')/name/'runs' for name in names}
    for name in names:
        for spec in mpc.inventory():
            mpc.rollout(ctx,dataset,spec,name,folders[name],limit,force=a.recompute)
            if not a.smoke_test:mpc.check_saved(name,spec,folders[name])
    mpc.results_summary(names,folders,mpc.ROOT/'results'/('smoke/comparison' if a.smoke_test else 'comparison'),dataset,ctx,limit)
if __name__=='__main__':main()
