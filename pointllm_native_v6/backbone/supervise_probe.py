"""Bounded-process evidence wrapper: preserve heartbeat, RSS and exact exit.

This supervisor does not alter model code, fabricate a result, retry failures,
or infer that SIGKILL means OOM. External destruction of the supervisor itself
can still leave only heartbeat evidence, which must remain an unknown cause.
"""
import argparse,json,os,signal,subprocess,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);p.add_argument('--console',type=Path,required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    command=a.command[1:] if a.command and a.command[0]=='--' else a.command
    if not command:raise SystemExit('A child command is required')
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    started=time.time();history=a.receipt.with_suffix('.heartbeat.jsonl');events=[];child=None;peak=0
    result={'status':'supervisor_started','command':command,'started_unix':started,'heartbeat_file':history.name,'peak_child_rss_kib':0,'events':events}
    def persist():
        result['elapsed_s']=time.time()-started;result['peak_child_rss_kib']=peak
        temp=a.receipt.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(a.receipt)
    def on_signal(sig,frame):
        events.append({'supervisor_received_signal':sig,'elapsed_s':time.time()-started});persist()
        if child is not None and child.poll() is None:child.terminate()
    signal.signal(signal.SIGTERM,on_signal);signal.signal(signal.SIGINT,on_signal)
    persist()
    env=dict(os.environ);env.update({'HF_ENABLE_PARALLEL_LOADING':'false','MALLOC_ARENA_MAX':'2','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
    result['controlled_environment']={key:env[key] for key in ['HF_ENABLE_PARALLEL_LOADING','MALLOC_ARENA_MAX','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']}
    with a.console.open('w') as console,history.open('a') as heart:
        child=subprocess.Popen(command,stdout=console,stderr=subprocess.STDOUT,env=env)
        result['child_pid']=child.pid;result['status']='child_running';persist()
        while child.poll() is None:
            rss=hwm=None
            try:
                status=Path(f'/proc/{child.pid}/status').read_text()
                for line in status.splitlines():
                    if line.startswith('VmRSS:'):rss=int(line.split()[1])
                    if line.startswith('VmHWM:'):hwm=int(line.split()[1])
                peak=max(peak,rss or 0,hwm or 0)
            except (FileNotFoundError,ProcessLookupError):pass
            row={'unix':time.time(),'elapsed_s':time.time()-started,'child_rss_kib':rss,'child_vmhwm_kib':hwm}
            heart.write(json.dumps(row)+'\n');heart.flush();result['last_heartbeat']=row;persist();time.sleep(2)
        code=child.wait();result['child_returncode']=code
        result['status']='child_completed' if code==0 else 'child_failed'
        result['termination_signal']=-code if code<0 else None
        result['termination_cause']='No cause inferred from an exit signal alone.' if code<0 else 'Normal process exit code recorded.'
        persist();print(json.dumps(result,indent=2),flush=True)
    raise SystemExit(0 if result['child_returncode']==0 else 1)


if __name__=='__main__':main()
