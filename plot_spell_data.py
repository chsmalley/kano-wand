#!/usr/bin/env python3
import argparse, json, math, sys
from pathlib import Path
import matplotlib.pyplot as plt

DEFAULT_RUNS=[Path('training_data'),Path('training_data_1'),Path('training_data_2'),Path('training_data_3')]

def load(p):
    with p.open(encoding='utf-8') as f:return json.load(f)
def series(records,key):
    out=[]
    for r in records:
        try:
            t,v=float(r['timestamp']),float(r[key])
            if math.isfinite(t) and math.isfinite(v):out.append((t,v))
        except (KeyError,TypeError,ValueError):pass
    if out:
        t0=out[0][0]; return [(t-t0,v) for t,v in out]
    return []
def gaps(records,threshold=.2):
    ts=[]
    for r in records:
        try:
            t=float(r['timestamp'])
            if math.isfinite(t):ts.append(t)
        except (KeyError,TypeError,ValueError):pass
    if not ts:return []
    return [(b-ts[0],b-a) for a,b in zip(ts,ts[1:]) if b-a>threshold]
def mark(ax,records):
    for t,g in gaps(records):
        ax.axvline(t,linestyle='--',linewidth=.8,alpha=.45)
        ax.text(t,.98,f'{g:.2f}s',transform=ax.get_xaxis_transform(),rotation=90,va='top',fontsize=7)
def summary(d,p):
    m,o=d.get('motion',[]),d.get('orientation',[])
    print('\n'+'='*72); print(f'File: {p}'); print(f"Spell: {d.get('spell','Unknown')}  Repetition: {d.get('repetition','?')}")
    print(f'Motion: {len(m)} samples   Quaternion: {len(o)} samples')
    if m:
        try: print(f"Duration: {float(m[-1]['timestamp'])-float(m[0]['timestamp']):.3f} s")
        except: pass
    gm,go=gaps(m),gaps(o); print(f'Motion gaps > 0.2 s: {len(gm)}   Quaternion gaps > 0.2 s: {len(go)}')
    if gm: print(f'Largest motion gap: {max(gm,key=lambda x:x[1])[1]:.3f} s')
    if go: print(f'Largest quaternion gap: {max(go,key=lambda x:x[1])[1]:.3f} s')
    print('='*72)
def plot_one(d,p,kind,show_gaps,save):
    m,o=d.get('motion',[]),d.get('orientation',[])
    groups={'motion':([('acc_x','acc_y','acc_z','Accelerometer'),('mag_x','mag_y','mag_z','Magnetometer'),('pitch','roll','yaw','Pitch / Roll / Yaw')],m),
            'quaternion':([('x','y','z','Quaternion')],o)}
    if kind=='combined':
        groups={'combined':([('acc_x','acc_y','acc_z','Accelerometer'),('mag_x','mag_y','mag_z','Magnetometer'),('pitch','roll','yaw','Pitch / Roll / Yaw')],m), 'quat':([('x','y','z','w','Quaternion')],o)}
    if kind=='combined':
        fig,axes=plt.subplots(4,1,figsize=(13,13)); specs=[groups['combined'][0][0],groups['combined'][0][1],groups['combined'][0][2],groups['quat'][0][0]]; recs=[m,m,m,o]
    else:
        specs=groups[kind][0]; recs=[groups[kind][1]]*len(specs); fig,axes=plt.subplots(len(specs),1,figsize=(12,4*len(specs)),squeeze=False); axes=axes.ravel()
    for ax,spec,rec in zip(axes,specs,recs):
        for key in spec[:-1]:
            s=series(rec,key)
            if s:
                x,y=zip(*s); ax.plot(x,y,label=key)
        ax.set_title(spec[-1]); ax.grid(True,alpha=.3); ax.legend(); ax.set_xlabel('Time since recording started (seconds)')
        if show_gaps: mark(ax,rec)
    fig.suptitle(f"{d.get('spell',p.parent.name)} — repetition {d.get('repetition','?')}\n{p}")
    fig.tight_layout()
    if save:
        save.mkdir(parents=True,exist_ok=True); out=save/f'{p.stem}_{kind}.png'; fig.savefig(out,dpi=150,bbox_inches='tight'); print(f'Saved: {out}')
    if not getattr(plot_one,'no_show',False): plt.show()
    plt.close(fig)
def find(run,spell=None):
    if spell:return sorted((run/spell).glob('recording_*.json'))
    return sorted(run.glob('*/recording_*.json'))
def main():
    ap=argparse.ArgumentParser(description='Plot Kano Wand motion and quaternion JSON recordings.')
    ap.add_argument('--file',type=Path); ap.add_argument('--spell'); ap.add_argument('--run',type=Path); ap.add_argument('--all-runs',action='store_true')
    ap.add_argument('--plot',choices=['motion','quaternion','combined'],default='combined'); ap.add_argument('--save',type=Path); ap.add_argument('--show-gaps',action='store_true'); ap.add_argument('--no-show',action='store_true')
    a=ap.parse_args(); plot_one.no_show=a.no_show
    if a.file: files=[a.file]
    elif a.spell:
        runs=DEFAULT_RUNS if a.all_runs else [a.run or Path('training_data')]; files=sum((find(r,a.spell) for r in runs),[])
    else:
        files=[]
        for r in DEFAULT_RUNS: files+=find(r)
        if not files: print('No recording_*.json files found.'); return 1
        for i,p in enumerate(files,1):
            try:d=load(p); print(f'{i:3}: {d.get("spell",p.parent.name):22} rep {d.get("repetition","?")}  {p}')
            except: print(f'{i:3}: {p}')
        try: files=[files[int(input('\nEnter recording number: '))-1]]
        except: return 1
    if not files: print(f'No recordings found for {a.spell}.'); return 1
    for p in files:
        if not p.exists(): print(f'File not found: {p}',file=sys.stderr); continue
        try:d=load(p)
        except Exception as e: print(f'Could not read {p}: {e}',file=sys.stderr); continue
        summary(d,p); plot_one(d,p,a.plot,a.show_gaps,a.save)
    return 0
if __name__=='__main__': raise SystemExit(main())
