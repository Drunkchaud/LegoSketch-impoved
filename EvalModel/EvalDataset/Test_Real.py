"""Pretrained LegoSketch full-frequency evaluation. Default is a file-only preflight.

Run from WSL in the existing Lego environment. --run explicitly starts inference.
Keeps original Cluster prediction policy; does not run ES, MS or traditional baselines.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import inspect
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import re

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FILES = {'kosarak': 'kosarak.npz', 'webdocs': 'webdocs.npz',
         'mawi_src_ip': '202401011400_src-ip.npz'}
FILES.update({name+'_test':name+'_freqs_test.npz' for name in ('kosarak','webdocs','mawi_src_ip')})


def dataset_path(args,name):
    return Path(args.input_files[name]) if getattr(args,'input_files',None) else args.data_dir / FILES[name]


def load_table(path):
    import numpy as np
    with np.load(path,allow_pickle=False) as z:
        if not {'items','freqs'}.issubset(z.files):raise ValueError(str(path)+': requires items and freqs')
        ids,freqs=z['items'],z['freqs']
    if ids.dtype!=np.int64 or ids.ndim!=1 or freqs.shape!=ids.shape:
        raise ValueError(str(path)+': expected int64 items and aligned 1D freqs')
    if not np.issubdtype(freqs.dtype,np.integer) or not len(ids) or np.any(freqs<=0):
        raise ValueError(str(path)+': expected nonempty positive integer frequencies')
    if not np.all(ids[1:]>ids[:-1]):raise ValueError(str(path)+': items must be sorted and unique')
    return ids,freqs


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    inputs=p.add_mutually_exclusive_group()
    inputs.add_argument('--datasets', nargs='+', choices=list(FILES))
    inputs.add_argument('--npz', nargs='+', type=Path, help='evaluate these frequency tables directly; no dataset registration or splitting')
    p.add_argument('--data-dir', type=Path, default=ROOT.parents[1] / '数据集' / 'processed')
    p.add_argument('--models', nargs='+', type=Path, default=[
        ROOT / 'EvalModel/EvalDataset/ModelRepository/FinalAblation' / f'Base_seed{i}_model'
        for i in (0, 1)])
    p.add_argument('--budgets-kib', nargs='+', type=int, default=list(range(300, 2400, 300)))
    p.add_argument('--hash', choices=['random', 'cuda'], default='random',
                   help='random matches Test_Lego default; cuda uses supplied CUDA hash implementation')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--repeats', type=int, default=11, help='local timing: first is warmup, then ten measured repetitions by default')
    p.add_argument('--save-predictions', action='store_true', help='save first-repeat predictions for independent error checking')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'EvalModel/EvalDataset/RealResults')
    p.add_argument('--run', action='store_true', help='without this flag, check paths only')
    p.add_argument('--no-plots', action='store_true')
    args = p.parse_args()
    if args.npz:
        names=[re.sub(r'[^\w.-]+','_',path.stem) for path in args.npz]
        if len(set(names))!=len(names):p.error('--npz files must have different names')
        if any(path.suffix.lower()!='.npz' for path in args.npz):p.error('--npz expects .npz files')
        args.datasets=names;args.input_files={name:str(path.resolve()) for name,path in zip(names,args.npz)}
    else:
        args.datasets=args.datasets or ['kosarak','webdocs','mawi_src_ip'];args.input_files={}
    if args.repeats < 2 or min(args.budgets_kib) <= 0:
        p.error('need repeats>=2 (one warmup) and positive budgets')
    return args


def preflight(args):
    paths = args.models + [dataset_path(args,name) for name in args.datasets]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError('\n'.join(missing))
    for name in args.datasets:
        ids,freqs=load_table(dataset_path(args,name))
        print('Validated:',name,'items=',len(ids),'stream_length=',sum(map(int,freqs)))
    print(json.dumps(dict(mode='RUN' if args.run else 'PREFLIGHT ONLY',
        datasets=args.datasets, files=[str(p.resolve()) for p in paths],
        budgets_kib=args.budgets_kib, hash=args.hash, device=args.device,
        repeats=args.repeats, prediction='original Cluster gate and min(decoder, CM)',
        timing='aggregate weighted-item write and full distinct-item query; NOT per-packet streaming'),
        indent=2, ensure_ascii=False))


def timed(torch, device, operation):
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = operation()
    torch.cuda.synchronize(device)
    return result, time.perf_counter() - start


def load_model(torch, path, device):
    # Existing repository checkpoints are trusted full Python model objects.
    kwargs = {'map_location': device}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = False
    return torch.load(str(path), **kwargs).to(device).eval()


def errors(torch, pred, truth):
    if not torch.isfinite(pred).all():
        raise ValueError('Nonfinite model predictions')
    diff = pred.double() - truth.double()
    return dict(AAE=diff.abs().mean().item(), ARE=(diff.abs()/truth.double()).mean().item(),
                MSE=diff.square().mean().item(), negative_predictions=int((pred < 0).sum().item()))


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    import random
    import numpy as np
    import torch
    from EvalModel.EvalDataset.EvalModelUtil import Cluster, replace_model_hash

    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Use the existing CUDA Lego environment for this evaluator')
    stamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')
    dataset_label = '_'.join(dict.fromkeys(args.datasets))
    output = args.output_dir / f'{dataset_label}_{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    sources = list((ROOT/'SourceCode').rglob('*.py')) + [Path(__file__).resolve(),
        ROOT/'EvalModel/EvalDataset/EvalModelUtil.py', ROOT/'EvalModel/EvalDataset/DatasetLoader.py']
    manifest = dict(config={k: str(v) if isinstance(v, Path) else
                    [str(x) if isinstance(x, Path) else x for x in v] if isinstance(v, list) else v
                    for k, v in vars(args).items()}, python=sys.version, platform=platform.platform(),
        torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(device), created=datetime.now().astimezone().isoformat(),
        sources={str(p.relative_to(ROOT)): sha256(p) for p in sources},
        models={str(p.resolve()): sha256(p) for p in args.models}, datasets={},
        protocol='Evaluate each supplied frequency table as-is; pretrained weights; one write and full query.',
        budget='counter-memory only; network/statistics/temporary tensors excluded',
        time='synchronized wall seconds; load/clear excluded; first repetition warmup excluded',
        throughput='distinct weighted items/s for write and distinct queries/s; NOT events/s',
        MSE='mean((prediction-frequency)^2); ASE terminology not assumed',
        paper_reference='LegoSketch.pdf sections 5.1, 5.2, 5.4; Algorithm 2',
        paper_throughput_reproduction=False,
        status='running')
    try:
        manifest['git_commit'] = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        manifest['git_commit'] = None
    def save_manifest():
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    save_manifest()
    rows = []
    timing_rows = []
    diagnostic_rows = []
    try:
        with torch.no_grad():
            for name in args.datasets:
                path = dataset_path(args,name)
                ids,freqs=load_table(path)
                if ids.dtype != np.int64 or ids.ndim != 1 or freqs.shape != ids.shape:
                    raise ValueError('Expected int64 items and aligned one-dimensional frequencies')
                if not np.issubdtype(freqs.dtype,np.integer) or not len(ids) or np.any(freqs<=0):
                    raise ValueError('Expected positive integer frequencies')
                if not np.all(ids[1:] > ids[:-1]):
                    raise ValueError('Expected sorted unique IDs')
                manifest['datasets'][name] = dict(path=str(path.resolve()),sha256=sha256(path),
                    distinct_items=len(ids),stream_length=int(freqs.sum()))
                sidecar=path.with_suffix('.json')
                if sidecar.is_file():
                    manifest['datasets'][name]['conversion']=json.loads(sidecar.read_text(encoding='utf-8'))
                save_manifest()
                x=torch.as_tensor(ids,device=device).view(-1,1)
                y=torch.as_tensor(freqs,device=device)
                for model_path in args.models:
                    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
                    model=load_model(torch,model_path,device)
                    if args.hash=='cuda':
                        model=replace_model_hash(model)
                    model.clear()
                    unit_kib=model.memory_module.memory_matrix.numel()*model.memory_module.memory_matrix.element_size()/1024
                    model.address_module.set_prod_sparse(True)
                    for budget in args.budgets_kib:
                        count=int(budget//unit_kib)
                        if count<1:
                            raise ValueError(f'Budget {budget} smaller than one brick {unit_kib} KiB')
                        # New Cluster each repetition also resets per-brick frequency sums.
                        for repeat in range(args.repeats):
                            cluster=Cluster(model,count,args.hash=='cuda')
                            torch.cuda.reset_peak_memory_stats(device)
                            _,write_s=timed(torch,device,lambda:cluster.write(x,y))
                            (pred,cm),query_s=timed(torch,device,lambda:cluster.dec_query(x))
                            peak=torch.cuda.max_memory_allocated(device)
                            if repeat == 0:
                                # Extra untimed query: diagnostic copies must not contaminate timing.
                                details=[]
                                check_pred,check_cm=cluster.dec_query(x, diagnostics=details)
                                if not torch.equal(pred,check_pred) or not torch.equal(cm,check_cm):
                                    raise ValueError('Diagnostic query changed prediction')
                                prefix=f'{name}_{model_path.stem}_{budget}kib'
                                raw=torch.empty_like(y,device='cpu',dtype=pred.dtype)
                                clipped=torch.empty_like(raw)
                                for detail in details:
                                    pos=detail.pop('positions')
                                    raw[pos]=detail.pop('raw_decoder').view(-1)
                                    clipped[pos]=detail.pop('clipped_decoder').view(-1)
                                    rule=detail.pop('rule')
                                    truth=y.detach().cpu()[pos].view(-1,1)
                                    detail.update(dataset=name,checkpoint=str(model_path.resolve()),budget_kib=budget,
                                        rule_AAE=errors(torch,rule,truth)['AAE'])
                                    diagnostic_rows.append(detail)
                                write_csv(output/'brick_diagnostics.csv',diagnostic_rows)
                                diagnostic_metrics={label:errors(torch,value,y.detach().cpu().view(-1,1))
                                    for label,value in [('raw_decoder',raw.view(-1,1)),('clipped_decoder',clipped.view(-1,1)),
                                                        ('final',pred.cpu()),('rule',cm.cpu())]}
                                (output/(prefix+'_diagnostics.json')).write_text(json.dumps(diagnostic_metrics,indent=2),encoding='utf-8')
                                if args.save_predictions:
                                    np.savez_compressed(output/(prefix+'_predictions.npz'),items=ids,freqs=freqs,
                                        raw_decoder=raw.numpy(),clipped_decoder=clipped.numpy(),
                                        final=pred.cpu().numpy().ravel(),rule=cm.cpu().numpy().ravel())
                                del details,raw,clipped,check_pred,check_cm,rule,truth
                            timing_rows.append(dict(dataset=name,checkpoint=str(model_path.resolve()),hash=args.hash,
                                requested_kib=budget,actual_counter_kib=count*unit_kib,repeat=repeat,warmup=repeat==0,
                                distinct_items=len(ids),stream_length=int(freqs.sum()),write_seconds=write_s,query_seconds=query_s,
                                weighted_items_per_second=len(ids)/write_s,distinct_queries_per_second=len(ids)/query_s,
                                peak_allocated_bytes=peak,protocol='local_aggregate_frequency_table_not_paper_throughput'))
                            write_csv(output/'timing_runs.csv',timing_rows)
                            for method,result in [('Lego',pred),('weighted_CM_readhead',cm)]:
                                row=dict(dataset=name,checkpoint=str(model_path.resolve()),hash=args.hash,
                                    requested_kib=budget,actual_counter_kib=count*unit_kib,brick_count=count,
                                    repeat=repeat,warmup=repeat==0,method=method,distinct_items=len(ids),
                                    stream_length=int(freqs.sum()),**errors(torch,result,y.view(-1,1)))
                                rows.append(row)
                            write_csv(output/'runs.csv',rows)
                            print(name,model_path.name,budget,repeat,'completed',flush=True)
                            del cluster,pred,cm,result
                        torch.cuda.empty_cache()
                    del model
                del x,y,ids,freqs
        measured=[r for r in rows if not r['warmup']]
        summary=[]
        for key in sorted({(r['dataset'],r['requested_kib'],r['method']) for r in measured}):
            subset=[r for r in measured if (r['dataset'],r['requested_kib'],r['method'])==key]
            row=dict(dataset=key[0],requested_kib=key[1],method=key[2],checkpoint_count=len(args.models))
            # Timing repetitions are not independent training seeds. Summarize each checkpoint first.
            for field in ['AAE','ARE','MSE']:
                per_model=[np.mean([r[field] for r in subset if r['checkpoint']==str(p.resolve())]) for p in args.models]
                row[field+'_mean']=float(np.mean(per_model))
                row[field+'_checkpoint_std']=float(np.std(per_model))
            summary.append(row)
        write_csv(output/'summary.csv',summary)
        timing_summary=[]
        for name,budget,checkpoint in sorted({(r['dataset'],r['requested_kib'],r['checkpoint']) for r in timing_rows}):
            subset=[r for r in timing_rows if not r['warmup'] and
                    (r['dataset'],r['requested_kib'],r['checkpoint'])==(name,budget,checkpoint)]
            row=dict(dataset=name,requested_kib=budget,checkpoint=checkpoint,samples=len(subset),
                protocol='local_aggregate_frequency_table_not_paper_throughput')
            for field in ['write_seconds','query_seconds','weighted_items_per_second','distinct_queries_per_second']:
                values=[r[field] for r in subset]
                row[field+'_median']=float(np.median(values))
                row[field+'_p25']=float(np.percentile(values,25))
                row[field+'_p75']=float(np.percentile(values,75))
            timing_summary.append(row)
        write_csv(output/'timing_summary.csv',timing_summary)
        if not args.no_plots and len(set(args.budgets_kib)) > 1:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            for name in args.datasets:
                for metric in ['AAE','ARE']:
                    fig,ax=plt.subplots()
                    # Rule readhead is an internal diagnostic, not an independent CMS baseline.
                    for method in ['Lego']:
                        values=sorted([r for r in summary if r['dataset']==name and r['method']==method],key=lambda r:r['requested_kib'])
                        ax.errorbar([r['requested_kib'] for r in values],[r[metric+'_mean'] for r in values],
                            yerr=[r[metric+'_checkpoint_std'] for r in values],marker='o',label=method+' (checkpoint std)')
                    ax.set(xlabel='Requested counter budget (KiB)',ylabel=metric,title=name,ylim=(0,None))
                    ax.legend();fig.tight_layout();fig.savefig(output/f'{name}_{metric}.png');plt.close(fig)
        if len(set(args.budgets_kib)) == 1:
            print('Single budget: tables only; no space-accuracy curve generated.')
        manifest['status']='complete'
    except Exception as exc:
        manifest['status']='failed';manifest['error']=repr(exc)
        raise
    finally:
        save_manifest()
    print('Results:',output)


if __name__=='__main__':
    args=parse_args()
    preflight(args)
    if args.run:
        run(args)
    else:
        print('NPZ format validated; no model loaded or inference run. Add --run to evaluate.')
