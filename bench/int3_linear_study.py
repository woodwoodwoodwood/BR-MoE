"""Real-input linear tuning with complete routed+shared MoE and full INT3 e2e."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
for sub in ('tools','bench','BR-MoE/kernels','BR-MoE/kernels/triton_int3'):
    sys.path.insert(0,str(ROOT/sub))
import torch
import triton
from full_int3_study import graph_us
from moe_large_batch_study import collect
from moe_fuse_study import e2e,weights
from brmoe_int3_vllm import linear_method as linear
from brmoe_int3_vllm.kernel import get_fused_moe_int3,get_moe_cuda_ext
from marlin_int3_moe.moe_cuda import fused_moe_int3_cuda

ORIGINAL_TILES,ORIGINAL_GEMM=linear.pick_tiles,linear.int3_moe_gemm
ORIGINAL_IMPL=linear._brmoe_int3_linear_impl


def install(config):
    linear.pick_tiles,linear.int3_moe_gemm=ORIGINAL_TILES,ORIGINAL_GEMM
    linear._brmoe_int3_linear_impl=ORIGINAL_IMPL
    # Keep historical controls stable after the production default changes.
    os.environ['BRMOE_LINEAR_BACKEND']='legacy'
    os.environ['BRMOE_PREFILL_BACKEND']='legacy'
    os.environ['BRMOE_SMALL_DECODE_BACKEND']='legacy'
    if config=='production':
        os.environ['BRMOE_LINEAR_BACKEND']='auto'
        os.environ['BRMOE_PREFILL_BACKEND']='auto'
        os.environ['BRMOE_SMALL_DECODE_BACKEND']='auto'
        return
    if config=='previous':
        os.environ['BRMOE_LINEAR_BACKEND']='auto'
        return
    if config=='baseline':return
    if config.startswith('prefill_'):
        install('previous')
        from brmoe_int3_vllm.linear_tc import int3_linear_tc
        bm,bn,bk,stages,warps=map(int,config.split('_')[1:])
        def prefill(x,w,s,z,gs):
            if x.shape[0]>128:
                return int3_linear_tc(x,w,s,z,gs,block_m=bm,block_n=bn,block_k=bk,
                                      num_stages=stages,num_warps=warps)
            return ORIGINAL_IMPL(x,w,s,z,gs)
        linear._brmoe_int3_linear_impl=prefill
        return
    if config in ('hybrid','hybrid64') or config.startswith('hybrid_m'):
        install('single_32_3')
        from brmoe_int3_vllm.linear_tc import int3_linear_tc
        cutoff=int(config.removeprefix('hybrid_m')) if config.startswith('hybrid_m') else 8
        def hybrid(x,w,s,z,gs):
            if cutoff<x.shape[0]<=128:
                return int3_linear_tc(x,w,s,z,gs,block_m=32,block_n=32 if config=='hybrid' else 64,
                                      block_k=128,num_stages=3,num_warps=4)
            return ORIGINAL_IMPL(x,w,s,z,gs)
        linear._brmoe_int3_linear_impl=hybrid
        return
    if config.startswith('tc_'):
        from brmoe_int3_vllm.linear_tc import int3_linear_tc
        parts=config.split('_')
        force_small=parts[1]=='all'
        bm,bn,bk,stages,warps=map(int,parts[2:] if force_small else parts[1:])
        def tc(x,w,s,z,gs):
            if x.shape[0]<=8 and not force_small:return ORIGINAL_IMPL(x,w,s,z,gs)
            return int3_linear_tc(x,w,s,z,gs,block_m=bm,block_n=bn,block_k=bk,num_stages=stages,num_warps=warps)
        linear._brmoe_int3_linear_impl=tc
        return
    # Benchmark-only hooks; promote only after full-model measurements.
    bits=config.split('_')
    bk=int(bits[1]) if len(bits)>1 else 32
    stages=int(bits[2]) if len(bits)>2 else 3
    def tiles(K,N,gs,M=None):
        bm,bn,old_bk,slot=ORIGINAL_TILES(K,N,gs,M)
        return bm,bn,bk,bm
    def gemm(*a,**kw):
        kw['num_stages']=stages
        return ORIGINAL_GEMM(*a,**kw)
    linear.pick_tiles,linear.int3_moe_gemm=tiles,gemm


def category(name):
    return 'attention' if '.self_attn.' in name else ('shared' if '.shared_experts.' in name else 'first_mlp')


def load_data(args):
    inputs=torch.load(args.input/'linear_inputs.pt',weights_only=True)
    weights_cpu=torch.load(args.input/'linear_weights.pt',weights_only=True)
    packed={name:tuple(v.cuda() if isinstance(v,torch.Tensor) else v for v in vals)
            for name,vals in weights_cpu.items()}
    return inputs,packed


def micro(args):
    data,packed=load_data(args)
    records=[]
    batches=list(map(int,args.batch_sizes.split(',')))
    for m in batches+([512,2048] if args.prefill else []):
        # Prefill: latest observed sample for each representative projection.
        chosen={s['layer']:s for s in data if s['m']==m and (m in (512,2048) or s['batch']==m)}
        if not chosen and args.slice_small and m<8:
            # Real batch-8 decode rows, sliced only for a small-M tuning probe.
            # Full-model validation still uses real requests at the target batch.
            chosen={s['layer']:{**s,'x':s['x'][:m]} for s in data if s['m']==8 and s['batch']==8}
        if not chosen:continue
        if args.quick:
            seen=set();selected={}
            for name,s in chosen.items():
                shape=(s['x'].shape[-1],packed[name][1].shape[-1])
                if shape not in seen:selected[name]=s;seen.add(shape)
            chosen=selected
        inputs={name:(s['x'].cuda(),*packed[name]) for name,s in sorted(chosen.items())}
        install('baseline')
        refs={name:linear._brmoe_int3_linear_impl(*a).clone().float() for name,a in inputs.items()}
        for config in args.configs.split(','):
            install(config)
            errors={}
            for name,a in inputs.items():
                got=linear._brmoe_int3_linear_impl(*a).float()
                ref=refs[name];rel=float((got-ref).norm()/ref.norm().clamp_min(1e-8))
                assert torch.isfinite(got).all() and rel<.003,(m,config,name,rel)
                errors[name]=rel
            for cat in ('attention','shared','first_mlp','all'):
                group=[a for name,a in inputs.items() if cat=='all' or category(name)==cat]
                if not group:continue
                def fn():
                    for a in group:linear._brmoe_int3_linear_impl(*a)
                us=graph_us(fn,repeats=7)
                rec=dict(m=m,config=config,category=cat,layers=len(group),us=us,max_relative_l2=max(errors.values()))
                records.append(rec);print(json.dumps(rec),flush=True)
                (args.out/'linear_micro.json').write_text(json.dumps(records,indent=2))
        del refs,inputs


def moe(args):
    """27 complete expert blocks: shared MLP + routed MoE + their sum.

    Uses the actual shared gate input; asserts it equals the captured routed x.
    Gate/top-k have already been executed to capture ids and weights.
    """
    data,packed=load_data(args)
    import vllm._custom_ops  # Register the same CUDA activation used by the model.
    routes=torch.load(args.input/'real_inputs.pt',weights_only=True)
    get_fused_moe_int3();ext=get_moe_cuda_ext();assert ext is not None
    names=sorted({s['layer'] for s in routes})
    pks={name:weights(name) for name in names}
    records=[]
    for m in map(int,args.batch_sizes.split(',')):
        samples=[{**s,**{k:s[k].cuda() for k in ('x','weights','ids')}} for s in routes if s['m']==m]
        assert len(samples)==27
        def block(s):
            root=s['layer'].removesuffix('.experts')+'.shared_experts.'
            # gate_up is fused in vLLM, as in captured linear_weights.pt.
            a=linear._brmoe_int3_linear_impl(s['x'],*packed[root+'gate_up_proj'])
            act=torch.empty((a.shape[0],a.shape[1]//2),dtype=a.dtype,device=a.device)
            torch.ops._C.silu_and_mul(act,a)
            shared=linear._brmoe_int3_linear_impl(act,*packed[root+'down_proj'])
            p,pk=pks[s['layer']]
            routed=fused_moe_int3_cuda(s['x'],s['weights'],s['ids'],pk,ext,packed=p,fuse_ops=True)
            return shared+routed
        for s in samples:
            root=s['layer'].removesuffix('.experts')+'.shared_experts.gate_up_proj'
            x=next(t['x'] for t in data if t['layer']==root and t['m']==m and t['batch']==m)
            assert torch.equal(s['x'].cpu(),x),(root,m,'shared and routed input differ')
        install('baseline');refs=[block(s).clone().float() for s in samples]
        for config in args.configs.split(','):
            install(config);errs=[]
            for s,ref in zip(samples,refs):
                got=block(s).float();rel=float((got-ref).norm()/ref.norm().clamp_min(1e-8))
                assert torch.isfinite(got).all() and rel<.003,(m,config,rel)
                errs.append(rel)
            def fn():
                for s in samples:block(s)
            rec=dict(m=m,config=config,layers=27,us=graph_us(fn,repeats=7),max_relative_l2=max(errs))
            records.append(rec);print(json.dumps(rec),flush=True)
            (args.out/'complete_moe.json').write_text(json.dumps(records,indent=2))


def verify(args):
    from int3_moe.packing import unpack_int3,pack_int3
    _,packed=load_data(args)
    representatives={}
    for name,vals in packed.items():
        w,s,z,gs=vals
        representatives.setdefault((s.shape[0]*gs,s.shape[1]),(name,vals))
    torch.manual_seed(20260926)
    if any(c.startswith(('tc_','hybrid','production')) for c in args.configs.split(',')):
        # Exercise N/K tails and GS=128, beyond the six real model shapes.
        for k,n,gs in ((192,80,64),(256,96,128)):
            q=torch.randint(0,8,(n,k),dtype=torch.int32,device='cuda')
            w=pack_int3(q,transposed=True)
            s=(torch.rand(k//gs,n,device='cuda')*.02+.005).half()
            z=(torch.rand_like(s)*5+.5).half()
            representatives[(k,n)]=('synthetic_tail',(w,s,z,gs))
    torch.backends.cuda.matmul.allow_tf32=False
    records=[]
    for (k,n),(name,(w,s,z,gs)) in representatives.items():
        q=unpack_int3(w,k,transposed=True).half()  # [N,K]
        deq=((q-z.repeat_interleave(gs,0).T).half()*s.repeat_interleave(gs,0).T).half().float()
        test_m = (1,2,3,4,5,7,8,9,16,17,31,32,33,64,65,127,128,129,257)
        if args.prefill:
            test_m += (511,512,513,1024,2048)
        for m in test_m:
            x=(torch.randn(m,k*2,device='cuda')*.1).half()[:,::2]
            for config in args.configs.split(','):
                install(config)
                for _ in range(2):linear._brmoe_int3_linear_impl(x,w,s,z,gs)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):result=linear._brmoe_int3_linear_impl(x,w,s,z,gs)
                for replay in (0,1):
                    x.copy_((torch.randn_like(x)*.1).half())
                    result.fill_(float('nan'));graph.replay()
                    got=result.float();ref=x.float()@deq.T
                    rel=float((got-ref).norm()/ref.norm().clamp_min(1e-8))
                    assert torch.isfinite(got).all() and rel<.003,(m,k,n,config,rel)
                    records.append(dict(m=m,k=k,n=n,config=config,replay=replay,relative_l2=rel))
        print('VERIFY SHAPE',k,n,'PASS',flush=True)
        del deq,q
    (args.out/'verify.json').write_text(json.dumps(records,indent=2))
    print('VERIFY PASS',len(records),max(r['relative_l2'] for r in records),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('phase',choices=['collect','profile','micro','moe','e2e','verify'])
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--input',type=Path)
    ap.add_argument('--batch-sizes',default='8,16,32,64,128')
    ap.add_argument('--configs',default='baseline,single_32_3,single_64_1,single_64_3')
    ap.add_argument('--linear-config',default='baseline')
    ap.add_argument('--quick',action='store_true')
    ap.add_argument('--prefill',action='store_true')
    ap.add_argument('--prefill-only',action='store_true',help='collect/profile: one output token and capture actual prefill shapes')
    ap.add_argument('--slice-small',action='store_true',help='micro only: slice real M8 inputs for M1/2/4 probes')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0',BRMOE_GROUPED_GEMV='0',
                      BRMOE_CUDA_FUSE='1',BRMOE_MOE_SMEM='legacy')
    args.save_linear=args.phase!='profile'
    metadata=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,triton=triton.__version__,
        linear_source=str(Path(linear.__file__).resolve()),
        linear_sha256=hashlib.sha256(Path(linear.__file__).read_bytes()).hexdigest(),
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    metadata['sources']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__).resolve(),ROOT/'tools/brmoe_int3_vllm/linear_tc.py',
                  ROOT/'tools/brmoe_int3_vllm/prefill.py',
                  ROOT/'tools/brmoe_int3_vllm/kernel.py',
                  ROOT/'BR-MoE/kernels/triton_int3/int3_moe/grouped_tc.py',
                  ROOT/'BR-MoE/kernels/triton_int3/int3_moe/kernel.py') if p.exists()}
    (args.out/f'metadata_{args.phase}_{args.linear_config}.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata),flush=True)
    if args.phase in ('collect','profile'):
        install(args.linear_config)
        collect(args)
    elif args.phase=='e2e':
        install(args.linear_config)
        args.config='fused4';args.cuda_cfg=None
        e2e(args)
    else:dict(collect=collect,micro=micro,moe=moe,verify=verify)[args.phase](args)


if __name__=='__main__':main()
