"""Small decode: real input GEMV, complete MoE, counters and clean end-to-end."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
for sub in ('tools','bench','BR-MoE/kernels','BR-MoE/kernels/triton_int3'):
    sys.path.insert(0,str(ROOT/sub))
import torch
import triton
import int3_linear_study as linear_study
from brmoe_int3_vllm import linear_method as linear
from brmoe_int3_vllm import moe_method as moe_method
from brmoe_int3_vllm import kernel as plugin
from int3_moe.gemv_reduce import linear_gemv, fused_moe_gemv
from full_int3_study import graph_us
from moe_fuse_study import weights as load_weights

OLD_LINEAR=linear._brmoe_int3_linear_impl
OLD_MOE=moe_method.brmoe_int3_moe
OLD_SANITIZE=plugin._sanitize_routing

def params(c):
    _,bn,g,s,w=c.split('_')
    return dict(block_n=int(bn),groups=int(g),splits=int(s),warps=int(w))


def install(config, limit=8):
    linear._brmoe_int3_linear_impl=OLD_LINEAR
    moe_method.brmoe_int3_moe=OLD_MOE
    plugin._sanitize_routing=OLD_SANITIZE
    os.environ['BRMOE_SMALL_DECODE_BACKEND']='auto' if config=='production' else 'legacy'
    if config=='production':return
    for c in config.split('+'):
        route_limit=limit
        if c=='routed2':
            c='v_64_4_8_2'
            route_limit=2
        if c=='baseline':continue
        if c=='san':
            from int3_moe.gemv_reduce import sanitize_routing
            def sanitize(ids,tw):
                return sanitize_routing(ids,tw) if ids.shape[0]<=8 else OLD_SANITIZE(ids,tw)
            plugin._sanitize_routing=sanitize
            continue
        if c in ('adaptive','fine'):
            def adaptive(x,w,s,z,gs,_fine=c=='fine'):
                m=x.shape[0]
                if m<=min(limit,2 if _fine else 4) and x.dtype==torch.float16 and gs==64:
                    return linear_gemv(x,w,s,z,gs,block_n=64,groups=1 if _fine else (4 if m==1 else 2),
                                       splits=32 if _fine else 16,warps=2,half2=True,rows=1 if m==1 else 2)
                return OLD_LINEAR(x,w,s,z,gs)
            linear._brmoe_int3_linear_impl=adaptive
            continue
        kw=params(c)
        if c.startswith(('p_','h_','b_')):
            kw['half2']=not c.startswith('p_')
            kw['rows']=2 if c.startswith('b_') else 1
            def gemv(x,w,s,z,gs,_kw=kw):
                if x.shape[0]<=limit and x.dtype==torch.float16 and gs==64:
                    return linear_gemv(x,w,s,z,gs,**_kw)
                return OLD_LINEAR(x,w,s,z,gs)
            linear._brmoe_int3_linear_impl=gemv
        elif c.startswith(('r_','v_','g_')):
            kw['half2']=not c.startswith('r_')
            kw['rows']=2 if c.startswith('g_') else 1
            def moe(x,tw,ids,layer,gs,out_dtype=None,_kw=kw,_limit=route_limit):
                if x.shape[0]<=_limit and gs==64 and x.dtype==torch.float16 and getattr(layer,'w_transposed',False):
                    ids,tw=plugin._sanitize_routing(ids,tw)
                    return fused_moe_gemv(x,tw,ids,plugin.build_packed(layer,gs),out_dtype=out_dtype or x.dtype,**_kw)
                return OLD_MOE(x,tw,ids,layer,gs,out_dtype)
            moe_method.brmoe_int3_moe=moe
        else:raise ValueError(config)


def micro(args):
    data=torch.load(args.input/'linear_inputs.pt',weights_only=True)
    cpu=torch.load(args.input/'linear_weights.pt',weights_only=True)
    if args.quick:
        reps={}
        for name,v in cpu.items():reps.setdefault((v[0].shape,v[1].shape),name)
        cpu={n:cpu[n] for n in reps.values()}
    weights={n:tuple(v.cuda() if isinstance(v,torch.Tensor) else v for v in values) for n,values in cpu.items()}
    result=[]
    for m in map(int,args.m_values.split(',')):
        inputs={s['layer']:s['x'] for s in data if s['m']==m and s['batch']==m and s['layer'] in weights}
        if not inputs and args.slice_small:
            inputs={s['layer']:s['x'][:m] for s in data if s['m']==8 and s['batch']==8 and s['layer'] in weights}
        assert inputs,(m,'missing real input')
        inputs={n:x.cuda() for n,x in sorted(inputs.items())}
        install('baseline')
        ref={n:OLD_LINEAR(x,*weights[n]).clone().float() for n,x in inputs.items()}
        for c in args.configs.split(','):
            install(c,args.limit)
            errors=[]
            for n,x in inputs.items():
                v=linear._brmoe_int3_linear_impl(x,*weights[n]).float();r=ref[n]
                err=float((v-r).norm()/r.norm().clamp_min(1e-8));assert torch.isfinite(v).all() and err<.003,(m,c,n,err)
                errors.append(err)
            for cat in ['attention','shared','first_mlp','all']:
                names=[n for n in inputs if cat=='all' or linear_study.category(n)==cat]
                if not names:continue
                def fn():
                    for n in names:linear._brmoe_int3_linear_impl(inputs[n],*weights[n])
                rec=dict(m=m,config=c,category=cat,layers=len(names),us=graph_us(fn,repeats=args.repeats),max_relative_l2=max(errors))
                result.append(rec);print(json.dumps(rec),flush=True)
                (args.out/'micro.json').write_text(json.dumps(result,indent=2))
                if args.trace and cat=='all':trace(fn,args.out/f'linear_m{m}_{c}.json')


def trace(fn,path):
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):fn()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
        g.replay();torch.cuda.synchronize()
    p.export_chrome_trace(str(path))


def moe(args):
    from types import SimpleNamespace
    import vllm._custom_ops
    data=torch.load(args.input/'real_inputs.pt',weights_only=True)
    names=sorted({s['layer'] for s in data});names=names[:1] if args.quick else names
    packed={n:load_weights(n) for n in names}
    layers={}
    for n,(p,pk) in packed.items():
        layers[n]=SimpleNamespace(w13_q=p['w13_q'],w13_s=p['s13'],w13_z=p['z13'],w2_q=p['w2_q'],w2_s=p['s2'],w2_z=p['z2'],brmoe_cuda_packed=pk,w_transposed=True)
    cpu=torch.load(args.input/'linear_weights.pt',weights_only=True)
    shared={n:tuple(v.cuda() if isinstance(v,torch.Tensor) else v for v in vals) for n,vals in cpu.items() if '.shared_experts.' in n}
    result=[]
    for m in map(int,args.m_values.split(',')):
        inp={s['layer']:s for s in data if s['batch']==m and s['m']==m and s['layer'] in names};assert len(inp)==len(names),(m,len(inp))
        inp={n:{**s,**{k:s[k].cuda() for k in ['x','weights','ids']}} for n,s in inp.items()}
        def one(n,scope):
            s=inp[n];p=packed[n][0]
            out=moe_method.brmoe_int3_moe(s['x'],s['weights'],s['ids'],layers[n],p['group_size'])
            if scope=='routed':return out
            pre=n.removesuffix('.experts')+'.shared_experts.'
            h=linear._brmoe_int3_linear_impl(s['x'],*shared[pre+'gate_up_proj'])
            act=torch.empty(m,h.shape[-1]//2,device='cuda',dtype=torch.float16)
            torch.ops._C.silu_and_mul(act,h)
            return out+linear._brmoe_int3_linear_impl(act,*shared[pre+'down_proj'])
        install('baseline');refs={scope:{n:one(n,scope).clone().float() for n in names} for scope in ['routed','complete']}
        for c in args.configs.split(','):
            install(c,args.limit)
            for scope in ['routed','complete']:
                errors=[]
                for n in names:
                    got=one(n,scope).float();ref=refs[scope][n]
                    err=float((got-ref).norm()/ref.norm().clamp_min(1e-8));assert torch.isfinite(got).all() and err<.003,(m,c,n,err)
                    errors.append(err)
                def fn():
                    for n in names:one(n,scope)
                rec=dict(m=m,config=c,scope=scope,layers=len(names),us=graph_us(fn,repeats=args.repeats),max_relative_l2=max(errors))
                result.append(rec);print(json.dumps(rec),flush=True)
                (args.out/'moe.json').write_text(json.dumps(result,indent=2))
                if args.trace and scope=='routed':trace(fn,args.out/f'moe_m{m}_{c}.json')


def e2e(args):
    old_install=linear_study.install
    def tuned_install(config):
        old_install(config)
        if config=='production':install(args.config,args.limit)
    linear_study.install=tuned_install
    sys.argv=['a100_full_int3_study.py','--case','fp16' if args.config=='fp16' else 'production','--out',str(args.out),
              '--batch-sizes',args.m_values,'--repeat',str(args.repeats)]
    # The shared harness checks the identity of the linear implementation. For
    # experimental hooks use a named case, leaving formal production checks intact.
    if args.config not in ('baseline','fp16','production'):
        sys.argv[2]='small_decode_candidate'
        def experiment_install(config):
            if config=='small_decode_candidate':
                old_install('production');install(args.config,args.limit)
            else:old_install(config)
        linear_study.install=experiment_install
    runpy.run_path(str(ROOT/'bench/a100_full_int3_study.py'),run_name='__main__')


def verify(args):
    """Reuse the independent FP32 reference and changed-input graph checks."""
    original=linear_study.install
    def checked(config):
        original('production')
        install(config,args.limit)
    linear_study.install=checked
    args.prefill=False
    linear_study.verify(args)


def verify_moe(args):
    from full_int3_study import load_packed
    from int3_moe.packing import pack_int3,unpack_int3
    from int3_moe.gemv_reduce import _WORKSPACE,sanitize_routing
    torch.manual_seed(20260926)
    torch.backends.cuda.matmul.allow_tf32=False
    cases=[('real',load_packed('model.layers.1.mlp.experts','cuda'))]
    for k,i,gs in [(192,96,32),(256,128,64),(256,128,128)]:
        p=dict(group_size=gs,layout='int3',w_transposed=True);e=7
        for tag,n,red in [('13',2*i,k),('2',k,i)]:
            q=torch.randint(0,8,(e*n,red),device='cuda',dtype=torch.int32)
            p['w'+tag+'_q']=pack_int3(q,transposed=True).reshape(red//32*3,e,n).permute(1,0,2).contiguous()
            p['s'+tag]=(torch.rand(e,red//gs,n,device='cuda')*.02+.005).half()
            p['z'+tag]=(torch.rand(e,red//gs,n,device='cuda')*5+.5).half()
        cases.append((f'tail_gs{gs}',p))
    records=[]
    for name,p in cases:
        e,_,two_i=p['w13_q'].shape;k=p['w2_q'].shape[-1];i=two_i//2;gs=p['group_size'];topk=6 if e==64 else 3
        def dequant(tag,red,n):
            q=unpack_int3(p['w'+tag+'_q'].transpose(1,2).reshape(e*n,-1),red,transposed=False).reshape(e,n,red).half()
            return ((q-p['z'+tag].repeat_interleave(gs,1).transpose(1,2)).half()*p['s'+tag].repeat_interleave(gs,1).transpose(1,2)).half().float()
        w13,w2=dequant('13',k,two_i),dequant('2',i,k)
        for m in map(int,args.m_values.split(',')):
            x=(torch.randn(m,k*2,device='cuda')*.1).half()[:,::2]
            tw=torch.empty(m,topk*2,device='cuda')[:,::2]
            ids=torch.empty(m,topk*2,device='cuda',dtype=torch.int32)[:,::2]
            ids.copy_(torch.arange(topk,device='cuda'));tw.fill_(1/topk)
            for config in args.configs.split(','):
                kw=params(config);kw.update(half2=not config.startswith('r_'),rows=2 if config.startswith('g_') else 1)
                def fn():
                    clean_ids,clean_tw=sanitize_routing(ids,tw)
                    return fused_moe_gemv(x,clean_tw,clean_ids,p,**kw)
                for _ in range(2):fn()
                g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):out=fn()
                for pattern in ['hot','random','invalid','all_invalid','zero_weight','hot_again']:
                    x.copy_((torch.randn_like(x)*.1).half())
                    ids.copy_(torch.rand(m,e,device='cuda').argsort(1)[:,:topk] if pattern=='random' else torch.arange(topk,device='cuda'))
                    tw.copy_(torch.softmax(torch.randn_like(tw),-1))
                    if pattern=='invalid':ids[:,::2]=-1
                    if pattern=='all_invalid':ids.fill_(-1)
                    if pattern=='zero_weight':tw.zero_()
                    for value in _WORKSPACE.values():
                        if isinstance(value,torch.Tensor) and value.is_floating_point():value.fill_(float('nan'))
                        if isinstance(value,tuple):
                            for v in value:v.fill_(-999)
                    out.fill_(float('nan'));g.replay();got=out.clone().float()
                    gold=torch.zeros(m,k,device='cuda',dtype=torch.float32)
                    for ex in range(e):
                        row,slot=torch.where(ids==ex)
                        if not row.numel():continue
                        h=x[row].float()@w13[ex].T
                        act=(h[:,:i]*torch.sigmoid(h[:,:i])*h[:,i:]).half().float()
                        val=(act@w2[ex].T)*tw[row,slot,None]
                        gold.index_add_(0,row,val)
                    gold=gold.half().float()
                    rel=float((got-gold).norm()/gold.norm().clamp_min(1e-8))
                    assert torch.isfinite(got).all() and rel<.003,(name,m,config,pattern,rel)
                    records.append(dict(shape=name,m=m,config=config,routing=pattern,relative_l2=rel))
                print('VERIFY MOE',name,m,config,'PASS',flush=True)
                (args.out/'verify_moe.json').write_text(json.dumps(records,indent=2))
        del w13,w2
    print('VERIFY MOE PASS',len(records),max(r['relative_l2'] for r in records),flush=True)


def counters(args):
    cpu=torch.load(args.input/'linear_weights.pt',weights_only=True)
    name='model.layers.1.self_attn.o_proj'
    w,s,z,gs=(v.cuda() if isinstance(v,torch.Tensor) else v for v in cpu[name])
    x=torch.randn(1,w.shape[0]//3*32,device='cuda',dtype=torch.float16)
    install(args.config,args.limit)
    def fn():return linear._brmoe_int3_linear_impl(x,w,s,z,gs)
    for _ in range(4):fn()
    torch.cuda.synchronize()
    # Flush L2 outside the measured range to model weights revisited after the
    # rest of the model. Nsight timings are not used as end-to-end latency.
    trash=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8)
    trash.zero_();torch.cuda.synchronize()
    torch.cuda.nvtx.range_push('measure')
    fn();torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


def verify_dispatch(args):
    from types import SimpleNamespace
    p,pk=load_weights('model.layers.1.mlp.experts')
    layer=SimpleNamespace(w13_q=p['w13_q'],w13_s=p['s13'],w13_z=p['z13'],
                          w2_q=p['w2_q'],w2_s=p['s2'],w2_z=p['z2'],
                          brmoe_cuda_packed=pk,w_transposed=True)
    torch.manual_seed(20260926)
    records=[]
    for m in map(int,args.m_values.split(',')):
        x=(torch.randn(m,4096,device='cuda')*.1).half()[:,::2]
        ids=torch.empty(m,12,device='cuda',dtype=torch.int32)[:,::2]
        tw=torch.empty(m,12,device='cuda')[:,::2]
        ids.copy_(torch.arange(6,device='cuda'));tw.fill_(1/6)
        for dtype in (torch.float16,torch.float32):
            graphs={};outputs={}
            for mode in ['baseline','production']:
                install(mode)
                # The legacy GEMV assumes contiguous activations. Compare
                # identical values while exercising the new path's strides;
                # the graph captures the reference copy, so replays update it.
                def fn():return OLD_MOE(x.contiguous() if mode=='baseline' else x,tw,ids,layer,64,dtype)
                for _ in range(2):fn()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):outputs[mode]=fn()
                graphs[mode]=graph
            for pattern in ['hot','random','invalid','all_invalid','zero_weight','hot_again']:
                x.copy_((torch.randn_like(x)*.1).half())
                ids.copy_(torch.rand(m,64,device='cuda').argsort(1)[:,:6] if pattern=='random' else torch.arange(6,device='cuda'))
                tw.copy_(torch.softmax(torch.randn_like(tw),-1))
                if pattern=='invalid':ids[:,::2]=-1
                if pattern=='all_invalid':ids.fill_(-1)
                if pattern=='zero_weight':tw.zero_()
                original_ids=ids.clone();original_weights=tw.clone()
                observed={}
                for mode in ['baseline','production']:
                    outputs[mode].fill_(float('nan'));graphs[mode].replay()
                    # Legacy CUDA outputs can alias a shared workspace across
                    # the two graphs. Preserve each result before replaying the
                    # other graph, including when the requested dtype is FP32.
                    observed[mode]=outputs[mode].float().clone()
                a,b=observed['baseline'],observed['production']
                err=float((a-b).norm()/a.norm().clamp_min(1e-8))
                assert torch.isfinite(b).all() and err<.003,(m,dtype,pattern,err)
                assert torch.equal(ids,original_ids) and torch.equal(tw,original_weights),'route inputs mutated'
                records.append(dict(m=m,dtype=str(dtype),pattern=pattern,relative_l2=err))
            print('VERIFY DISPATCH',m,str(dtype),'PASS',flush=True)
            (args.out/'verify_dispatch.json').write_text(json.dumps(records,indent=2))
    print('VERIFY DISPATCH PASS',len(records),max(r['relative_l2'] for r in records),flush=True)


def model_profile(args):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from full_int3_study import MODEL,TOKENIZER
    from vllm_perf import make_prompt
    install(args.config,args.limit)
    tok=AutoTokenizer.from_pretrained(TOKENIZER,trust_remote_code=True)
    prompt=make_prompt(tok,128)
    llm=LLM(model=str(MODEL),tokenizer=str(TOKENIZER),tokenizer_mode='hf',
            quantization='brmoe_int3',dtype='float16',trust_remote_code=True,
            max_model_len=768,max_num_batched_tokens=2048,max_num_seqs=512,
            gpu_memory_utilization=.9,enable_prefix_caching=False,disable_log_stats=True)
    for m in map(int,args.m_values.split(',')):
        sp=SamplingParams(max_tokens=16,min_tokens=16,temperature=0,ignore_eos=True)
        llm.generate([prompt]*m,sp,use_tqdm=False)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            llm.generate([prompt]*m,sp,use_tqdm=False)
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(args.out/f'model_bs{m}.json'))
        print('PROFILE',m,'16 output tokens',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['micro','moe','e2e','verify','verify_moe','verify_dispatch','counters','model_profile'])
    ap.add_argument('--input',type=Path);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--configs',default='baseline,p_64_4_16_2,p_64_4_8_2,p_128_4_16_2,p_32_4_16_2')
    ap.add_argument('--config',default='baseline');ap.add_argument('--m-values',default='1,2,4,8')
    ap.add_argument('--limit',type=int,default=8);ap.add_argument('--repeats',type=int,default=5)
    ap.add_argument('--quick',action='store_true');ap.add_argument('--slice-small',action='store_true');ap.add_argument('--trace',action='store_true')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    os.environ.update(BRMOE_LINEAR_BACKEND='auto',BRMOE_PREFILL_BACKEND='auto',BRMOE_CUDA_FUSE='1',BRMOE_GROUPED_GEMV='0',BRMOE_MOE_SMEM='legacy',VLLM_ENABLE_V1_MULTIPROCESSING='0')
    assert torch.cuda.get_device_capability()==(8,0)
    linear_study.install('production')
    files=[Path(__file__),ROOT/'BR-MoE/kernels/triton_int3/int3_moe/gemv_reduce.py',ROOT/'tools/brmoe_int3_vllm/linear_method.py',ROOT/'tools/brmoe_int3_vllm/kernel.py',ROOT/'tools/brmoe_int3_vllm/small_decode.py']
    meta=dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},gpu=torch.cuda.get_device_name(),torch=torch.__version__,triton=triton.__version__,sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (args.out/'metadata.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
    globals()[args.phase](args)

if __name__=='__main__':main()
