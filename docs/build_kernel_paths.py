"""Generate the offline kernel guide and three source-controlled SVG figures."""
from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parent
COLORS = dict(cyan='#22d3ee', green='#34d399', purple='#a78bfa',
              amber='#fbbf24', rose='#fb7185', slate='#94a3b8')


class Figure:
    def __init__(self, title, subtitle, height):
        self.height = height
        self.parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 {height}" role="img" aria-labelledby="title desc">
<title id="title">{escape(title)}</title><desc id="desc">{escape(subtitle)}</desc>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10" fill="#64748b"/></marker></defs>
<style>text{{font-family:system-ui,-apple-system,'Noto Sans CJK SC','Microsoft YaHei',sans-serif}} .body{{fill:#cbd5e1;font-size:16px}} .head{{fill:#f8fafc;font-size:20px;font-weight:650}} .small{{fill:#94a3b8;font-size:14px}} .arrow{{stroke:#64748b;stroke-width:2;fill:none;marker-end:url(#arrow)}} </style>
<rect width="1280" height="{height}" rx="18" fill="#020617"/>''']
        self.text(36, 48, title, size=29, color='#f8fafc', weight=700)
        self.text(36, 80, subtitle, size=16)

    def text(self, x, y, text, size=16, color='#94a3b8', weight=400):
        self.parts.append(f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" font-weight="{weight}">{escape(text)}</text>')

    def box(self, x, y, w, h, title, lines=(), color='cyan'):
        c = COLORS[color]
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="#0f172a" stroke="{c}" stroke-width="1.5"/>')
        self.text(x+20, y+31, title, 20, c, 650)
        for j, line in enumerate(lines):
            self.text(x+20, y+60+j*25, line, 16, '#cbd5e1')

    def arrow(self, points):
        d = 'M'+' L'.join(f'{x},{y}' for x, y in points)
        self.parts.append(f'<path d="{d}" class="arrow"/>')

    def done(self):
        return '\n'.join(self.parts+['</svg>'])


def overview():
    f = Figure('BR-MoE · 全 INT3 推理算子路径',
               '2026-09-26  ·  按实际分派顺序阅读  ·  权重 INT3，激活 FP16（W3A16）', 1140)
    f.box(36, 110, 1208, 79, 'vLLM V1 Engine → 量化插件 brmoe_int3',
          ['M = 本次调用的 token 行数；普通 decode 时通常等于 batch，prefill 时由分块与调度决定。'])
    f.arrow([(470,189),(470,223)])
    f.arrow([(1086,189),(1086,223)])
    f.box(36, 223, 860, 98, 'Routed MoE · 27 层 / 64 专家 / top-k = 6',
          ['输入 x[M,2048] + expert_ids[M,6] + weights[M,6]', '规整无效专家槽位，随后依次匹配下列条件。'])
    f.box(928, 223, 316, 98, 'Attention / shared 线性层',
          ['同样使用 INT3 权重', '独立调用 linear_method.py'], 'slate')
    f.text(56, 354, '从 ① 到 ④，命中第一条即执行对应路径', 16, '#7dd3fc', 600)
    rows = [
        (374, '① grouped GEMV 实验分支', ['GROUPED_GEMV=1 · sm_120 · M=4…16', '且融合 CUDA 未启用或不可用'],
         'Grouped GEMV · SIMT', ['专家内共享权重解码；独立实验路径', '排序 → 两次 GEMV → 激活 / 归约'], 'rose'),
        (506, '② CUDA 可用，且 M ≤ t', ['t：A100=2；5090 融合=2、原版=8', '其余架构当前 t=8'],
         'Routed GEMV · SIMT', ['每条路由独立计算，适用于极小 M', '由 CUDA wrapper 委托 Triton GEMV'], 'amber'),
        (638, '③ CUDA 可用，且 t < M ≤ 512', ['CUDA 扩展和 Marlin 权重副本均存在', 'FUSE=1 使用右侧融合流程'],
         'Marlin CUDA · Tensor Core', ['融合：align → gather → W13 → SiLU', '→ W2 → top-k reduce；关闭则原 wrapper'], 'green'),
        (770, '④ CUDA 不可用，或 M > 512', ['进入 Triton 自动分派', 'GEMV 阈值：sm_80=4、sm_120=8'],
         'Triton 自动路径', ['小 M 用 routed GEMV', '其余用 grouped GEMM / Tensor Core'], 'purple'),
    ]
    for y, title, lines, right, details, color in rows:
        f.box(36, y, 414, 112, title, lines, color)
        f.arrow([(450,y+56),(484,y+56)])
        f.box(484, y, 412, 112, right, details, color)
    f.arrow([(1086,321),(1086,388)])
    f.box(928, 388, 316, 112, 'M ≤ 8 → GEMV',
          ['复用 E=1、top-k=1 的路径', '没有跨专家路由'], 'amber')
    f.box(928, 542, 316, 112, 'M > 8 → Triton GEMM',
          ['K-major INT3 解包 + 张量核', '该分派不受 MoE 融合开关影响'], 'purple')
    f.arrow([(1244,321),(1262,321),(1262,598),(1244,598)])
    f.box(928, 718, 316, 164, '如何阅读开关',
          ['FUSE = BRMOE_CUDA_FUSE', 'GROUPED_GEMV =', 'BRMOE_GROUPED_GEMV', '两个开关当前均默认关闭。'], 'slate')
    f.box(36, 920, 1208, 168, '加载期：一次准备两种权重布局',
          ['Checkpoint N-major → repack_moe → Marlin B1 / B2 + 重排 scale / zero → CUDA 路径',
           'Checkpoint N-major → 转置 K-major → packed INT3 + scale / zero → Triton GEMV / GEMM',
           '两种布局不可混用；每次 decode 不重新打包。所有路由分组与有效长度均在 GPU 上更新。',
           '融合路径已在 5090 完成完整 MoE 与端到端验证；A100 融合复测仍在排队。'], 'slate')
    return f.done()


def fused():
    f = Figure('融合 CUDA MoE · 9 次 kernel launch',
               'BRMOE_CUDA_FUSE=1 · split-K=1 · 输入、激活及矩阵乘写出保持 FP16', 1080)
    f.text(40, 128, '执行顺序 / 一层完整 routed MoE', 18, '#34d399', 650)
    steps = [
        (154, 130, '01–04  Align + 反向路由表',
         ['count → scan/pad → scatter → block expert IDs',
          '每个专家补齐 16 行；scatter 同时写 route_pos',
          'STI: sorted_row → token；route_pos: route → sorted_row'], 'cyan'),
        (318, 97, '05  Gather + padding 清零',
         ['按 STI 读取 x → A_sorted[P,2048]（FP16）', '由 GPU metadata 跳过未使用的静态缓冲块'], 'green'),
        (449, 97, '06  W13 · Marlin INT3 grouped GEMM',
         ['A_sorted × W_gate/up → inter[P,2816]（FP16）', '每个 CTA 处理同一专家的 16 行 × N-tile'], 'green'),
        (580, 97, '07  SiLU × up + 转换',
         ['FP32 计算 SiLU(gate) × up → act[P,1408]（FP16）', '单 kernel；掩蔽 padding 和无效缓冲范围'], 'green'),
        (711, 97, '08  W2 · Marlin INT3 grouped GEMM',
         ['act × W_down → sorted_out[P,2048]（FP16）', '复用同一套专家分组和行顺序'], 'green'),
        (842, 122, '09  Gather top-k + 加权归约 + cast',
         ['route_pos 找回每个 token 的 6 条专家输出',
          'FP32 加权求和 → Y[M,2048]（FP16）',
          '每个输出元素只有一个写者；无需输出清零或 scatter 原子'], 'cyan'),
    ]
    for j, (y, h, title, lines, color) in enumerate(steps):
        f.box(36, y, 700, h, title, lines, color)
        if j: f.arrow([(386, steps[j-1][0]+steps[j-1][1]), (386,y)])
    f.box(774, 154, 470, 180, '三个索引空间',
          ['token：0 … M-1', 'route = token × 6 + top-k 槽位',
           'sorted_row：按专家排序后的 padded 行',
           'P = Σexpert ceil(route_count / 16) × 16',
           'P 由 GPU 计算；workspace 分配使用静态上界。'], 'slate')
    f.box(774, 368, 470, 168, '矩阵乘内部仍执行',
          ['cp.async 分级搬运 A 与 INT3 packed 权重',
           '寄存器解包 → (q − zero) × scale',
           'FP16 Tensor Core MMA / FP32 累加',
           '共享内存重排 fragment → FP16 写出'], 'green')
    f.box(774, 570, 470, 168, '融合省在哪里？',
          ['原 wrapper：23 次 → 当前：9 次 / 层',
           '合并 gather、激活、最终加权归约周边操作',
           '避免扫描大块未使用的静态缓冲空间',
           'split-K>1 会另增 FP32 清零与原子归约。'], 'cyan')
    f.box(774, 772, 470, 192, 'MoE 内部与全模型的瓶颈',
          ['27 层完整 MoE：3.815 → 1.781 ms',
           '同卡全 INT3 TPOT：9.859 → 8.743 ms',
           '融合后两次矩阵乘占 MoE GPU 时间约 77%',
           '全模型 bs128：共享 / attention 投影约 13.5 ms',
           '下一步：单专家 tile 与权重解码复用（待测）。'], 'amber')
    f.text(36, 1021, '测量：run_39129 / 39130 / 39139 / 39143；128 输入 / 128 输出，重复相同 prompt。', 16)
    f.text(36, 1049, '9 次是融合 MoE 核心的计数；不包含 vLLM gate/top-k、路由规整或其他模型层。', 16)
    return f.done()


def grouped():
    f = Figure('Grouped GEMV · 同一专家内共享权重解码',
               '独立 Triton SIMT 实验路径 · BRMOE_GROUPED_GEMV=1 · 5090 / M=4…16', 1080)
    steps = [
        (150, 124, '01  按专家分组（GPU sort）',
         ['key = expert_id × (M × top-k) + original_route',
          '每个专家拆成最多 ROWS 条路由的小组',
          '输出 SORTED / STARTS / EXPERTS / COUNT'], 'rose'),
        (308, 97, '02–03  清零累加缓冲',
         ['inter[M×6,2816]：FP32，保留原路由顺序', 'out[M,2048]：FP32，供 split-K / top-k 原子累加'], 'slate'),
        (439, 124, '04  Grouped GEMV · W13',
         ['一个工作 tile = 专家小组 × N-tile × K-split',
          '一次解包权重，广播给同一专家的 ROWS 个输入',
          'FP32 部分和 atomic_add → inter[original_route]'], 'rose'),
        (597, 97, '05  SiLU × up',
         ['按原路由顺序读取 gate / up', 'FP32 激活乘法 → act[M×6,1408]（FP16）'], 'green'),
        (728, 124, '06  Grouped GEMV · W2 + weighted scatter',
         ['复用分组；输入索引改为 original_route',
          '乘路由权重，再 atomic_add 到 out[token]',
          'W2 的 K-split = max(1, W13 K-split / 2)'], 'rose'),
        (886, 72, '07  FP32 → FP16，返回 Y[M,2048]', [], 'cyan'),
    ]
    for j, (y, h, title, lines, color) in enumerate(steps):
        f.box(36, y, 710, h, title, lines, color)
        if j: f.arrow([(391, steps[j-1][0]+steps[j-1][1]),(391,y)])
    f.box(784, 150, 460, 198, '权重共享示例 · ROWS=4',
          ['4 个 token 路由到同一专家 e',
           'x0 ─┐', 'x1 ─┼─ 共享解码 W[e] → 4 组累加器',
           'x2 ─┤', 'x3 ─┘',
           '减少重复权重加载 / 解码，增加寄存器压力。'], 'amber')
    f.box(784, 382, 460, 198, 'Persistent 调度',
          ['grid = min(静态任务上界, SM 数 × 4)',
           '每个 program 读取 GPU COUNT',
           'tile += grid_size，循环领取实际工作 tile',
           '有效组数改变时不需要 CPU 同步',
           '每次 CUDA Graph replay 都重新分组',
           '中间结果按原 route 存放，无显式 x gather。'], 'purple')
    f.box(784, 614, 460, 172, '为何大 batch 不优先选它？',
          ['SIMT 显式乘加，没有 Tensor Core MMA',
           'ROWS 增大带来更多累加器和寄存器压力',
           'split-K 还需清零和 FP32 原子加',
           '专家内有更多 token 时，应重新比较 grouped GEMM。'], 'amber')
    f.box(784, 820, 460, 138, '和融合 CUDA 的关系',
          ['两者是替代分支，不是前后串联',
           '融合 CUDA 可用且开关开启时优先',
           '当前 grouped GEMV 的阈值只校准了 5090 小 M。'], 'slate')
    f.text(36, 1022, '源码：BR-MoE/kernels/triton_int3/int3_moe/grouped_gemv.py · 7 次为 FP16 输出的常规完整算子路径。', 16)
    return f.done()


def main():
    figures = [overview(), fused(), grouped()]
    files = ['brmoe-kernel-paths.svg', 'brmoe-fused-moe.svg', 'brmoe-grouped-gemv.svg']
    labels = ['01 · 分派总览', '02 · 融合 CUDA MoE', '03 · Grouped GEMV']
    for file, fig in zip(files, figures):
        (OUT/file).write_text(fig+'\n')
    tabs = ''.join(f'<button role="tab" id="tab{i}" aria-controls="panel{i}" aria-selected="{str(i==0).lower()}" tabindex="{0 if i==0 else -1}" data-index="{i}">{label}</button>' for i,label in enumerate(labels))
    panels = ''.join(f'<section role="tabpanel" aria-labelledby="tab{i}" id="panel{i}"{ " hidden" if i else ""}>{fig}</section>' for i,fig in enumerate(figures))
    html = '''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>BR-MoE · Kernel 路径与融合流程</title>
<style>
:root{color-scheme:dark;font-family:system-ui,-apple-system,"Microsoft YaHei",sans-serif;color:#e2e8f0;background:#020617}
*{box-sizing:border-box}body{margin:0}header,main,footer{max-width:1440px;margin:auto;padding:24px 28px}header{padding-bottom:10px}h1{font-size:clamp(23px,3vw,34px);margin:0 0 10px}p{color:#94a3b8;line-height:1.7;margin:8px 0}a{color:#7dd3fc}code{color:#a7f3d0}nav,.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center}button,.download{border:1px solid #334155;border-radius:8px;background:#0f172a;color:#cbd5e1;padding:11px 16px;cursor:pointer;font:inherit;text-decoration:none}button[aria-selected=true]{border-color:#34d399;color:#34d399;background:#052e2b}button:focus-visible,a:focus-visible{outline:2px solid #22d3ee;outline-offset:4px}.controls{margin:16px 0;font-size:14px}.controls span{color:#94a3b8}#viewport{overflow:auto;border:1px solid #1e293b;border-radius:16px;background:#020617}section svg{display:block;width:100%;height:auto;min-width:900px}section[hidden]{display:none}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:22px}.card{border:1px solid #1e293b;border-radius:12px;padding:18px;background:#0f172a}.value{color:#34d399;font-size:25px;font-weight:650}.caption{font-size:14px}footer{padding-top:8px;font-size:14px}@media(max-width:750px){header,main,footer{padding:18px 14px}.cards{grid-template-columns:1fr}button,.download{padding:10px}nav{gap:6px}}
</style></head><body>
<header><h1>BR-MoE · Kernel 执行路径</h1><p>三张图读懂分派、融合和 grouped GEMV。全 INT3 attention + MoE，激活 FP16。更新于 2026-09-26。</p></header>
<main><nav role="tablist" aria-label="执行路径图">__TABS__</nav>
<div class="controls"><button id="minus" aria-label="缩小图形">−</button><button id="fit">适应宽度</button><button id="plus" aria-label="放大图形">＋</button><span id="zoom" aria-live="polite">100%</span><a class="download" id="download" download="brmoe-kernel-paths.svg">下载当前 SVG</a><span>小屏可横向滚动；原始 SVG 可无限缩放。</span></div>
<div id="viewport">__PANELS__</div>
<div class="cards"><article class="card"><div class="value">23 → 9 kernels</div><p class="caption">一层完整 MoE；融合 gather、激活和 top-k 归约。矩阵乘的 FP16 中间写出保持原精度。</p></article><article class="card"><div class="value">bs32 · −11.3% TPOT</div><p class="caption">5090：9.859 → 8.743 ms。128 输入 / 128 输出、同卡前后对照、重复相同 prompt。</p></article><article class="card"><div class="value">bs128 · 线性层 13.5 ms</div><p class="caption">共享专家 + attention 投影的分阶段事件采样；routed MoE 约 5.48 ms。后续重点是单专家 tile 与权重解码复用。采样区间不能直接拼加为干净 TPOT。</p></article></div>
</main><footer><p>当前融合开关：<code>BRMOE_CUDA_FUSE=1</code>，默认关闭。5090 已验证，A100 融合复测仍排队。</p><p><a href="moe_large_batch_20260926.md">大 batch 实验与瓶颈</a> · <a href="moe_cuda_fusion_20260926.md">完整融合实验报告</a> · <a href="grouped_gemv_study_20260926.md">Grouped GEMV 实验报告</a> · <a href="grouped_gemv_explainer.html">Grouped GEMV 交互讲解</a> · <a href="../README.md">README</a></p><p>源码与图保持同仓库；运行 <code>python docs/build_kernel_paths.py</code> 可重新生成。HTML 内嵌全部图形，无网络依赖。</p></footer>
<script>
const tabs=[...document.querySelectorAll('[role=tab]')],panels=[...document.querySelectorAll('[role=tabpanel]')];
const names=['brmoe-kernel-paths.svg','brmoe-fused-moe.svg','brmoe-grouped-gemv.svg'];let current=0,scale=1,url;
function size(){panels[current].querySelector('svg').style.width=(scale*100)+'%';document.querySelector('#zoom').textContent=Math.round(scale*100)+'%';}
function choose(index){current=index;tabs.forEach((t,i)=>{t.setAttribute('aria-selected',i===index);t.tabIndex=i===index?0:-1;panels[i].hidden=i!==index});scale=1;size();if(url)URL.revokeObjectURL(url);const svg=panels[index].querySelector('svg').cloneNode(true);svg.removeAttribute('style');url=URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(svg)],{type:'image/svg+xml'}));const link=document.querySelector('#download');link.href=url;link.download=names[index];document.querySelector('#viewport').scrollTo(0,0);}
tabs.forEach((t,i)=>{t.addEventListener('click',()=>choose(i));t.addEventListener('keydown',e=>{let next;if(e.key==='ArrowRight')next=(i+1)%tabs.length;if(e.key==='ArrowLeft')next=(i+tabs.length-1)%tabs.length;if(e.key==='Home')next=0;if(e.key==='End')next=tabs.length-1;if(next!==undefined){e.preventDefault();choose(next);tabs[next].focus();}})});
document.querySelector('#plus').onclick=()=>{scale=Math.min(3,scale+.25);size()};document.querySelector('#minus').onclick=()=>{scale=Math.max(.75,scale-.25);size()};document.querySelector('#fit').onclick=()=>{scale=1;size()};choose(0);
</script></body></html>
'''.replace('__TABS__',tabs).replace('__PANELS__',panels)
    # SVG IDs are local in standalone files; namespace them in the combined page.
    for i, fig in enumerate(figures):
        html = html.replace(fig, fig.replace('id="title"',f'id="title{i}"').replace('id="desc"',f'id="desc{i}"').replace('title desc',f'title{i} desc{i}').replace('id="arrow"',f'id="arrow{i}"').replace('url(#arrow)',f'url(#arrow{i})').replace('.arrow{',f'.arrow{i}'+'{').replace('class="arrow"',f'class="arrow{i}"'))
    (OUT/'brmoe-kernel-paths.html').write_text(html)


if __name__ == '__main__':
    main()
