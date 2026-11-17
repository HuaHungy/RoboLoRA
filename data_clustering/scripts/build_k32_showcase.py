"""Build the presentation page for the final K=32 clustering review.

The page is intentionally self-contained apart from the locally exported GIF/JPG
assets. It reads only audited JSON results and never recomputes clustering.
"""

from __future__ import annotations

import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/k32_lambda003_calibrated5_showcase"
REVIEW = OUT / "review_metrics.json"
SUBSET = ROOT / "reports/five_task_subset20_lambda003_calibrated5_k32/summary.json"
FINAL_METRICS = (
    ROOT
    / "runs/mlp_temporal_20261109_230731_563725/evaluation"
    / "lambda0_03__calibrated5/k32/metrics.json"
)

SELECTED_SEVEN = (8, 24, 3, 6, 9, 31, 23)
AUTO_TOP_FIVE = (8, 24, 3, 6, 9)
TASKS = (
    ("adjust_bottle", "调整瓶子"),
    ("click_alarmclock", "点击闹钟"),
    ("place_can_basket", "罐子入篮"),
    ("place_container_plate", "容器放盘"),
    ("rotate_qrcode", "旋转二维码"),
)

K_COMPARISON = [
    {"k": 16, "silhouette": 0.2127, "ari": 0.8594, "position": 0.3011,
     "rotation": 0.2226, "gripper_delta": 0.5835, "size": "357–2877"},
    {"k": 32, "silhouette": 0.2465, "ari": 0.8718, "position": 0.4645,
     "rotation": 0.3699, "gripper_delta": 0.7163, "size": "144–2080"},
    {"k": 48, "silhouette": 0.2234, "ari": 0.7524, "position": 0.5292,
     "rotation": 0.4487, "gripper_delta": 0.7735, "size": "72–1499"},
]


def load(path: Path):
    return json.loads(path.read_text())


def safe_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def main() -> None:
    clusters = load(REVIEW)
    subset = load(SUBSET)
    final_metrics = load(FINAL_METRICS)
    ranking = subset["ranking"]
    rank_map = {int(row["cluster_id"]): row for row in ranking}

    task_totals = {name: int(subset["per_task"][name]["windows"]) for name, _ in TASKS}
    selected_task_counts = {
        name: sum(int(rank_map[c][f"{name}_windows"]) for c in SELECTED_SEVEN)
        for name, _ in TASKS
    }
    selected_total = sum(selected_task_counts.values())
    selected_balanced = sum(
        selected_task_counts[name] / task_totals[name] for name, _ in TASKS
    ) / len(TASKS)

    payload = {
        "clusters": clusters,
        "ranking": ranking,
        "selectedSeven": SELECTED_SEVEN,
        "autoTopFive": AUTO_TOP_FIVE,
        "tasks": [{"id": task, "label": label, "total": task_totals[task]}
                  for task, label in TASKS],
        "selectedTaskCounts": selected_task_counts,
        "selectedTotal": selected_total,
        "selectedBalanced": selected_balanced,
        "kComparison": K_COMPARISON,
        "finalMetrics": final_metrics,
    }

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark">
<title>RoboTwin 动作聚类 · K=32 结果展示</title>
<style>
:root{{--bg:#071019;--surface:#0e1a25;--surface2:#142332;--line:#284055;--text:#ecf4fa;--muted:#98adbc;--cyan:#48d7d0;--blue:#61a9ff;--amber:#ffbd59;--green:#6be0a4;--red:#ff7d7d;--shadow:0 18px 50px rgba(0,0,0,.28)}}
*{{box-sizing:border-box}}
html{{scroll-behavior:smooth}}
body{{margin:0;background:radial-gradient(circle at 78% -10%,#173752 0,transparent 32%),var(--bg);color:var(--text);font-family:Inter,"Noto Sans SC","Microsoft YaHei",system-ui,sans-serif;line-height:1.55}}
a{{color:inherit;text-decoration:none}}
button,select{{font:inherit}}
.shell{{max-width:1440px;margin:auto;padding:0 34px 72px}}
.topbar{{height:62px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid rgba(152,173,188,.18);position:sticky;top:0;background:rgba(7,16,25,.9);backdrop-filter:blur(14px);z-index:20}}
.brand{{display:flex;align-items:center;gap:10px;font-weight:700;letter-spacing:.03em}}
.brand i{{display:block;width:10px;height:10px;background:var(--cyan);box-shadow:0 0 18px var(--cyan);border-radius:50%}}
.nav{{display:flex;gap:24px;color:var(--muted);font-size:14px}}
.nav a:hover{{color:var(--text)}}
.hero{{padding:78px 0 54px;display:grid;grid-template-columns:1.35fr .65fr;gap:48px;align-items:end}}
.eyebrow{{color:var(--cyan);font:600 12px/1.2 ui-monospace,monospace;letter-spacing:.16em;text-transform:uppercase}}
h1{{font-size:clamp(42px,6vw,82px);line-height:.98;margin:18px 0 24px;letter-spacing:-.045em;max-width:920px}}
.lead{{font-size:19px;color:#bed0dc;max-width:780px;margin:0}}
.hero-meta{{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line);border:1px solid var(--line)}}
.hero-meta div{{background:var(--surface);padding:22px}}
.hero-meta b{{font-size:28px;display:block;line-height:1.1}}
.hero-meta span{{color:var(--muted);font-size:12px}}
.section{{padding:54px 0;border-top:1px solid rgba(152,173,188,.18)}}
.section-head{{display:grid;grid-template-columns:90px minmax(0,1fr);gap:24px;margin-bottom:32px}}
.section-no{{color:var(--cyan);font:600 13px ui-monospace,monospace;padding-top:8px}}
h2{{font-size:36px;line-height:1.15;margin:0 0 12px;letter-spacing:-.025em}}
.section-head p{{color:var(--muted);max-width:850px;margin:0}}
.pipeline{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}}
.stage{{position:relative;background:linear-gradient(145deg,var(--surface2),var(--surface));padding:22px;min-height:190px;border:1px solid var(--line)}}
.stage:not(:last-child):after{{content:"→";position:absolute;right:-19px;top:50%;z-index:2;width:25px;height:25px;text-align:center;background:var(--bg);color:var(--cyan)}}
.stage small{{color:var(--cyan);font:600 11px ui-monospace,monospace}}
.stage h3{{font-size:18px;margin:12px 0 8px}}
.stage p{{font-size:13px;color:var(--muted);margin:0}}
.equation{{margin-top:14px;color:#d9e8f2;font:12px ui-monospace,monospace}}
.k-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}
.k-card{{background:var(--surface);border:1px solid var(--line);padding:24px;position:relative}}
.k-card.chosen{{border-color:var(--cyan);box-shadow:inset 0 3px 0 var(--cyan)}}
.k-title{{display:flex;align-items:center;justify-content:space-between;margin-bottom:20px}}
.k-title b{{font-size:32px}}
.tag{{font-size:11px;padding:5px 8px;border:1px solid var(--line);color:var(--muted)}}
.chosen .tag{{color:var(--cyan);border-color:var(--cyan)}}
.metric{{display:grid;grid-template-columns:108px 1fr 52px;gap:10px;align-items:center;margin:12px 0;font-size:12px}}
.metric span:first-child{{color:var(--muted)}}
.track{{height:5px;background:#213444;overflow:hidden}}
.track i{{height:100%;display:block;background:var(--blue)}}
.chosen .track i{{background:var(--cyan)}}
.metric strong{{text-align:right;font:600 12px ui-monospace,monospace}}
.decision{{margin-top:18px;padding:18px 20px;border-left:3px solid var(--amber);background:rgba(255,189,89,.08);color:#d8e6ef;font-size:14px}}
.facts{{display:grid;grid-template-columns:repeat(5,1fr);gap:1px;background:var(--line);border:1px solid var(--line);margin-top:18px}}
.fact{{background:var(--surface);padding:18px}}
.fact strong{{display:block;font-size:22px}}
.fact span{{font-size:11px;color:var(--muted)}}
.gallery-tools{{display:flex;gap:10px;align-items:center;justify-content:space-between;flex-wrap:wrap;margin-bottom:18px}}
.filters{{display:flex;gap:8px;flex-wrap:wrap}}
.filters button,.gallery-tools select{{border:1px solid var(--line);background:var(--surface);color:var(--text);padding:9px 13px;cursor:pointer}}
.filters button.active{{background:var(--cyan);border-color:var(--cyan);color:#05201f;font-weight:700}}
.cluster-grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}}
.cluster-card{{background:var(--surface);border:1px solid var(--line);overflow:hidden;transition:transform .18s,border-color .18s}}
.cluster-card:hover{{transform:translateY(-3px);border-color:#52728b}}
.cluster-card.selected{{border-color:rgba(72,215,208,.75);box-shadow:inset 0 3px 0 var(--cyan)}}
.cluster-preview{{aspect-ratio:16/7;overflow:hidden;background:#09131c;position:relative}}
.cluster-preview img{{width:100%;height:100%;object-fit:cover;display:block}}
.cluster-preview:after{{content:"GIF × 4";position:absolute;right:9px;bottom:8px;background:rgba(5,14,22,.82);padding:4px 7px;font:10px ui-monospace,monospace;color:#dceaf3}}
.cluster-body{{padding:15px}}
.cluster-line{{display:flex;align-items:center;justify-content:space-between;gap:10px}}
.cluster-id{{font:700 23px ui-monospace,monospace}}
.selected-mark{{font-size:10px;color:var(--cyan);border:1px solid var(--cyan);padding:3px 6px}}
.cluster-stats{{font-size:12px;color:var(--muted);margin:8px 0 12px}}
.micro{{display:flex;gap:6px;flex-wrap:wrap;font-size:10px;color:#b8cbd7}}
.micro span{{background:var(--surface2);padding:4px 6px}}
.open-cluster{{width:100%;margin-top:13px;padding:8px;border:1px solid var(--line);background:transparent;color:var(--text);cursor:pointer}}
.open-cluster:hover{{border-color:var(--cyan);color:var(--cyan)}}
.selection-layout{{display:grid;grid-template-columns:.8fr 1.2fr;gap:28px}}
.flow{{display:grid;gap:10px}}
.flow-step{{display:grid;grid-template-columns:48px 1fr;gap:13px;align-items:start;background:var(--surface);border:1px solid var(--line);padding:16px}}
.flow-step b{{width:36px;height:36px;display:grid;place-items:center;border:1px solid var(--cyan);color:var(--cyan);font:600 12px ui-monospace,monospace}}
.flow-step h3{{margin:0 0 4px;font-size:16px}}
.flow-step p{{margin:0;color:var(--muted);font-size:12px}}
.ranking{{background:var(--surface);border:1px solid var(--line);padding:18px}}
.ranking h3{{margin:0 0 16px;font-size:16px}}
.rank-row{{display:grid;grid-template-columns:34px 44px 1fr 62px 54px;gap:9px;align-items:center;margin:7px 0;font-size:11px;color:var(--muted)}}
.rank-row.selected{{color:var(--text)}}
.rank-row b{{font:600 12px ui-monospace,monospace}}
.rankbar{{height:7px;background:#213444}}
.rankbar i{{display:block;height:100%;background:var(--blue)}}
.rank-row.selected .rankbar i{{background:var(--cyan)}}
.matrix-wrap{{overflow:auto;margin-top:24px;border:1px solid var(--line)}}
.matrix{{width:100%;border-collapse:collapse;min-width:820px;background:var(--surface)}}
.matrix th,.matrix td{{padding:12px 11px;border-bottom:1px solid var(--line);text-align:center;font-size:12px}}
.matrix th:first-child,.matrix td:first-child{{text-align:left;position:sticky;left:0;background:var(--surface);z-index:1}}
.matrix th{{color:var(--muted);font-weight:500}}
.heat{{display:inline-grid;place-items:center;min-width:34px;height:26px;background:rgba(97,169,255,var(--heat));color:var(--text)}}
.coverage{{margin-top:18px;display:grid;grid-template-columns:repeat(5,1fr);gap:10px}}
.coverage-card{{padding:14px;background:var(--surface);border:1px solid var(--line)}}
.coverage-card b{{font-size:20px}}
.coverage-card span{{display:block;color:var(--muted);font-size:11px}}
.footnote{{font-size:12px;color:var(--muted);margin-top:18px}}
.provenance{{display:grid;grid-template-columns:1fr 1fr;gap:10px;font:11px ui-monospace,monospace;color:var(--muted)}}
.provenance div{{background:var(--surface);border:1px solid var(--line);padding:13px;overflow-wrap:anywhere}}
dialog{{width:min(1240px,94vw);max-height:92vh;padding:0;border:1px solid #3c5d73;background:#0b1721;color:var(--text);box-shadow:var(--shadow)}}
dialog::backdrop{{background:rgba(0,5,10,.84);backdrop-filter:blur(4px)}}
.modal-head{{display:flex;justify-content:space-between;align-items:start;gap:20px;padding:22px 24px;border-bottom:1px solid var(--line);position:sticky;top:0;background:#0b1721;z-index:2}}
.modal-head h3{{font-size:24px;margin:0}}
.modal-head p{{font-size:12px;color:var(--muted);margin:5px 0 0}}
.close{{border:1px solid var(--line);background:transparent;color:var(--text);padding:8px 12px;cursor:pointer}}
.clips{{display:grid;grid-template-columns:1fr 1fr;gap:14px;padding:18px}}
.clip{{background:var(--surface);border:1px solid var(--line);padding:12px}}
.clip img{{width:100%;display:block;background:#071019}}
.clip h4{{margin:10px 0 4px;font-size:13px}}
.clip p{{margin:0;color:var(--muted);font-size:11px}}
@media(max-width:1050px){{.hero{{grid-template-columns:1fr}}.pipeline{{grid-template-columns:1fr 1fr}}.stage:after{{display:none}}.cluster-grid{{grid-template-columns:repeat(3,1fr)}}.selection-layout{{grid-template-columns:1fr}}.facts{{grid-template-columns:repeat(3,1fr)}}}}
@media(max-width:720px){{.shell{{padding:0 18px 50px}}.nav{{display:none}}.hero{{padding-top:46px}}h1{{font-size:45px}}.pipeline,.k-grid{{grid-template-columns:1fr}}.cluster-grid{{grid-template-columns:1fr 1fr}}.facts,.coverage{{grid-template-columns:1fr 1fr}}.clips{{grid-template-columns:1fr}}.section-head{{grid-template-columns:1fr;gap:4px}}.provenance{{grid-template-columns:1fr}}}}
@media(max-width:440px){{.cluster-grid,.facts,.coverage,.hero-meta{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<main class="shell">
  <header class="topbar">
    <div class="brand"><i></i> RoboLoRAHub · Cluster Review</div>
    <nav class="nav"><a href="#method">方法</a><a href="#choice">K=32</a><a href="#gallery">32簇窗口</a><a href="#selection">筛选确认</a></nav>
  </header>

  <section class="hero">
    <div>
      <div class="eyebrow">Robot action representation · visual assisted</div>
      <h1>从 18,325 个动作窗口<br>到 32 个可复核动作簇</h1>
      <p class="lead">以末端动作动态和状态变化为主体，图像编码仅提供约 5% 的距离贡献；通过量化指标选择 K=32，再用 128 个三相机 GIF 对每个簇进行训练代表与随机验证复核。</p>
    </div>
    <div class="hero-meta">
      <div><b>50</b><span>任务</span></div><div><b>2,500</b><span>轨迹</span></div>
      <div><b>18,325</b><span>动作窗口</span></div><div><b>128</b><span>复核 GIF</span></div>
    </div>
  </section>

  <section class="section" id="method">
    <div class="section-head"><div class="section-no">01 / METHOD</div><div><h2>聚类前先学习动作表示</h2><p>窗口长度 50、步长 25。不同机器人不直接比较原始关节角，而是统一使用双臂末端相对运动、姿态、夹爪和状态转移；视觉编码作为辅助距离进入融合空间。</p></div></div>
    <div class="pipeline">
      <article class="stage"><small>INPUT</small><h3>50-step 窗口</h3><p>起始观测、末端动作序列、状态变化、夹爪变化、结束状态。</p><div class="equation">window = [t, t+49]</div></article>
      <article class="stage"><small>MOTION</small><h3>MLP 动作编码</h3><p>动作重建与状态预测自监督训练；加入 λ=0.03 的相邻窗口一致性约束。</p><div class="equation">z_motion ∈ R⁶⁴</div></article>
      <article class="stage"><small>VISION AUX</small><h3>图像上下文</h3><p>起始图像特征压缩到 8 维，只修正运动相似但场景差异明显的边界。</p><div class="equation">z_visual ∈ R⁸</div></article>
      <article class="stage"><small>CALIBRATE</small><h3>距离贡献校准</h3><p>仅用训练集成对距离校准，目标视觉贡献 5%，避免外观主导聚类。</p><div class="equation">α = 0.023761</div></article>
      <article class="stage"><small>CLUSTER</small><h3>K-means · K=32</h3><p>训练窗口拟合中心；验证窗口只做预测。n_init=10，并比较多随机种子稳定性。</p><div class="equation">z = [√(1−α)zₘ, √αzᵥ]</div></article>
    </div>
  </section>

  <section class="section" id="choice">
    <div class="section-head"><div class="section-no">02 / WHY 32</div><div><h2>K=32 是分离度、稳定性和粒度的平衡点</h2><p>候选 K 使用相同窗口、编码器、episode 级训练/验证划分和验证抽样。轮廓系数评价簇间分离，ARI 评价初始化稳定性，R² 评价簇中心对动作特征的重建能力。</p></div></div>
    <div class="k-grid" id="k-grid"></div>
    <div class="decision"><b>选择逻辑：</b>K=16 稳定但动作粒度偏粗；K=48 的动作重建更细，但轮廓系数和初始化稳定性明显下降。K=32 同时取得候选中的最高轮廓系数与最高稳定性，并保留更可管理的专家数量，因此固定为后续窗口复核、五任务检索和 LoRA Bank 的工程工作点。</div>
    <div class="facts">
      <div class="fact"><strong>0.2059</strong><span>最终 λ=.03 / K32 验证轮廓</span></div>
      <div class="fact"><strong>0.7368</strong><span>最终初始化稳定性 ARI</span></div>
      <div class="fact"><strong>5.07%</strong><span>验证集视觉距离贡献</span></div>
      <div class="fact"><strong>172</strong><span>最终最小簇窗口数</span></div>
      <div class="fact"><strong>11.33%</strong><span>最终最大簇占比</span></div>
    </div>
  </section>

  <section class="section" id="gallery">
    <div class="section-head"><div class="section-no">03 / REVIEW</div><div><h2>32 个簇逐簇窗口复核</h2><p>每簇固定导出 2 个最接近训练中心且来自不同轨迹的代表窗口，以及 2 个来自不同轨迹的随机验证窗口。点击任意簇播放头部、左腕、右腕三相机 GIF；青色簇为后续选入 LoRA Bank 的七个簇。</p></div></div>
    <div class="gallery-tools">
      <div class="filters"><button class="active" data-filter="all">全部 32 簇</button><button data-filter="selected">LoRA 七簇</button><button data-filter="related">五任务有命中</button></div>
      <select id="sort" aria-label="簇排序"><option value="id">按簇 ID</option><option value="rank">按五任务相关性</option><option value="size">按簇窗口数</option></select>
    </div>
    <div class="cluster-grid" id="cluster-grid"></div>
  </section>

  <section class="section" id="selection">
    <div class="section-head"><div class="section-no">04 / RETRIEVAL</div><div><h2>从 32 簇到五任务相关的七个 LoRA 专家</h2><p>从五个目标任务各抽取 20 条完整轨迹，共形成 490 个动作窗口。聚类中心和标签保持冻结，使用任务等权覆盖率排序；自动 Top-5 后，再结合边界接近程度和人工复核扩展到 Top-7。</p></div></div>
    <div class="selection-layout">
      <div class="flow">
        <div class="flow-step"><b>A</b><div><h3>等量构造五任务子集</h3><p>100 条轨迹；每任务 20 条。抽样不读取聚类标签，避免按结果选数据。</p></div></div>
        <div class="flow-step"><b>B</b><div><h3>冻结 K=32 标签与中心</h3><p>不在五任务子集上重新拟合，保留全库动作空间中的原始簇定义。</p></div></div>
        <div class="flow-step"><b>C</b><div><h3>任务等权覆盖率排序</h3><p>score(c)=五个任务中“簇 c 窗口数 / 该任务全部窗口数”的平均值。</p></div></div>
        <div class="flow-step"><b>D</b><div><h3>自动 Top-5</h3><p>C08、C24、C03、C06、C09；覆盖 165/490 个窗口，避免长轨迹任务支配排名。</p></div></div>
        <div class="flow-step"><b>E</b><div><h3>边界复核扩展 Top-7</h3><p>C31 与第 5 名仅差 0.024 个百分点，C23 仍有跨任务覆盖；结合 GIF 后保留 C31、C23。</p></div></div>
      </div>
      <div class="ranking"><h3>32 簇相关性排序 · 青色为最终七簇</h3><div id="ranking"></div></div>
    </div>
    <div class="matrix-wrap"><table class="matrix" id="matrix"></table></div>
    <div class="coverage" id="coverage"></div>
    <p class="footnote">相关性分数衡量目标任务动作分布的覆盖，并不等同于技能语义准确率、簇纯度或策略成功率。Bootstrap 以整条轨迹为重采样单位；GIF 人工复核用于确认运动、夹爪与可见交互是否相对一致。</p>
  </section>

  <section class="section">
    <div class="section-head"><div class="section-no">05 / TRACE</div><div><h2>结果可追溯</h2><p>页面只展示已完成实验和固定标签，不在浏览器中重新计算结果。</p></div></div>
    <div class="provenance">
      <div>最终标签<br>runs/mlp_temporal_20261109_230731_563725/evaluation/lambda0_03__calibrated5/k32/labels.npy</div>
      <div>五任务排序<br>reports/five_task_subset20_lambda003_calibrated5_k32/summary.json</div>
      <div>标签 SHA256<br>{html.escape(subset['labels_sha256'])}</div>
      <div>窗口索引 SHA256<br>{html.escape(subset['index_sha256'])}</div>
    </div>
  </section>
</main>

<dialog id="cluster-dialog">
  <div class="modal-head"><div><h3 id="modal-title"></h3><p id="modal-meta"></p></div><button class="close" id="modal-close">关闭</button></div>
  <div class="clips" id="modal-clips"></div>
</dialog>

<script id="showcase-data" type="application/json">{safe_json(payload)}</script>
<script>
const D=JSON.parse(document.getElementById('showcase-data').textContent);
const selected=new Set(D.selectedSeven), rankById=new Map(D.ranking.map(x=>[x.cluster_id,x]));
const arm={{left:'左臂',right:'右臂',both:'双臂',small_translation:'小位移'}};
const esc=s=>String(s??'').replace(/[&<>\"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c]));
const pct=(x,n=1)=>(100*x).toFixed(n)+'%';

function renderK(){{
  const maxima={{silhouette:.27,ari:1,position:.6,gripper_delta:.85}};
  document.getElementById('k-grid').innerHTML=D.kComparison.map(k=>`<article class="k-card ${{k.k===32?'chosen':''}}"><div class="k-title"><b>K=${{k.k}}</b><span class="tag">${{k.k===32?'选定工作点':'候选'}}</span></div>${{
    [['silhouette','轮廓系数'],['ari','稳定性 ARI'],['position','位置 R²'],['gripper_delta','夹爪变化 R²']].map(([key,label])=>`<div class="metric"><span>${{label}}</span><div class="track"><i style="width:${{Math.min(100,k[key]/maxima[key]*100)}}%"></i></div><strong>${{k[key].toFixed(4)}}</strong></div>`).join('')}}<div class="cluster-stats">簇规模范围：${{k.size}} 个窗口</div></article>`).join('');
}}

let currentFilter='all';
function visibleClusters(){{
  let rows=D.clusters.filter(c=>currentFilter==='all'||(currentFilter==='selected'&&selected.has(c.cluster))||(currentFilter==='related'&&(rankById.get(c.cluster)?.subset_windows||0)>0));
  const mode=document.getElementById('sort').value;
  rows.sort((a,b)=>mode==='rank'?(rankById.get(a.cluster).rank-rankById.get(b.cluster).rank):mode==='size'?(b.count-a.count):(a.cluster-b.cluster));
  return rows;
}}
function renderClusters(){{
  document.getElementById('cluster-grid').innerHTML=visibleClusters().map(c=>{{
    const r=rankById.get(c.cluster), sample=c.samples[0];
    return `<article class="cluster-card ${{selected.has(c.cluster)?'selected':''}}"><div class="cluster-preview"><img loading="lazy" src="${{esc(sample.strip)}}" alt="簇 C${{String(c.cluster).padStart(2,'0')}} 的代表窗口"></div><div class="cluster-body"><div class="cluster-line"><span class="cluster-id">C${{String(c.cluster).padStart(2,'0')}}</span>${{selected.has(c.cluster)?'<span class="selected-mark">LoRA BANK</span>':''}}</div><div class="cluster-stats">${{c.count.toLocaleString()}} 窗口 · ${{c.tasks}} 任务${{r.subset_windows?` · 相关性 #${{r.rank}}`:''}}</div><div class="micro"><span>${{arm[c.dominant_arm.majority]||c.dominant_arm.majority}} ${{pct(c.dominant_arm.majority_fraction,0)}}</span><span>夹爪 ${{esc(c.gripper_event.majority)}} ${{pct(c.gripper_event.majority_fraction,0)}}</span></div><button class="open-cluster" data-cluster="${{c.cluster}}">播放 4 个窗口 GIF</button></div></article>`;
  }}).join('');
  document.querySelectorAll('.open-cluster').forEach(b=>b.addEventListener('click',()=>openCluster(+b.dataset.cluster)));
}}

function openCluster(id){{
  const c=D.clusters.find(x=>x.cluster===id),r=rankById.get(id),dlg=document.getElementById('cluster-dialog');
  document.getElementById('modal-title').textContent=`C${{String(id).padStart(2,'0')}} · 训练代表与随机验证`;
  document.getElementById('modal-meta').textContent=`${{c.count}} 个窗口 · ${{c.tasks}} 个任务 · 主运动 ${{arm[c.dominant_arm.majority]||c.dominant_arm.majority}} ${{pct(c.dominant_arm.majority_fraction)}}${{r.subset_windows?` · 五任务相关性第 ${{r.rank}} 名`:''}}`;
  document.getElementById('modal-clips').innerHTML=c.samples.map(s=>`<article class="clip"><img src="${{esc(s.gif)}}" alt="${{esc(s.kind)}} 窗口 ${{s.window_id}} 三相机 GIF"><h4>${{s.kind==='representative_train'?'训练代表':'随机验证'}} · 窗口 ${{s.window_id}} · ${{esc(s.dataset_id)}}</h4><p>帧 ${{s.start}}–${{s.end}} · 路径 L/R ${{s.path_m[0].toFixed(3)}} / ${{s.path_m[1].toFixed(3)}} m<br>夹爪 [${{s.gripper_start.map(x=>x.toFixed(2)).join(', ')}}] → [${{s.gripper_end.map(x=>x.toFixed(2)).join(', ')}}]<br>${{esc(s.instruction)}}</p></article>`).join('');
  dlg.showModal();
}}
function closeDialog(){{const d=document.getElementById('cluster-dialog');d.close();document.getElementById('modal-clips').innerHTML='';}}

function renderRanking(){{
  const max=D.ranking[0].balanced_task_coverage;
  document.getElementById('ranking').innerHTML=D.ranking.map(r=>`<div class="rank-row ${{selected.has(r.cluster_id)?'selected':''}}"><span>#${{r.rank}}</span><b>C${{String(r.cluster_id).padStart(2,'0')}}</b><div class="rankbar"><i style="width:${{100*r.balanced_task_coverage/max}}%"></i></div><span>${{pct(r.balanced_task_coverage,2)}}</span><span>${{r.subset_windows}} 窗口</span></div>`).join('');
}}
function renderMatrix(){{
  const rows=D.selectedSeven.map(id=>rankById.get(id));
  document.getElementById('matrix').innerHTML=`<thead><tr><th>最终七簇</th>${{D.tasks.map(t=>`<th>${{t.label}}<br><small>n=${{t.total}}</small></th>`).join('')}}<th>等权覆盖</th><th>富集</th><th>Top-5 重采样频率</th></tr></thead><tbody>${{rows.map(r=>`<tr><td><b>C${{String(r.cluster_id).padStart(2,'0')}}</b> · #${{r.rank}}</td>${{D.tasks.map(t=>{{const v=r[t.id+'_windows'];return `<td><span class="heat" style="--heat:${{Math.max(.08,v/t.total*.9)}}">${{v}}</span></td>`}}).join('')}}<td>${{pct(r.balanced_task_coverage,2)}}</td><td>${{r.enrichment_vs_global.toFixed(2)}}×</td><td>${{pct(r.bootstrap_top5_frequency,1)}}</td></tr>`).join('')}}</tbody>`;
  document.getElementById('coverage').innerHTML=D.tasks.map(t=>{{const n=D.selectedTaskCounts[t.id],p=n/t.total;return `<div class="coverage-card"><b>${{pct(p,1)}}</b><span>${{t.label}} · ${{n}}/${{t.total}} 窗口</span></div>`}}).join('');
}}

document.querySelectorAll('[data-filter]').forEach(b=>b.addEventListener('click',()=>{{document.querySelectorAll('[data-filter]').forEach(x=>x.classList.remove('active'));b.classList.add('active');currentFilter=b.dataset.filter;renderClusters();}}));
document.getElementById('sort').addEventListener('change',renderClusters);
document.getElementById('modal-close').addEventListener('click',closeDialog);
document.getElementById('cluster-dialog').addEventListener('click',e=>{{if(e.target===e.currentTarget)closeDialog();}});
renderK();renderClusters();renderRanking();renderMatrix();
</script>
</body>
</html>
"""
    (OUT / "index.html").write_text(document)
    print(f"Wrote {OUT / 'index.html'} ({len(document):,} chars)")
    print(f"Selected seven cover {selected_total}/490 windows; balanced coverage {selected_balanced:.2%}")


if __name__ == "__main__":
    main()
