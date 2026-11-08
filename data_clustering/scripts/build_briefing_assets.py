"""Export report-ready figures from the reviewed run; no model/data mutation."""

from pathlib import Path
import json
import shutil
import zipfile

import cv2
import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import FancyBboxPatch
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / 'reports/clustering_briefing_20261104'
RUN = BASE / 'runs/lerobot_ee_reviewed64'
REVIEW = BASE / 'reports/k32_latent64_review'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BOLD = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({'font.family': font_manager.FontProperties(fname=FONT).get_name(),
                     'axes.unicode_minus': False, 'font.size': 12,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlepad': 15, 'savefig.facecolor': 'white'})
COLORS = list(plt.get_cmap('tab20').colors) + list(plt.get_cmap('tab20b').colors[:12])
NAVY, TEAL, ORANGE, MUTED = '#17324d', '#178e88', '#dc8541', '#62758a'


def read(path):
    return json.loads(Path(path).read_text())


def save(fig, name):
    fig.savefig(OUT / (name + '.png'), dpi=200)
    fig.savefig(OUT / (name + '.svg'))
    plt.close(fig)


def pipeline():
    fig = plt.figure(figsize=(16, 9)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    fig.text(.055, .91, '无人工技能标签的末端动作聚类', fontsize=27, weight='bold', color=NAVY)
    fig.text(.055, .855, '将连续操作轨迹压缩为动作表示，再发现候选运动模式', fontsize=17, color=MUTED)
    cards = [
        ('01  轨迹与窗口', '50 个任务 · 2,500 条轨迹\n\n窗口长度 50，步长 25\n共 18,325 个动作窗口'),
        ('02  末端动态特征', '动态输入：50 × 16\n状态变化输入：14 维\n\n位置和旋转以窗口起点为参考'),
        ('03  自监督表示', '双路 MLP → 融合投射\n64 维向量 + L2 归一化\n\n动作重建 + 状态变化重建'),
        ('04  聚类与追溯', 'K-means：K=32\n每次 10 次初始化\n\n输出簇编号、中心、代表片段'),
    ]
    for i, (title, body) in enumerate(cards):
        x = .055 + i * .233
        ax.add_patch(FancyBboxPatch((x, .40), .208, .35, boxstyle='round,pad=0.01',
                                   facecolor='#f0f5fa', edgecolor='#dae4ee', linewidth=1.2))
        fig.text(x + .014, .695, title, fontsize=17, weight='bold', color=NAVY)
        fig.text(x + .014, .62, body, fontsize=14, color=NAVY, va='top', linespacing=1.7)
        if i < 3:
            ax.annotate('', xy=(x + .228, .57), xytext=(x + .213, .57),
                        arrowprops={'arrowstyle': '-|>', 'color': TEAL, 'lw': 2})
    ax.add_patch(FancyBboxPatch((.055, .21), .895, .10, boxstyle='round,pad=.01', facecolor='#e7f3f0', edgecolor='none'))
    fig.text(.075, .265, '训练：2,000 条轨迹 / 14,660 个窗口       验证：500 条轨迹 / 3,665 个窗口', fontsize=16, color=NAVY)
    fig.text(.075, .225, '按整条轨迹划分；归一化、模型训练和聚类中心拟合均只使用训练集', fontsize=13, color=TEAL)
    fig.text(.055, .12, '当前输入：双臂末端位姿与夹爪；图像仅用于结果复核，尚未输入聚类模型。', fontsize=14, color=MUTED)
    fig.text(.055, .075, '状态分支重建已输入的首尾变化；当前产物是候选运动簇，尚未验证为可复用技能。', fontsize=13, color=MUTED)
    save(fig, '01_聚类方案')


def evaluation(selection):
    old, new = selection['original'], selection['selected']
    fig, axs = plt.subplots(1, 3, figsize=(16, 8.5))
    fig.subplots_adjust(left=.07, right=.97, top=.75, bottom=.25, wspace=.42)
    fig.suptitle('当前结果：几何分组与稳定性有所改善，重建误差略增', x=.055, ha='left', y=.95,
                 fontsize=24, color=NAVY, weight='bold')
    fig.text(.055, .865, '原方案：128 维 / 30 轮中选第 29 轮     当前方案：64 维 / 第 30 轮     两者均为 K=32', color=MUTED, fontsize=14)
    panels = [('validation_silhouette', '主指标：验证轮廓系数 ↑', (0, .4), '.3f'),
              ('validation_ari_mean', '辅助：聚类初始化一致性 ARI ↑', (0, 1), '.3f'),
              ('validation_action_mse', '辅助：动作重建 MSE ↓', (0, .05), '.4f')]
    for ax, (key, title, limits, fmt) in zip(axs, panels):
        values = [old[key], new[key]]
        ax.bar([0, 1], values, color=[MUTED, TEAL], width=.53)
        ax.set(xticks=[0, 1], xticklabels=['原方案', '当前方案'], ylim=limits, title=title)
        ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
        for i, value in enumerate(values): ax.text(i, value + limits[1] * .025, format(value, fmt), ha='center', fontsize=19, weight='bold', color=NAVY)
    fig.text(.055, .15, '轮廓系数：固定随机抽取 2,000 个验证窗口，在原始 64 / 128 维空间中计算；不是分类准确率。', fontsize=13, color=MUTED)
    fig.text(.055, .10, 'ARI：固定网络，K-means 种子 0、1、2 与正式种子 42 的验证集分配一致性均值；不是技能标签正确率。', fontsize=12, color=MUTED)
    fig.text(.055, .05, 'MSE：标准化动作动态的离线重建误差；当前尚未评估独立任务迁移和下游机器人控制成功率。', fontsize=12, color=MUTED)
    save(fig, '02_评估指标对比')


def distribution(rows, z, labels, metrics):
    tr = np.array([i for i, r in enumerate(rows) if r['split'] == 'train'])
    va = np.array([i for i, r in enumerate(rows) if r['split'] == 'validation'])
    with threadpool_limits(limits=4):
        pca = PCA(n_components=2, random_state=42).fit(z[tr]); xy = pca.transform(z[va])
    explained = float(pca.explained_variance_ratio_.sum())
    fig, axs = plt.subplots(1, 2, figsize=(16, 8.5), gridspec_kw={'width_ratios': [1.12, 1]})
    fig.subplots_adjust(left=.07, right=.97, top=.77, bottom=.23, wspace=.35)
    fig.suptitle('32 个候选运动簇：存在结构，也存在交叠', x=.055, ha='left', y=.95,
                 fontsize=25, color=NAVY, weight='bold')
    fig.text(.055, .865, '全部数据得到 18,325 × 64 的表示；左图仅展示 3,665 个验证窗口', fontsize=15, color=MUTED)
    colors = ListedColormap(COLORS)
    sc = axs[0].scatter(xy[:, 0], xy[:, 1], c=labels[va], cmap=colors,
                       norm=BoundaryNorm(np.arange(-.5, 32.5, 1), 32), s=9, alpha=.62, linewidths=0, rasterized=True)
    axs[0].set(xlabel=f'PC1（{pca.explained_variance_ratio_[0]:.1%} 方差）',
               ylabel=f'PC2（{pca.explained_variance_ratio_[1]:.1%} 方差）', title='PCA 二维投影：仅用于诊断')
    fig.colorbar(sc, ax=axs[0], ticks=[0, 4, 8, 12, 16, 20, 24, 28, 31], label='簇编号', fraction=.04, pad=.03)
    sizes = np.bincount(labels, minlength=32)
    axs[1].bar(np.arange(32), sizes, color=COLORS)
    axs[1].set(xlabel='簇编号', ylabel='窗口数量（全量）', xticks=np.arange(0, 32, 4), title='簇规模：183～2,100 个窗口')
    axs[1].grid(axis='y', alpha=.15); axs[1].set_axisbelow(True)
    fig.text(.055, .14, f'前两个主成分仅保留 {explained:.1%} 的方差；二维图不能代替原始 64 维聚类指标。', fontsize=14, color=MUTED)
    fig.text(.055, .085, f'32 个簇均非空，最大簇占 {sizes.max()/len(labels):.1%}。K-means 会分配全部窗口，覆盖率 100% 不等于技能识别正确率。', fontsize=13, color=MUTED)
    save(fig, '03_潜在空间与簇分布')
    return explained


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def filmstrip(samples, cluster, positions, title, subtitle, descriptions, filename):
    chosen = [s for s in samples if s['cluster'] == cluster]
    chosen = [chosen[i] for i in positions]
    im = Image.new('RGB', (1920, 1080), 'white'); draw = ImageDraw.Draw(im)
    draw.text((64, 40), title, font=font(46, True), fill=NAVY)
    draw.text((64, 110), subtitle, font=font(25), fill=MUTED)
    columns = [535, 975, 1415]
    for x, title_ in zip(columns, ['窗口起点', '窗口中点', '窗口终点']):
        draw.text((x + 140, 169), title_, font=font(27, True), fill=NAVY)
    for index, (row, desc) in enumerate(zip(chosen, descriptions)):
        y = 211 + index * 267
        draw.rounded_rectangle((48, y, 1872, y + 255), radius=12, fill='#f2f6fa')
        draw.text((72, y + 26), desc, font=font(31, True), fill=NAVY)
        kind = '训练代表' if row['kind'] == 'representative_train' else '随机验证片段'
        draw.text((72, y + 84), kind, font=font(25), fill=TEAL)
        draw.text((72, y + 129), f"窗口 {row['window_id']} · C{cluster:02d}", font=font(23), fill=MUTED)
        draw.text((72, y + 170), row['dataset_id'], font=font(18), fill=MUTED)
        with h5py.File(row['native_source_path'], 'r') as raw:
            for x, t in zip(columns, [row['start'], (row['start'] + row['end']) // 2, row['end']]):
                rgb = cv2.imdecode(np.frombuffer(raw['observation/head_camera/rgb'][t], np.uint8), cv2.IMREAD_COLOR)
                if rgb is None: raise ValueError('Image decode failed')
                tile = ImageOps.contain(Image.fromarray(rgb), (425, 225), Image.Resampling.LANCZOS)
                im.paste(tile, (x + (425 - tile.width) // 2, y + 8))
                draw.text((x + 155, y + 231), f'原始帧 {t}', font=font(17), fill=MUTED)
    draw.text((64, 1030), '真实轨迹抽样；簇名称是事后解释，不是训练标签。画面相近不能证明所有样本具有相同技能语义。', font=font(22), fill=MUTED)
    im.save(OUT / (filename + '.png'))
    first = chosen[0]
    shutil.copy2(REVIEW / first['gif'], OUT / f"动图_C{cluster:02d}_窗口{first['window_id']}.gif")
    return [{'figure': filename, **r} for r in chosen]


def motion_curves(samples, action):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.subplots_adjust(left=.07, right=.97, top=.78, bottom=.16, hspace=.4, wspace=.3)
    fig.suptitle('同簇运动与夹爪曲线：从“看起来相似”到数值对照', x=.055, ha='left', y=.95,
                 fontsize=24, color=NAVY, weight='bold')
    fig.text(.055, .865, '每簇 2 个训练代表 + 2 个随机验证窗口；全部 50 步均参与曲线绘制', fontsize=15, color=MUTED)
    palette = [NAVY, TEAL, ORANGE, '#9364ae']
    for col, (cluster, arm, title) in enumerate([(27, 1, 'C27 · 右手张开接近'), (5, 1, 'C05 · 右手闭合 / 抬升'), (6, 0, 'C06 · 左手释放 / 撤离')]):
        chosen = [s for s in samples if s['cluster'] == cluster]
        for i, row in enumerate(chosen):
            a = action[row['window_id']]
            disp = np.r_[0, np.linalg.norm(a[:, arm * 6:arm * 6 + 3], axis=1)]
            grip = np.r_[a[:, 12 + arm], a[-1, 12 + arm] + a[-1, 14 + arm]]
            kind = '代表' if i < 2 else '验证'
            label = f'{kind} · 窗口{row["window_id"]}'
            axes[0, col].plot(np.arange(51), disp, color=palette[i], label=label, linewidth=2)
            axes[1, col].plot(np.arange(51), grip, color=palette[i], linewidth=2)
        axes[0, col].set(title=title, ylabel='距窗口起点的位移 / 米')
        axes[0, col].legend(fontsize=8, loc='upper left')
        axes[1, col].set(xlabel='窗口内帧偏移', ylabel='夹爪状态（0 闭合，1 张开）', ylim=(-.05, 1.05))
        for ax in axes[:, col]: ax.grid(alpha=.15)
    fig.text(.055, .075, '位移取各自起点坐标系下的位置向量模长，并非累计路径长度；这里的帧偏移不代表精确物理时间。', fontsize=13, color=MUTED)
    save(fig, '07_运动与夹爪一致性')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    selection = read(BASE / 'reports/parameter_review/selection.json')
    rows = [json.loads(s) for s in (RUN / 'windows/index.jsonl').read_text().splitlines()]
    z = np.load(RUN / 'embeddings/z.npy'); labels = np.load(RUN / 'clusters/kmeans_k32/labels.npy')
    samples = read(REVIEW / 'selected_windows.json')
    action = np.load(RUN / 'windows/action.npy', mmap_mode='r')
    assert z.shape == (18325, 64) and len(labels) == len(rows)
    pipeline(); evaluation(selection)
    variance = distribution(rows, z, labels, read(RUN / 'clusters/kmeans_k32/metrics.json'))
    provenance = []
    provenance += filmstrip(samples, 27, [0, 1, 2], '一致性示例一：跨物体的右手张开接近（C27）',
                            '不同任务和物体中出现相近的接近运动；本图选取已有抽样中的两个代表和一个验证片段。',
                            ['鞋子相关片段', '订书机相关片段', '积木相关片段'], '04_跨任务接近动作')
    provenance += filmstrip(samples, 5, [0, 1, 2], '一致性示例二：右手闭合与后续抬升样式（C05）',
                            '物体外观不同，但夹爪闭合和末端运动样式相近；窗口仍可能覆盖多个子阶段。',
                            ['积木相关片段', '印章相关片段', '盒状物体相关片段'], '05_闭合与抬升样式')
    provenance += filmstrip(samples, 28, [0, 2, 3], '当前局限：同一运动簇仍可能包含不同交互（C28）',
                            '闭合夹爪状态下的相似末端运动，尚不足以区分按压、持物与搬运等效果。',
                            ['订书机操作附近片段', '持瓶 / 搬运片段一', '持瓶 / 搬运片段二'], '06_混合簇与改进方向')
    motion_curves(samples, action)
    metadata = {'source_run': str(RUN), 'sample_selection': 'illustrative subset of the prior fixed representative/random review; not a quantitative semantic evaluation',
                'pca_two_component_variance_ratio': variance, 'main_metric': {'name': 'validation_silhouette', 'value': selection['selected']['validation_silhouette'], 'sample_count': 2000, 'validation_count': 3665, 'space_dimensions': 64},
                'selection': selection['recommended_parameters'], 'figure_samples': provenance}
    (OUT / '图像来源与指标口径.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f'Generated 7 figures + 3 GIF clips; PCA explained variance={variance:.4%}; output={OUT}')


if __name__ == '__main__':
    main()
