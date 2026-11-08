"""Build a traceable visual/numeric review without changing an existing run.

Run from data_clustering/: python scripts/review_clusters.py --help
"""

import argparse
from collections import Counter, defaultdict
import html
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
SMALL = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 11)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def window_descriptors(action):
    position = action[:, :, [0, 1, 2, 6, 7, 8]].reshape(len(action), -1, 2, 3)
    path = np.linalg.norm(np.diff(position, axis=1, prepend=np.zeros_like(position[:, :1])), axis=3).sum(axis=1)
    grip_start = action[:, 0, 12:14]
    grip_end = action[:, -1, 12:14] + action[:, -1, 14:16]
    grip_delta = grip_end - grip_start
    # Review heuristics only, never training labels.
    dominant = np.where(path.max(axis=1) < .03, 'small_translation',
                        np.where(path[:, 0] > 3 * path[:, 1], 'left',
                                 np.where(path[:, 1] > 3 * path[:, 0], 'right', 'both')))
    events = np.where(grip_delta > .2, 'open', np.where(grip_delta < -.2, 'close', 'hold'))
    patterns = np.array([f'{left}/{right}' for left, right in events])
    return path, grip_start, grip_end, dominant, patterns


def frame(raw, camera, index, size):
    encoded = raw[f'observation/{camera}/rgb'][index]
    # Preserve the already-audited converter's channel convention.
    rgb = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
    if rgb is None:
        raise ValueError(f'Cannot decode {camera} frame {index}')
    return Image.fromarray(rgb).resize(size, Image.Resampling.LANCZOS)


def make_assets(raw, row, destination):
    start, end = row['start'], row['end']
    strip = Image.new('RGB', (576, 166), 'white')
    for i, t in enumerate([start, (start + end) // 2, end]):
        strip.paste(frame(raw, 'head_camera', t, (192, 144)), (i * 192, 22))
        ImageDraw.Draw(strip).text((i * 192 + 4, 3), f'frame {t}', fill='black', font=SMALL)
    strip.save(destination.with_suffix('.jpg'), quality=90)
    frames = []
    for t in range(start, end + 1, 5):
        im = Image.new('RGB', (768, 216), 'white')
        draw = ImageDraw.Draw(im)
        for camera_index, camera in enumerate(['head_camera', 'left_camera', 'right_camera']):
            im.paste(frame(raw, camera, t, (256, 192)), (camera_index * 256, 24))
            draw.text((camera_index * 256 + 3, 4), f'{camera} | frame {t}', fill='black', font=SMALL)
        frames.append(im)
    frames[0].save(destination.with_suffix('.gif'), save_all=True, append_images=frames[1:],
                   duration=round(5000 / row['fps']), loop=0, optimize=False)
    return strip


def build(run, out, labels_path=None, z_path=None, k=32):
    out.mkdir(parents=True, exist_ok=False)
    assets = out / 'assets'; assets.mkdir()
    rows = [json.loads(line) for line in (run / 'windows/index.jsonl').read_text().splitlines()]
    labels = np.load(labels_path or run / f'clusters/kmeans_k{k}/labels.npy')
    z = np.load(z_path or run / 'embeddings/z.npy')
    action = np.load(run / 'windows/action.npy', mmap_mode='r')
    assert len(rows) == len(labels) == len(z) == len(action)
    path, g0, g1, dominant, patterns = window_descriptors(action)
    rng = np.random.default_rng(20261104)
    cards, reviews, selections = [], [], []
    for c in range(k):
        ids = np.flatnonzero(labels == c)
        train = np.array([i for i in ids if rows[i]['split'] == 'train'])
        validation = np.array([i for i in ids if rows[i]['split'] == 'validation'])
        center = z[train].mean(axis=0)
        ranking = train[np.argsort(np.linalg.norm(z[train] - center, axis=1))]
        chosen, used = [], set()
        for i in ranking:
            if rows[i]['episode_id'] not in used:
                chosen.append(('representative_train', int(i))); used.add(rows[i]['episode_id'])
            if len(chosen) == 2: break
        for i in rng.permutation(validation):
            if rows[i]['episode_id'] not in used:
                chosen.append(('random_validation', int(i))); used.add(rows[i]['episode_id'])
            if len(chosen) == 4: break
        if len(chosen) != 4:
            raise ValueError(f'Cluster {c} lacks distinct episodes for unbiased review')
        def categorical(values):
            counts = Counter(values[ids].tolist())
            return {'counts': dict(counts), 'majority': counts.most_common(1)[0][0],
                    'majority_fraction': counts.most_common(1)[0][1] / len(ids)}
        review = {'cluster': c, 'count': len(ids), 'tasks': len({rows[i]['dataset_id'] for i in ids}),
                  'dominant_arm': categorical(dominant), 'gripper_event': categorical(patterns),
                  'path_length_median_m': np.median(path[ids], axis=0).tolist(),
                  'gripper_start_median': np.median(g0[ids], axis=0).tolist(),
                  'gripper_end_median': np.median(g1[ids], axis=0).tolist(), 'samples': []}
        card = Image.new('RGB', (576, 850), 'white'); draw = ImageDraw.Draw(card)
        draw.text((5, 5), f'C{c:02d} | n={len(ids)} | {review["tasks"]} tasks', fill='black', font=FONT)
        draw.text((5, 25), f'arm: {review["dominant_arm"]["majority"]} {review["dominant_arm"]["majority_fraction"]:.0%} | grip: {review["gripper_event"]["majority"]} {review["gripper_event"]["majority_fraction"]:.0%}', fill='black', font=SMALL)
        for j, (kind, i) in enumerate(chosen):
            row = rows[i]; name = f'c{c:02d}_w{i:05d}'; target = assets / name
            with h5py.File(row['native_source_path'], 'r') as raw:
                strip = make_assets(raw, row, target)
            desc = f'{kind} | w{i} | {row["dataset_id"]}'
            y = 48 + j * 200
            draw.text((5, y), desc, fill='black', font=SMALL)
            draw.text((5, y + 16), f'path L/R={path[i, 0]:.2f}/{path[i, 1]:.2f}m; grip {g0[i, 0]:.2f}/{g0[i, 1]:.2f} -> {g1[i, 0]:.2f}/{g1[i, 1]:.2f}', fill='black', font=SMALL)
            card.paste(strip, (0, y + 32))
            item = {**row, 'cluster': c, 'kind': kind, 'gif': f'assets/{name}.gif',
                    'strip': f'assets/{name}.jpg', 'path_m': path[i].tolist(),
                    'gripper_start': g0[i].tolist(), 'gripper_end': g1[i].tolist()}
            review['samples'].append(item); selections.append(item)
        card.save(out / f'cluster_{c:02d}.jpg', quality=92)
        cards.append(card); reviews.append(review)
        print(f'Cluster {c:02d}: 2 representatives + 2 random validation clips exported', flush=True)
    for start in range(0, k, 4):
        page = Image.new('RGB', (1152, 1700), '#d0d0d0')
        for i, card in enumerate(cards[start:start + 4]): page.paste(card, ((i % 2) * 576, (i // 2) * 850))
        page.save(out / f'overview_{start:02d}_{min(start+3,k-1):02d}.jpg', quality=94)
    write(out / 'review_metrics.json', reviews)
    write(out / 'selected_windows.json', selections)
    parts = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>动作聚类片段复核</title>',
             '<style>body{font-family:sans-serif;max-width:1250px;margin:24px auto;background:#f6f8fa;color:#17212b}article{background:white;padding:20px;margin:20px 0;border:1px solid #ddd}summary{cursor:pointer}img{max-width:100%}.clips{display:grid;grid-template-columns:1fr 1fr;gap:20px}p{line-height:1.6}</style>',
             f'<h1>K={k} 动作聚类复核</h1><p>每簇 2 个不同轨迹的训练代表 + 2 个随机验证片段。GIF 展示头部、左腕、右腕三路相机，每 5 帧采样；播放时间按名义帧率，仅用于观察顺序。夹爪 0=闭合、1=张开。各阈值仅为复核启发式，不是技能真值。</p>',
             '<p>点击 GIF 查看动态。请记录：运动一致性、夹爪一致性、物体交互是否可见、是否混合多个阶段。任务名不作为技能标签。</p>']
    for r in reviews:
        c = r['cluster']; parts.append(f'<article id="c{c}"><h2>簇 {c} · {r["count"]} 个窗口 · {r["tasks"]} 个任务</h2>')
        parts.append(f'<p>主要运动臂：{r["dominant_arm"]["majority"]}，占 {r["dominant_arm"]["majority_fraction"]:.1%}；主要夹爪净变化：{r["gripper_event"]["majority"]}，占 {r["gripper_event"]["majority_fraction"]:.1%}。</p><div class="clips">')
        for s in r['samples']:
            parts.append(f'<section><b>{s["kind"]} / 窗口 {s["window_id"]}</b><p>{html.escape(s["dataset_id"])} · 帧 {s["start"]}–{s["end"]}<br>{html.escape(str(s["instruction"]))}</p><a href="{s["gif"]}"><img loading="lazy" src="{s["strip"]}"></a><details><summary>播放三相机片段</summary><img loading="lazy" src="{s["gif"]}"></details><p>末端路径 L/R：{s["path_m"][0]:.3f} / {s["path_m"][1]:.3f} 米；夹爪 {s["gripper_start"]} → {s["gripper_end"]}</p></section>')
        parts.append('</div></article>')
    parts.append('</html>'); (out / 'index.html').write_text('\n'.join(parts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=Path('runs/lerobot_ee_all'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--embeddings', type=Path)
    parser.add_argument('--k', type=int, default=32)
    args = parser.parse_args()
    build(args.run, args.output, args.labels, args.embeddings, args.k)
