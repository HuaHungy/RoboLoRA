"""Label-independent gripper-event sampling and traceable visual review."""

from collections import defaultdict
import csv
from io import BytesIO
import html
import json
from pathlib import Path

import numpy as np

from .config import fresh_dir, write_json
from .progress import Progress


def detect_events(grip, minimum_change=.2, step_threshold=.005, max_gap=2):
    """Group same-direction gripper increments; events are kinematic, not skill labels."""
    events = []
    for arm in range(2):
        delta = np.diff(grip[:, arm])
        active = np.flatnonzero(np.abs(delta) >= step_threshold)
        groups = []
        for t in active:
            direction = int(np.sign(delta[t]))
            if groups and t - groups[-1][-1] <= max_gap + 1 and np.sign(delta[groups[-1][-1]]) == direction:
                groups[-1].append(int(t))
            else:
                groups.append([int(t)])
        for group in groups:
            start, end = group[0], group[-1] + 1
            change = float(grip[end, arm] - grip[start, arm])
            if abs(change) < minimum_change - 1e-7:
                continue
            events.append({'arm': arm, 'direction': 'open' if change > 0 else 'close',
                           'start': start, 'end': end, 'change': change,
                           'grip_before': float(grip[start, arm]), 'grip_after': float(grip[end, arm])})
    return sorted(events, key=lambda x: (x['start'], x['arm']))


def collect_events(windows, rows):
    episodes = {}
    for row in rows:
        if row['split'] == 'validation':
            episodes.setdefault(row['episode_id'], row)
    result = []
    for episode_id, row in sorted(episodes.items()):
        with np.load(windows / row['episode_cache']) as episode:
            events = detect_events(episode['gripper'])
        for event in events:
            result.append({**event, 'event_id': f"{episode_id}:a{event['arm']}:{event['start']}-{event['end']}",
                           'episode_id': episode_id, 'episode_cache': row['episode_cache'],
                           'source_path': row['source_path'], 'dataset_id': row['dataset_id'],
                           'fps': row['fps'], 'split': 'validation'})
    return result


def select_events(events, per_group=8, seed=20261110):
    rng = np.random.default_rng(seed)
    selected = []
    for arm in (0, 1):
        for direction in ('open', 'close'):
            candidates = [e for e in events if e['arm'] == arm and e['direction'] == direction]
            order = rng.permutation(len(candidates))
            used, chosen = set(), []
            for i in order:
                e = candidates[i]
                if e['episode_id'] not in used:
                    chosen.append(e); used.add(e['episode_id'])
                if len(chosen) == per_group:
                    break
            selected.extend(chosen)
    return selected


def bracket_event(event, rows, ids):
    ids = sorted(ids, key=lambda i: rows[i]['start'])
    middle = (event['start'] + event['end']) / 2
    left = [i for i in ids if (rows[i]['start'] + rows[i]['end']) / 2 <= middle]
    right = [i for i in ids if (rows[i]['start'] + rows[i]['end']) / 2 > middle]
    center_pair = [left[-1], right[0]] if left and right else None
    if center_pair and rows[center_pair[1]]['start'] - rows[center_pair[0]]['start'] != 25:
        center_pair = None
    before = [i for i in ids if rows[i]['end'] <= event['start']]
    after = [i for i in ids if rows[i]['start'] >= event['end']]
    clean_pair = [before[-1], after[0]] if before and after else None
    return {'center_pair': center_pair, 'clean_flank_pair': clean_pair,
            'center_windows_can_overlap_event': True}


def event_scores(events, rows, labels):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row['episode_id']].append(i)
    counts = defaultdict(lambda: [0, 0])
    for event in events:
        pair = bracket_event(event, rows, groups[event['episode_id']])['center_pair']
        if pair is None:
            continue
        key = ('left' if event['arm'] == 0 else 'right') + '_' + event['direction']
        for category in ('all', key):
            counts[category][0] += 1
            counts[category][1] += int(labels[pair[0]] == labels[pair[1]])
    return {k: {'eligible_events': v[0], 'same_cluster_events': v[1], 'same_cluster_fraction': v[1] / v[0],
                'interpretation': 'screening proxy only; not a skill-merge error rate'} for k, v in counts.items()}


def export_review(out, windows, rows, selected, label_sets, skip_images=False, smoke=False):
    """No labels enter event selection. Existing reference labels are only read here."""
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    import pyarrow.parquet as pq
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fresh_dir(out); assets = fresh_dir(out / 'assets')
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row['episode_id']].append(i)
    font_path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font = ImageFont.truetype(str(font_path), 13) if font_path.exists() else ImageFont.load_default()
    cards, records, manual = [], [], []
    progress = Progress('夹爪事件随机验证片段', len(selected), '片段')
    for n, event in enumerate(selected):
        stem = f'event_{n:03d}'
        with np.load(windows / event['episode_cache']) as episode:
            grip = episode['gripper'].copy()
        lo, hi = max(0, event['start'] - 50), min(len(grip) - 1, event['end'] + 50)
        brackets = bracket_event(event, rows, groups[event['episode_id']])
        comparison = {}
        for name, labels in label_sets.items():
            pair = brackets['center_pair']; clean = brackets['clean_flank_pair']
            comparison[name] = {'center_pair_labels': [int(labels[i]) for i in pair] if pair else None,
                                'center_pair_same_cluster': bool(labels[pair[0]] == labels[pair[1]]) if pair else None,
                                'clean_flank_labels': [int(labels[i]) for i in clean] if clean else None}
            manual.append({'event_id': event['event_id'], 'variant': name, 'arm': event['arm'], 'direction': event['direction'],
                           'visible_interaction': '', 'short_behavior_preserved': '', 'suspected_merge': '', 'notes': ''})
        fig, ax = plt.subplots(figsize=(9, 2.6))
        times = np.arange(lo, hi + 1)
        ax.plot(times, grip[lo:hi + 1, 0], label='left'); ax.plot(times, grip[lo:hi + 1, 1], label='right')
        ax.axvspan(event['start'], event['end'], color='orange', alpha=.25, label='kinematic event')
        for i in (brackets['center_pair'] or []):
            center = (rows[i]['start'] + rows[i]['end']) / 2
            ax.axvline(center, color='grey', linestyle='--', alpha=.6)
        ax.set(xlabel='Recorded frame index (nominal time)', ylabel='Gripper: 0 closed / 1 open', ylim=(-.05, 1.05))
        ax.legend(loc='best', fontsize=8)
        ax.set_title(f"{'SMOKE ONLY | ' if smoke else ''}{event['dataset_id']} | arm {event['arm']} {event['direction']}")
        fig.tight_layout(); fig.savefig(assets / f'{stem}_gripper.png', dpi=120); plt.close(fig)
        image_status, image_error = 'skipped_by_option', None
        clip_frames = []
        if not skip_images:
            try:
                table = pq.ParquetFile(event['source_path']).read(columns=['observation.images.cam_high'])
                actual_hi = min(hi, len(table) - 1)  # The reconstructed terminal numeric state has no image.
                if actual_hi < lo:
                    raise ValueError('No actual images in the requested interval')
                stride = max(1, int(np.ceil((actual_hi - lo + 1) / 36)))
                clip_frames = list(range(lo, actual_hi + 1, stride))
                strip_frames = [max(0, event['start'] - 10), min((event['start'] + event['end']) // 2, len(table) - 1),
                                min(event['end'] + 10, len(table) - 1)]
                requested = sorted(set(clip_frames + strip_frames))
                items = table.take(requested)['observation.images.cam_high'].to_pylist()
                pictures = {}
                for t, item in zip(requested, items):
                    if not isinstance(item, dict) or not item.get('bytes'):
                        raise ValueError('Expected embedded image bytes in the LeRobot Parquet column')
                    with Image.open(BytesIO(item['bytes'])) as im:
                        picture = ImageOps.pad(im.convert('RGB'), (384, 288), color='black')
                    draw = ImageDraw.Draw(picture)
                    draw.rectangle((0, 0, 384, 23), fill='black')
                    draw.text((4, 3), f"frame {t} | grip L/R {grip[t,0]:.2f}/{grip[t,1]:.2f}", font=font, fill='white')
                    pictures[t] = picture
                frames = [pictures[t] for t in clip_frames]
                frames[0].save(assets / f'{stem}.gif', save_all=True, append_images=frames[1:], loop=0,
                               duration=max(20, round(1000 * stride / event['fps'])))
                strip = Image.new('RGB', (1152, 288), 'white')
                for j, t in enumerate(strip_frames):
                    strip.paste(pictures[t], (384 * j, 0))
                strip.save(assets / f'{stem}.jpg', quality=90)
                image_status = 'complete'
            except (OSError, ValueError, KeyError, IndexError) as exc:
                image_status, image_error = 'unavailable', str(exc)
        record = {**event, 'brackets': brackets,
                  'bracket_window_provenance': {key: [rows[i] for i in value] if value else None
                                                for key, value in brackets.items() if key.endswith('pair')},
                  'comparison': comparison, 'image_status': image_status, 'image_error': image_error,
                  'image_frames': clip_frames, 'gripper_curve': f'assets/{stem}_gripper.png',
                  'gif': f'assets/{stem}.gif' if image_status == 'complete' else None,
                  'strip': f'assets/{stem}.jpg' if image_status == 'complete' else None}
        records.append(record)
        parts = [f'<article><h2>{n+1}. {html.escape(event["dataset_id"])} · arm {event["arm"]} {event["direction"]}</h2>',
                 f'<p>{html.escape(event["event_id"])} · 夹爪 {event["grip_before"]:.2f} → {event["grip_after"]:.2f}</p>']
        if image_status == 'complete':
            parts += [f'<a href="assets/{stem}.gif"><img loading="lazy" src="assets/{stem}.jpg"></a>',
                      f'<details><summary>播放真实头部相机片段</summary><img loading="lazy" src="assets/{stem}.gif"></details>']
        else:
            parts.append(f'<p>图像状态：{html.escape(image_status)}；{html.escape(image_error or "")}</p>')
        parts += [f'<img loading="lazy" src="assets/{stem}_gripper.png">',
                  '<details><summary>各方案窗口标签（簇编号不可跨模型比较）</summary><table><tr><th>方案</th><th>事件中心两侧窗口</th><th>完全位于事件前后的窗口</th></tr>']
        for name, item in comparison.items():
            parts.append(f'<tr><td>{html.escape(name)}</td><td>{item["center_pair_labels"]}</td><td>{item["clean_flank_labels"]}</td></tr>')
        parts.append('</table></details></article>'); cards.append('\n'.join(parts))
        progress.update(n + 1)
    write_json(out / 'selected_events.json', records)
    with (out / 'manual_review.csv').open('w', newline='') as stream:
        fields = ['event_id', 'variant', 'arm', 'direction', 'visible_interaction', 'short_behavior_preserved', 'suspected_merge', 'notes']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(manual)
    header = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>夹爪转换复核</title>
<style>body{max-width:1200px;margin:24px auto;font-family:sans-serif;background:#f5f6f8}article{background:white;padding:20px;margin:20px 0}img{max-width:100%}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #ddd}</style>
<h1>夹爪开合事件：随机验证片段复核</h1><p>样本按左右臂、开/闭分层，在查看聚类标签前固定随机抽取。0=闭合，1=张开。相同样本用于全部方案。</p>
<p>同簇不等于误合并，换簇不等于语义正确：中心两侧的50步窗口可能都覆盖事件，完全位于事件前后的窗口又可能相距较远。请结合实际运动、物体交互及短行为是否保留，填写 manual_review.csv；自动结果不提供抓取/释放语义真值。</p>'''
    if smoke:
        header += '<p><b>仅功能验证，非全量实验结果。</b></p>'
    (out / 'index.html').write_text(header + '\n'.join(cards) + '</html>')
    status = {'selected_events': len(records), 'images_complete': sum(r['image_status'] == 'complete' for r in records),
              'images_unavailable': sum(r['image_status'] == 'unavailable' for r in records), 'skip_images': skip_images,
              'semantic_review_status': 'pending_human_inspection'}
    write_json(out / 'status.json', status)
    return status
