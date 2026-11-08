"""Run the frozen, low-weight visual clustering experiment."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from action_clustering.visual_aux import run


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='冻结图片编码 + 低权重视觉辅助聚类（可复用逐轨迹图片缓存）')
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    run(args.config)
