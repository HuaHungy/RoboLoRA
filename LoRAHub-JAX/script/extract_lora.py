"""
VLA LoRAHub - 从 Pi0 checkpoint 提取 LoRA 模块
================================================

使用方式:
  python script/extract_lora.py extract -c pi0_weights/29999/ -o lora_modules/task.npz -t grab_roller
  python script/extract_lora.py all -t vla_task
  python script/extract_lora.py info
"""

import os
import sys
import argparse
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from vla_lorahub.checkpoint_utils import (
    extract_lora_from_checkpoint,
    load_lora_module,
)


def extract_single(args):
    """提取单个 checkpoint 的 LoRA"""
    extract_lora_from_checkpoint(
        checkpoint_dir=args.checkpoint,
        output_path=args.output,
        task_name=args.task,
    )


def extract_all(args):
    """批量提取所有 checkpoint 的 LoRA"""
    weights_dir = os.path.join(ROOT, "pi0_weights")
    if not os.path.isdir(weights_dir):
        print(f"[ERROR] 找不到 pi0_weights 目录: {weights_dir}")
        sys.exit(1)

    # 找到所有 checkpoint 目录 (包含 params/_METADATA 的目录)
    checkpoints = []
    for item in sorted(os.listdir(weights_dir)):
        item_path = os.path.join(weights_dir, item)
        if os.path.isdir(item_path):
            meta_path = os.path.join(item_path, "params", "_METADATA")
            if os.path.exists(meta_path):
                checkpoints.append(item_path)

    print(f"[INFO] 找到 {len(checkpoints)} 个 checkpoint:")
    for ckpt in checkpoints:
        print(f"  - {ckpt}")

    output_dir = os.path.join(ROOT, "lora_modules")
    os.makedirs(output_dir, exist_ok=True)

    for ckpt in checkpoints:
        step_name = os.path.basename(ckpt)
        output_path = os.path.join(output_dir, f"task_{args.task}_{step_name}.npz")
        try:
            extract_lora_from_checkpoint(
                checkpoint_dir=ckpt,
                output_path=output_path,
                task_name=f"{args.task}_step{step_name}",
            )
        except Exception as e:
            print(f"[ERROR] 提取 {step_name} 失败: {e}")

    print(f"\n[DONE] 所有 LoRA 模块已保存到 {output_dir}/")
    print(f"[提示] 现在可以在 vla_lorahub/constant.py 中注册这些模块路径")


def main():
    parser = argparse.ArgumentParser(
        description="从 Pi0 Orbax checkpoint 提取 LoRA 模块"
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # 单文件提取
    single_parser = subparsers.add_parser("extract", help="提取单个 checkpoint")
    single_parser.add_argument("--checkpoint", "-c", required=True,
                               help="checkpoint 目录路径 (例如 pi0_weights/5000/)")
    single_parser.add_argument("--output", "-o", required=True,
                               help="输出 .npz 文件路径")
    single_parser.add_argument("--task", "-t", default="unknown",
                               help="任务名称 (记录到 metadata)")

    # 批量提取
    all_parser = subparsers.add_parser("all", help="批量提取所有 checkpoint")
    all_parser.add_argument("--task", "-t", default="vla_task", help="任务名称")

    # 查看已有模块
    info_parser = subparsers.add_parser("info", help="查看已有 LoRA 模块信息")
    info_parser.add_argument("module_path", nargs="?",
                             help=".npz 文件路径 (可选, 不提供则扫描 lora_modules/ 目录)")

    args = parser.parse_args()

    if args.command == "extract":
        extract_single(args)
    elif args.command == "all":
        extract_all(args)
    elif args.command == "info":
        if args.module_path:
            params, meta = load_lora_module(args.module_path)
            print(f"[模块信息] {args.module_path}")
            print(f"  任务: {meta.get('task_name', 'unknown')}")
            print(f"  参数数量: {len(params)}")
            print(f"  LoRA keys:")
            for key in sorted(params.keys()):
                print(f"    {key}: shape={params[key].shape}")
        else:
            modules_dir = os.path.join(ROOT, "lora_modules")
            if os.path.isdir(modules_dir):
                npz_files = glob.glob(os.path.join(modules_dir, "*.npz"))
                if npz_files:
                    print(f"[lora_modules/ 目录] 共 {len(npz_files)} 个模块:")
                    for f in sorted(npz_files):
                        params, meta = load_lora_module(f)
                        print(f"  {os.path.basename(f)}")
                        print(f"    任务: {meta.get('task_name', 'unknown')}")
                        print(f"    参数数: {len(params)}")
                else:
                    print("[提示] lora_modules/ 目录为空, 请先运行 extract 或 all 命令")
            else:
                print("[提示] lora_modules/ 目录不存在, 请先运行 extract 或 all 命令")
    else:
        parser.print_help()
        print("\n快捷用法:")
        print("  # 提取单个 checkpoint")
        print("  python script/extract_lora.py extract -c pi0_weights/29999/ -o lora_modules/task1.npz -t grab_roller")
        print()
        print("  # 批量提取所有 checkpoint")
        print("  python script/extract_lora.py all -t grab_roller")
        print()
        print("  # 查看已有模块")
        print("  python script/extract_lora.py info")


if __name__ == "__main__":
    main()
