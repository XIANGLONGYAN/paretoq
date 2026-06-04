"""
ParetoQ 训练数据生成脚本
从 HuggingFace 数据集生成 jsonl 格式的训练/评估数据

用法:
    # wikitext-2 训练集 -> wikitext_wikitext-2-raw-v1_train.jsonl
    python gen_data.py --dataset_name wikitext --dataset_config wikitext-2-raw-v1 --output_dir /jizhicfs/jackxlyan/dataset/paretoq_data --splits test

    # wikitext-103 训练集+测试集
    python gen_data.py --dataset_name wikitext --dataset_config wikitext-103-raw-v1 --output_dir /tmp/paretoq_data --splits train test

    # 其他数据集（如 c4）
    python gen_data.py --dataset_name allenai/c4 --dataset_config en --output_dir /tmp/paretoq_data --splits train --max_samples 50000
"""

import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Generate jsonl training data for ParetoQ")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/tmp/paretoq_data",
        help="Directory to save generated jsonl files",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext",
        help="HuggingFace dataset name (e.g., wikitext, allenai/c4)",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="wikitext-2-raw-v1",
        help="Dataset config/subset name (e.g., wikitext-2-raw-v1, en)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "test"],
        help="Which splits to download (e.g., train validation test)",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=10,
        help="Minimum character length of text to keep (filter out short lines)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to keep per split (None = keep all)",
    )
    parser.add_argument(
        "--text_key",
        type=str,
        default="text",
        help="The key of the text field in the dataset (e.g., text, content, raw)",
    )
    args = parser.parse_args()
    if args.dataset_config in ("None", "none", ""):
        args.dataset_config = None
    return args


def generate_jsonl(dataset_split, output_path, min_length, text_key="text", max_samples=None):
    """将 dataset split 转换为 jsonl 文件"""
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for example in dataset_split:
            text = example.get(text_key)
            if text is None:
                raise KeyError(f"Could not find text key '{text_key}' in dataset example. Available keys: {list(example.keys())}")
            
            text = text.strip()
            if len(text) >= min_length:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                count += 1
                if max_samples and count >= max_samples:
                    break
    return count


def main():
    args = parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Dataset: {args.dataset_name} / {args.dataset_config}")
    print(f"Output dir: {args.output_dir}")
    print(f"Splits: {args.splits}")
    print(f"Min text length: {args.min_length} chars")
    print(f"Max samples per split: {args.max_samples or 'unlimited'}")
    print("-" * 50)

    from datasets import load_dataset

    for split in args.splits:
        filename = f"{args.dataset_name}_{args.dataset_config}_{split}.jsonl"
        # 处理数据集名中的斜杠（如 allenai/c4 -> allenai_c4）
        filename = filename.replace("/", "_")
        output_path = os.path.join(args.output_dir, filename)
        print(f"Loading {split} split...")
        ds = load_dataset(args.dataset_name, args.dataset_config, split=split)
        count = generate_jsonl(ds, output_path, args.min_length, args.text_key, args.max_samples, trust_remote_code=True)
        print(f"  -> Saved {count} samples to {output_path}")

    print("-" * 50)
    print("Done!")


if __name__ == "__main__":
    main()
