#!/bin/bash

# ==============================================================================
# ParetoQ 数据集收集与处理脚本 (collect_datasets.sh)
#
# 本脚本旨在通过运行 gen_data.py，自动下载并生成以下 6 个数据集的 jsonl 格式划分版本：
# 1. wikitext2 (wikitext-2-raw-v1)
# 2. c4 (allenai/c4/en)
# 3. Fineweb-edu (HuggingFaceFW/fineweb-edu)
# 4. RedPajama (togethercomputer/RedPajama-Data-1T)
# 5. SlimPajama (cerebras/SlimPajama-627B)
# 6. Ultra-FineWeb (openbmb/Ultra-FineWeb)
#
# 特性：
# - 统一保存至目标目录: /data2/datasets/
# - 为每个 dataset_name 相同的数据集创建一个文件夹，并下载到该文件夹中。
# - 在执行每个数据集/划分的下载生成前，会自动检查该目录中是否已有对应文件，若有则自动跳过。
# ==============================================================================

# 1. 基础配置
BASE_DIR="/data2/datasets"
MIN_LENGTH=10

# 2. 创建目标输出目录
mkdir -p "$BASE_DIR"

echo "=============================================================="
echo "      开始检查并收集 ParetoQ 训练/评估数据集"
echo "      基础目录: $BASE_DIR"
echo "=============================================================="

# 3. 数据集处理核心函数
# 参数:
#   $1: dataset_name   (HuggingFace 数据集路径，如 "allenai/c4")
#   $2: dataset_config (数据集子配置，如 "en")
#   $3: split          (数据集划分，如 "train", "test", "validation", "en")
#   $4: max_samples    (最大样本数限制，可选。若为空则不限制数量)
download_and_generate() {
    local name="$1"
    local config="$2"
    local split="$3"
    local max_samples="$4"
    local text_key="$5"

    # 按照 gen_data.py 的命名逻辑转换斜杠为下划线
    local clean_name="${name//\//_}"
    
    # 为每一个dataset_name相同的数据集创建一个文件夹
    local dataset_dir="$BASE_DIR/$clean_name"
    mkdir -p "$dataset_dir"
    
    local filename="${clean_name}_${config}_${split}.jsonl"
    local filepath="$dataset_dir/$filename"

    echo ""
    echo "--> [检查阶段] 数据集: $name / $config | 划分: $split"
    
    # 检查文件是否已存在且非空
    if [ -s "$filepath" ]; then
        echo "    发现已有文件且非空: $filename"
        echo "    >>> [跳过] 已有本地版本，跳过此下载与处理步骤。"
    else
        echo "    未找到本地文件或文件为空，开始调用脚本下载/处理..."
        
        # 构建 Python 命令
        local cmd="python gen_data.py \
            --dataset_name \"$name\" \
            --dataset_config \"$config\" \
            --splits \"$split\" \
            --output_dir \"$dataset_dir\" \
            --min_length $MIN_LENGTH"

        if [ -n "$max_samples" ]; then
            cmd="$cmd --max_samples $max_samples"
            echo "    >>> 设定最大样本数限制: $max_samples"
        else
            echo "    >>> 设定无样本数限制（下载完整划分）"
        fi

        if [ -n "$text_key" ]; then
            cmd="$cmd --text_key \"$text_key\""
            echo "    >>> 设定文本字段键名: $text_key"
        fi

        local success=0
        local retry_count=0
        while [ $success -eq 0 ]; do
            echo "    运行命令: $cmd"
            eval "$cmd"
            local exit_code=$?

            if [ $exit_code -eq 0 ]; then
                success=1
                echo "    >>> [成功] $filename 处理完成并保存！"
            else
                retry_count=$((retry_count + 1))
                echo "    >>> [失败] 处理 $name ($config) - $split 时出错 (第 $retry_count 次失败)。"
                echo "    >>> 将在 3 秒后自动重启下载..."
                sleep 3
            fi
        done
    fi
    echo "--------------------------------------------------------------"
}

# 4. 执行各个数据集的下载与划分生成

# --- [1] wikitext2 ---
# 包含 train, test 划分
download_and_generate "wikitext" "wikitext-2-raw-v1" "train"
download_and_generate "wikitext" "wikitext-2-raw-v1" "test"

# --- [2] c4 (allenai/c4) ---
# c4 数据集极其庞大，通常使用 en 语言子集。
# 强烈建议在此设置合适的最大样本数（例如 50000 样本用于训练，10000 样本用于验证/评估）
# 如果需要获取完整无限制的数据，请把第四个参数删掉或设为空
download_and_generate "allenai/c4" "en" "train" # 50000
download_and_generate "allenai/c4" "en" "validation" # 10000

# --- [3] Fineweb-edu (HuggingFaceFW/fineweb-edu) ---
# 常用子集为 sample-10BT，主要包含 train 划分
download_and_generate "HuggingFaceFW/fineweb-edu" "sample-10BT" "train" # 50000

# --- [4] RedPajama (togethercomputer/RedPajama-Data-1T) ---
# 采用 default 子集，主要包含 train 划分
# download_and_generate "togethercomputer/RedPajama-Data-1T" "default" "train" # 50000

# --- [5] SlimPajama (gmongaras/SlimPajama-627B_Reupload) ---
# 包含 train, test, validation 划分
# download_and_generate "gmongaras/SlimPajama-627B_Reupload" "None" "train" # 50000
# download_and_generate "gmongaras/SlimPajama-627B_Reupload" "None" "test" # 10000

# --- [6] Ultra-FineWeb (openbmb/Ultra-FineWeb) ---
# 包含 en (英语) 和 zh (中文) 划分。这里默认下载 en 划分。
download_and_generate "openbmb/Ultra-FineWeb" "None" "en" "" "content"

# --- [7] wikitext103 ---
# 包含 train, test, validation 划分
download_and_generate "wikitext" "wikitext-103-raw-v1" "train"
download_and_generate "wikitext" "wikitext-103-raw-v1" "test"

echo ""
echo "=============================================================="
echo "                    所有数据集检查与收集完毕！"
echo "=============================================================="
