#!/bin/bash
# 批量生成课程音频 + PPTX
# 用法: bash script/batch_build.sh

set -e

cd "$(dirname "$0")/.."

LESSONS=(
    "孙权劝学"
    "木兰诗"
    "卖油翁"
    "陋室铭"
    "爱莲说"
    "活板"
    "河中石兽"
    "登幽州台歌"
    "望岳"
    "登飞来峰"
    "游山西村"
    "己亥杂诗（其五）"
    "竹里馆"
    "春夜洛城闻笛"
    "逢入京使"
    "晚春"
    "泊秦淮"
    "贾生"
    "过松源晨炊漆公店（其五）"
    "约客"
)

TOTAL=${#LESSONS[@]}
FAILED=()

for i in "${!LESSONS[@]}"; do
    lesson="${LESSONS[$i]}"
    echo "========================================"
    echo "[$((i+1))/$TOTAL] $lesson"
    echo "========================================"

    if python3 script/buildclass.py "$lesson" --audio --pptx; then
        echo "✓ $lesson 完成"
    else
        echo "✗ $lesson 失败"
        FAILED+=("$lesson")
    fi
    echo ""
done

echo "========================================"
echo "全部完成: $((TOTAL - ${#FAILED[@]}))/$TOTAL"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "失败列表:"
    for f in "${FAILED[@]}"; do
        echo "  - $f"
    done
fi
