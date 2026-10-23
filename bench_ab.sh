#!/bin/bash
# 交错 A/B 跑分：同窗口交错 R 轮（外加每 np 2 轮不计入热身），各取中位数
# 用法: ./bench_ab.sh <binA> <binB> [np列表,默认"1 2 4"] [轮数,默认10]
A="$1"; B="$2"; NPS="${3:-1 2 4}"; R="${4:-10}"
DATA="作业基于MPI的基本矩阵运算的附件/RandMatrix.txt"

run() { # np 二进制 → 输出 TOTAL 秒
    mpirun --bind-to core -np "$1" "$2" "$DATA" /tmp/ab_out.txt 2>/dev/null \
        | grep -oE 'TOTAL\(max\)=[0-9.]+' | cut -d= -f2
}
med() {
    printf '%s\n' "$@" | sort -n | awk '{a[NR]=$1} END{if(NR%2)print a[(NR+1)/2]; else printf "%.3f",(a[NR/2]+a[NR/2+1])/2}'
}
for np in $NPS; do
    run "$np" "$A" >/dev/null; run "$np" "$A" >/dev/null  # 热身×2
    run "$np" "$B" >/dev/null
    la=(); lb=()
    for r in $(seq "$R"); do
        la+=("$(run "$np" "$A")"); lb+=("$(run "$np" "$B")")
    done
    ma=$(med "${la[@]}"); mb=$(med "${lb[@]}")
    awk -v a="$ma" -v b="$mb" -v np="$np" \
        'BEGIN{printf "np=%s  A=%s  B=%s  B相对A %+.1f%%\n", np, a, b, (b-a)/a*100}'
done
