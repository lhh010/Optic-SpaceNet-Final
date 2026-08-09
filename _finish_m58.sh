#!/bin/bash
# M5-8 训练收尾: 记录 M8 v8 结果 -> 补帕累托图 -> 生成图 -> 更新文档 -> 提交推送
set -e
cd /mnt/e/LT-Simulator/train-test
ER=eurosat_research

echo "==== [1/5] M8 v8 结果提取 ===="
grep -E "\[DONE\]" logs/m8_v8.log | tail -1 || echo "M8 v8 未完成!"
DONE=$(grep -oE "\[DONE\][^\n]*" logs/m8_v8.log | tail -1)
echo "$DONE"
if [ -z "$DONE" ]; then echo "M8 v8 尚未完成, 退出"; exit 1; fi

TEST=$(echo "$DONE" | grep -oE "test_acc=[0-9.]+" | cut -d= -f2)
F1=$(echo "$DONE" | grep -oE "test_f1=[0-9.]+" | cut -d= -f2)
ECE=$(echo "$DONE" | grep -oE "ece=[0-9.]+" | cut -d= -f2)
echo "M8 v8: test=$TEST f1=$F1 ece=$ECE"

echo "==== [2/5] 权重复制 ===="
cp $ER/runs/m8_rf_stem5_v8probe15_14e75e08/best.pth $ER/weights/m8_rf_stem5_v8probe15.pth
cp $ER/weights/m8_rf_stem5_v8probe15.pth weights/
echo "OK"

echo "==== [3/5] summary 更新 ===="
python3 - "$TEST" "$F1" "$ECE" <<'EOF'
import sys
test, f1, ece = sys.argv[1], sys.argv[2], sys.argv[3]
p = "logs/m5_m8_summary.md"
s = open(p, encoding="utf-8").read()
s = s.replace("| M8 | 自训 clean (96.17%) | 待填 | 待填 | — | — | 🔄 训练中 (15:57 启动) |",
              f"| M8 | 自训 clean (96.17%) | — | **{test}%** | {f1} | {ece} | 2026-08-09 16:42 | ✅ 完成 |")
open(p, "w", encoding="utf-8").write(s)
print("summary OK")
EOF

echo "==== [4/5] 帕累托图补点+生成 ===="
python3 - "$TEST" <<'EOF'
import sys
test = sys.argv[1]
p = "eurosat_research/docs/plot_perf_vs_macs.py"
s = open(p, encoding="utf-8").read()
s = s.replace('    ("M8 clean (待定)",             2.16, None,  "M5-M8 (v8)", (0, 0)),',
              f'    ("M8 v8",                       2.16, {test}, "M5-M8 (v8)", (8, -10)),')
open(p, "w", encoding="utf-8").write(s)
print("plot point OK")
EOF
cd eurosat_research/docs && python3 plot_perf_vs_macs.py 2>&1 | tail -2
ls -la perf_vs_macs_qat.png pareto_hw_acc.png 2>/dev/null

echo "==== [5/5] 提交推送 ===="
cd /mnt/e/LT-Simulator/train-test
printf "train(m8-v8): M8 v8 done (test %s%%); M5-M8 all complete; Pareto plot updated\n" "$TEST" > _cmsg.txt
cmd.exe /c "git add -f eurosat_research/weights/m8_rf_stem5_v8probe15.pth weights/m8_rf_stem5_v8probe15.pth eurosat_research/docs/plot_perf_vs_macs.py eurosat_research/docs/perf_vs_macs_qat.png && git add logs/m5_m8_summary.md logs/m8_v8.log && git commit -F _cmsg.txt && git push origin main" 2>&1 | tail -3
rm -f _cmsg.txt
echo "==== 收尾完成 ===="
