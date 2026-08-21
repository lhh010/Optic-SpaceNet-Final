# -*- coding: utf-8 -*-
"""检查 e5 图所有文字 bbox 两两重叠 + 文字 bbox 与曲线/均值线像素重叠"""
import importlib.util
spec = importlib.util.spec_from_file_location("plotmod", "plot_e5_uint4_residual.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
fig = m.fig
ren = fig.canvas.get_renderer()

def collect_texts():
    items = []
    seen = set()
    if fig._suptitle is not None:
        items.append(("suptitle", fig._suptitle))
        seen.add(id(fig._suptitle))
    items += [("figtext:" + t.get_text()[:12], t) for t in fig.texts if id(t) not in seen]
    for i, ax in enumerate(m.ax1, m.ax2) if isinstance(m.ax1, list) else [(1, m.ax1), (2, m.ax2)]:
        pass
    for name, ax in (("L", m.ax1), ("R", m.ax2)):
        items.append((f"{name}:title", ax.title))
        for t in ax.texts:
            items.append((f"{name}:box:" + t.get_text()[:14].replace(chr(10), "|"), t))
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        for t, v in zip(ax.get_xticklabels(), ax.get_xticks()):
            if t.get_text() and xlim[0] <= v <= xlim[1]:
                items.append((f"{name}:xtick:{t.get_text()}", t))
        for t, v in zip(ax.get_yticklabels(), ax.get_yticks()):
            if t.get_text() and ylim[0] <= v <= ylim[1]:
                items.append((f"{name}:ytick:{t.get_text()}", t))
    return [(n, t, t.get_window_extent(renderer=ren)) for n, t in items if t.get_text()]

texts = collect_texts()
print("== 文字-文字重叠 ==")
found = False
for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        n1, _, b1 = texts[i]; n2, _, b2 = texts[j]
        if b1.overlaps(b2):
            inter_w = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
            inter_h = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
            if inter_w > 1 and inter_h > 1:
                found = True
                print(f"  [{n1}] x [{n2}]  交叠 {inter_w:.0f}x{inter_h:.0f}px")
                for nn, tt, bb in ((n1, texts[i][1], b1), (n2, texts[j][1], b2)):
                    kind = "tick" if ":tick:" in nn else "text"
                    print(f"      - {nn}: kind={kind} 全文={tt.get_text()!r} pos={tt.get_position()} bbox=({bb.x0:.0f},{bb.y0:.0f})-({bb.x1:.0f},{bb.y1:.0f})")
if not found: print("  无")

print("== 文字 bbox 与曲线/均值线重叠 ==")
found = False
for name, ax in (("L", m.ax1), ("R", m.ax2)):
    for ln in ax.lines:
        pts = ax.transData.transform(ln.get_path().vertices)
        for tn, t, bb in texts:
            if not tn.startswith(name + ":box"):
                continue
            hits = ((pts[:, 0] >= bb.x0) & (pts[:, 0] <= bb.x1) &
                    (pts[:, 1] >= bb.y0) & (pts[:, 1] <= bb.y1)).sum()
            if hits > 0:
                found = True
                print(f"  [{tn}] 压 {ln.get_color()} 线 {hits} 个采样点")
if not found: print("  无")

print("== 越界/截断检查 ==")
fw, fh = fig.canvas.get_width_height()
for n, t, bb in texts:
    if bb.x0 < 0 or bb.y0 < 0 or bb.x1 > fw or bb.y1 > fh:
        print(f"  [{n}] 超出画布: {bb}")
print("done")
