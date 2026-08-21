# -*- coding: utf-8 -*-
"""e5_noise_vs_signal_domains 文字碰撞 QA: 文字-文字 bbox / 文字-线像素 / 文字-标记 / 越界"""
import importlib.util
spec = importlib.util.spec_from_file_location("plotmod", "plot_e5_noise_vs_signal_domains.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
fig, ax = m.fig, m.ax
fig.canvas.draw()
ren = fig.canvas.get_renderer()

items = []
if fig._suptitle is not None and fig._suptitle.get_text():
    items.append(("suptitle", fig._suptitle))
items += [("figtext", t) for t in fig.texts if t.get_text()]
items.append(("title", ax.title))
lab_ids = {id(a) for a in m.lab_arts}
items += [("note:" + t.get_text()[:10].replace(chr(10), "|"), t) for t in ax.texts
          if t.get_text() and id(t) not in lab_ids and not hasattr(t, "xy")]
items += [("lab:" + a.get_text(), a) for a in m.lab_arts]
xlim, ylim = ax.get_xlim(), ax.get_ylim()
for t, v in zip(ax.get_xticklabels(), ax.get_xticks()):
    if t.get_text() and xlim[0] <= v <= xlim[1]:
        items.append(("xtick:" + t.get_text(), t))
for t, v in zip(ax.get_yticklabels(), ax.get_yticks()):
    if t.get_text() and ylim[0] <= v <= ylim[1]:
        items.append(("ytick:" + t.get_text(), t))
items = [(n, t, t.get_window_extent(renderer=ren)) for n, t in items if t.get_text()]

print("== 文字-文字 ==")
bad = 0
for i in range(len(items)):
    for j in range(i + 1, len(items)):
        n1, _, b1 = items[i]; n2, _, b2 = items[j]
        if b1.overlaps(b2):
            w = min(b1.x1, b2.x1) - max(b1.x0, b2.x0); h = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
            if w > 1 and h > 1:
                bad += 1
                print(f"  [{n1}] x [{n2}] {w:.0f}x{h:.0f}px")
print("  无" if not bad else f"  共 {bad} 处")

print("== 文字-线/标记 ==")
bad = 0
for ln in ax.lines:
    pts = ax.transData.transform(ln.get_path().vertices)
    mk = ln.get_marker() != "None"
    for n, t, bb in items:
        if not (n.startswith(("note", "lab"))):
            continue
        hits = ((pts[:, 0] >= bb.x0) & (pts[:, 0] <= bb.x1) &
                (pts[:, 1] >= bb.y0) & (pts[:, 1] <= bb.y1)).sum()
        if hits > 0:
            bad += 1
            print(f"  [{n}] 压 {ln.get_color()} 线 {hits} 采样点")
print("  无" if not bad else f"  共 {bad} 处")

leg = ax.get_legend()
if leg is not None:
    lb = leg.get_window_extent(renderer=ren)
    print("== 文字-图例 ==")
    badl = 0
    for n, t, bb in items:
        if not n.startswith(("note", "lab")):
            continue
        if bb.overlaps(lb):
            badl += 1
            print(f"  [{n}] 压图例框")
    if not badl: print("  无")

print("== 越界 ==")
fw, fh = fig.canvas.get_width_height()
ok = True
for n, t, bb in items:
    if bb.x0 < 0 or bb.y0 < 0 or bb.x1 > fw or bb.y1 > fh:
        ok = False
        print(f"  [{n}] 超出画布")
if ok: print("  无")
