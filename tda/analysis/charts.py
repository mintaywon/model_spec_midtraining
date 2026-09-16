"""SVG chart generators for the supervisor report."""
from html import escape
import math

W, PX0, PW = 760, 210, 500

def _t(x): return f"{x:.1f}"

def dose_response():
    """Line chart: rate vs k, log-x. The headline positive result.

    No inline series labels — at these values the five first-points sit within
    7px of each other, so they collided. The legend carries identity instead.
    """
    ks = [64, 320, 640, 1280]
    series = [
        ("EK-FAC · opponents removed", [.590,.595,.650,.610], "var(--s1)", False),
        ("SOURCE · opponents removed", [.545,.570,.565,.570], "var(--s1)", True),
        ("random control",             [.550,.600,.525,.550], "var(--ctrl)", False),
        ("SOURCE · proponents removed",[.565,.530,.520,.515], "var(--s2)", True),
        ("EK-FAC · proponents removed",[.585,.575,.550,.520], "var(--s2)", False),
    ]
    L, R = 62, 24
    PWD = W - L - R
    H, top, bot = 292, 16, 236
    ymin, ymax = 0.50, 0.665
    def X(k): return L + (math.log2(k)-6)/(math.log2(1280)-6) * PWD
    def Y(v): return bot - (v-ymin)/(ymax-ymin) * (bot-top)
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" class="chart" aria-label="Removal dose-response by k">']
    for gv in (0.50,0.55,0.60,0.65):
        o.append(f'<line x1="{L}" y1="{Y(gv):.1f}" x2="{L+PWD}" y2="{Y(gv):.1f}" stroke="var(--grid)" stroke-width="1"/>')
        o.append(f'<text x="{L-10}" y="{Y(gv)+4:.1f}" class="tick" text-anchor="end">{gv:.2f}</text>')
    for k in ks:
        o.append(f'<text x="{X(k):.1f}" y="{bot+22}" class="tick" text-anchor="middle">k = {k}</text>')
        o.append(f'<text x="{X(k):.1f}" y="{bot+36}" class="tick dim" text-anchor="middle">{k/6400:.0%} of corpus</text>')
    o.append(f'<text x="{L+PWD/2:.1f}" y="{H-4}" class="axlab" text-anchor="middle">documents removed, log scale</text>')
    o.append(f'<text x="{L-10}" y="{top+2}" class="axlab" text-anchor="end">rate</text>')
    for lab, vals, col, dash in series:
        d = " ".join(f"{'M' if i==0 else 'L'}{X(k):.1f},{Y(v):.1f}" for i,(k,v) in enumerate(zip(ks,vals)))
        da = ' stroke-dasharray="6 4"' if dash else ""
        o.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2.2"{da} stroke-linejoin="round"/>')
        for k,v in zip(ks,vals):
            o.append(f'<circle cx="{X(k):.1f}" cy="{Y(v):.1f}" r="4" fill="{col}" stroke="var(--card)" stroke-width="2"/>')
    o.append("</svg>")
    return "\n".join(o)

def dose_legend():
    """Legend carrying both hue (direction) and dash (method)."""
    items = [("EK-FAC · opponents","var(--s1)",False),("SOURCE · opponents","var(--s1)",True),
             ("random control","var(--ctrl)",False),
             ("EK-FAC · proponents","var(--s2)",False),("SOURCE · proponents","var(--s2)",True)]
    out=['<div class="legend">']
    for lab,col,dash in items:
        da=' stroke-dasharray="5 3"' if dash else ''
        sw=(f'<svg width="24" height="8" aria-hidden="true"><line x1="1" y1="4" x2="23" y2="4" '
            f'stroke="{col}" stroke-width="2.4"{da}/></svg>')
        out.append(f'<span class="lg">{sw}{escape(lab)}</span>')
    out.append('</div>')
    return "".join(out)

def control_strip():
    """Number line: six control draws, their band, and where each method arm sits."""
    ctrl = [0.585,0.530,0.560,0.525,0.595,0.585]
    mean, sd = 0.5633, 0.0301
    arms = [("EK-FAC opponents",0.6317,"var(--s1)"),("grad-dot opponents",0.600,"var(--s1)"),
            ("SOURCE opponents",0.595,"var(--s1)"),("ICL v2 opponents",0.585,"var(--s3)"),
            ("grad-dot proponents",0.535,"var(--s2)"),("ICL v2 proponents",0.540,"var(--s2)")]
    lo, hi = 0.505, 0.65
    H = 214; axis = 150
    def X(v): return PX0 - 150 + (v-lo)/(hi-lo) * (PW + 150)
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" class="chart" aria-label="Control spread against method arms">']
    o.append(f'<rect x="{X(mean-sd):.1f}" y="46" width="{X(mean+sd)-X(mean-sd):.1f}" height="{axis-46+16}" fill="var(--bandfill)"/>')
    o.append(f'<line x1="{X(mean):.1f}" y1="42" x2="{X(mean):.1f}" y2="{axis+16}" stroke="var(--ctrl)" stroke-width="2"/>')
    o.append(f'<text x="{X(mean):.1f}" y="34" class="slab" text-anchor="middle" fill="var(--ctrl)">control mean 0.563</text>')
    o.append(f'<text x="{X(mean):.1f}" y="{axis+34}" class="tick" text-anchor="middle">± 1 sd (0.030)</text>')
    o.append(f'<line x1="{X(lo):.1f}" y1="{axis}" x2="{X(hi):.1f}" y2="{axis}" stroke="var(--rule2)" stroke-width="1"/>')
    for v in (0.52,0.56,0.60,0.64):
        o.append(f'<line x1="{X(v):.1f}" y1="{axis}" x2="{X(v):.1f}" y2="{axis+5}" stroke="var(--rule2)" stroke-width="1"/>')
        o.append(f'<text x="{X(v):.1f}" y="{axis+19}" class="tick" text-anchor="middle">{v:.2f}</text>')
    for v in ctrl:
        o.append(f'<circle cx="{X(v):.1f}" cy="{axis}" r="5" fill="var(--ctrl)" stroke="var(--card)" stroke-width="1.5"/>')
    o.append(f'<text x="{X(lo):.1f}" y="{axis+36}" class="tick dim">six random-removal arms, k=640</text>')
    for i,(lab,v,col) in enumerate(arms):
        y = 62 + i*14
        o.append(f'<circle cx="{X(v):.1f}" cy="{y}" r="4" fill="{col}"/>')
        side = "end" if v < mean else "start"
        dx = -9 if v < mean else 9
        o.append(f'<text x="{X(v)+dx:.1f}" y="{y+4}" class="slab" text-anchor="{side}">{escape(lab)}</text>')
    o.append(f'<line x1="{X(0.595):.1f}" y1="46" x2="{X(0.595):.1f}" y2="{axis}" stroke="var(--ink2)" stroke-width="1" stroke-dasharray="3 3"/>')
    o.append(f'<text x="{X(0.595):.1f}" y="{H-6}" class="tick" text-anchor="middle">baseline 0.595</text>')
    o.append("</svg>")
    return "\n".join(o)

def hbars(rows, vmax, fmt="{:+.3f}", ref=None, ref_label="", vmin=0.0):
    n=len(rows); BH, GAP = 24, 11
    h = n*(BH+GAP) + 34
    o=[f'<svg viewBox="0 0 {W} {h}" role="img" class="chart">']
    span = vmax - vmin
    def X(v): return PX0 + (v-vmin)/span*PW
    if ref is not None:
        o.append(f'<line x1="{X(ref):.1f}" y1="4" x2="{X(ref):.1f}" y2="{n*(BH+GAP)}" stroke="var(--rule2)" stroke-width="1" stroke-dasharray="4 3"/>')
        o.append(f'<text x="{X(ref):.1f}" y="{n*(BH+GAP)+20}" class="tick" text-anchor="middle">{escape(ref_label)}</text>')
    for i,r in enumerate(rows):
        lab,v = r[0], r[1]; col = r[2] if len(r)>2 else "var(--s1)"
        y=i*(BH+GAP)+4
        x0, x1 = X(min(vmin,0) if vmin<0 else vmin), X(v)
        left = min(X(0) if vmin<0 else vmin_x(vmin), x1)
        o.append(f'<text x="{PX0-12}" y="{y+BH*0.7:.1f}" class="blab" text-anchor="end">{escape(lab)}</text>')
        base = X(0) if vmin < 0 <= vmax else PX0
        w = abs(x1-base)
        o.append(f'<rect x="{min(base,x1):.1f}" y="{y}" width="{max(w,2):.1f}" height="{BH}" rx="3" fill="{col}"/>')
        tx = x1 + (8 if x1>=base else -8); an = "start" if x1>=base else "end"
        o.append(f'<text x="{tx:.1f}" y="{y+BH*0.7:.1f}" class="bval" text-anchor="{an}">{fmt.format(v)}</text>')
    if vmin < 0 <= vmax:
        o.append(f'<line x1="{X(0):.1f}" y1="0" x2="{X(0):.1f}" y2="{n*(BH+GAP)}" stroke="var(--rule2)" stroke-width="1"/>')
    o.append("</svg>")
    return "\n".join(o)

def vmin_x(v): return PX0
