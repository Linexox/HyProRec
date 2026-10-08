"""Generate an editable SVG architecture figure for the HyProRec recommendation head."""

from html import escape
from pathlib import Path


OUT = Path(__file__).resolve().parent
W, H = 1600, 1000

INK = "#183642"
MUTED = "#5B6B73"
LINE = "#7B8C93"
TEAL = "#2A9D8F"
BLUE = "#457B9D"
GOLD = "#E9C46A"
ORANGE = "#F4A261"
CORAL = "#E76F51"
LAVENDER = "#8E7DBE"


def rect(x, y, w, h, fill="#FFFFFF", stroke=LINE, sw=2, r=14):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def text(x, y, value, *, size=20, fill=INK, anchor="middle", weight="400", italic=False):
    style = f'font-family="Arial, DejaVu Sans, sans-serif" font-size="{size}px" fill="{fill}" '
    style += f'text-anchor="{anchor}" font-weight="{weight}"'
    if italic:
        style += ' font-style="italic"'
    return f'<text x="{x}" y="{y}" {style}>{escape(value)}</text>'


def multiline(x, y, lines, *, size=18, fill=INK, leading=25, anchor="middle", weight="400"):
    out = [f'<text x="{x}" y="{y}" font-family="Arial, DejaVu Sans, sans-serif" '
           f'font-size="{size}px" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">']
    for idx, line in enumerate(lines):
        dy = 0 if idx == 0 else leading
        out.append(f'<tspan x="{x}" dy="{dy}px">{escape(line)}</tspan>')
    out.append('</text>')
    return ''.join(out)


def arrow(x1, y1, x2, y2, *, color=LINE, sw=2, dashed=False):
    dash = ' stroke-dasharray="8 6"' if dashed else ''
    return (f'<path d="M {x1} {y1} L {x2} {y2}" fill="none" stroke="{color}" '
            f'stroke-width="{sw}"{dash} marker-end="url(#arrow)"/>')


def build_svg():
    out = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="4" orient="auto" markerUnits="strokeWidth">
<path d="M 0 0 L 8 4 L 0 8 z" fill="{LINE}"/></marker></defs>
<rect width="100%" height="100%" fill="white"/>
''']

    out.append(text(35, 46, "Recommendation module: modality-specific preference and router fusion", size=30, anchor="start", weight="700"))
    out.append(text(35, 78, "A shared MLLM state is converted into four view-specific user preferences, scored against independent item tables, and fused by a learned router.", size=16, fill=MUTED, anchor="start"))
    out.append(text(35, 117, "1. MLLM representation", size=18, anchor="start", weight="700"))
    out.append(text(405, 117, "2. View-specific preference heads", size=18, anchor="start", weight="700"))
    out.append(text(405, 790, "3. Router fusion and ranking", size=18, anchor="start", weight="700"))

    out.append(rect(35, 155, 300, 555, fill="#EAF4F4", stroke=TEAL, sw=3, r=20))
    out.append(multiline(185, 230, ["MLLM hidden states H_b", "", "context tokens +", "hypergraph prompt tokens", "", "H_b ∈ R^(T × d_llm)"], size=21, leading=34, weight="600"))
    out.append(text(185, 685, "pooling mask m_b", size=16, fill=MUTED))

    views = [("TXT", BLUE, 155), ("IMG", TEAL, 300), ("VDO", ORANGE, 445), ("ADO", CORAL, 590)]
    pool_x, pool_w = 405, 285
    user_x, user_w = 725, 210
    item_x, item_w = 970, 270
    score_x, score_w = 1275, 240
    row_h = 105
    user_centers, score_centers = [], []

    for short, color, y in views:
        view = short.lower()
        out.append(rect(374, y, 10, row_h, fill=color, stroke=color, sw=0, r=3))
        out.append(text(392, y + 58, short, size=16, fill=color, anchor="end", weight="700"))
        out.append(rect(pool_x, y, pool_w, row_h, fill="#FFFFFF", stroke=color, sw=2.5, r=15))
        out.append(multiline(pool_x + pool_w / 2, y + 40, [f"{short} attention pooling", f"learned query q_{view}"], size=17, leading=26, weight="600"))
        out.append(rect(user_x, y, user_w, row_h, fill="#F7FBFB", stroke=color, sw=2.5, r=15))
        out.append(multiline(user_x + user_w / 2, y + 36, [f"u_b^{view}", f"norm(g_{view}(p_b^{view}))"], size=16, leading=25))
        out.append(rect(item_x, y, item_w, row_h, fill="#FFFDF6", stroke=GOLD, sw=2.2, r=15))
        out.append(multiline(item_x + item_w / 2, y + 34, [f"{short} item table", f"E^{view} = norm(W_i^{view} X^{view})"], size=15, leading=25, weight="600"))
        out.append(rect(score_x, y, score_w, row_h, fill="#FFF7F3", stroke=color, sw=2.5, r=15))
        out.append(multiline(score_x + score_w / 2, y + 37, [f"per-view logits s_b,:^{view}", f"u^{view} · E^{view}ᵀ / τ"], size=16, leading=25, weight="600"))

        mid = y + row_h / 2
        out.append(arrow(335, mid, pool_x, mid, color=color, sw=2.4))
        out.append(arrow(pool_x + pool_w, mid, user_x, mid, color=color, sw=2.4))
        out.append(arrow(user_x + user_w, y + 66, score_x, y + 66, color=color, sw=2.0))
        out.append(arrow(item_x + item_w, y + 39, score_x, y + 39, color=GOLD, sw=2.0))
        user_centers.append((user_x + user_w / 2, mid))
        score_centers.append((score_x + score_w / 2, mid))

    out.append(text(770, 745, "a_b,t^v = softmax_t(φ_v(H_b,t, q_v))", size=15, fill=MUTED))
    out.append(text(1090, 745, "p_b^v = Σ_t a_b,t^v H_b,t", size=15, fill=MUTED))

    out.append(rect(405, 820, 330, 95, fill="#F0F2FA", stroke=LAVENDER, sw=2.8, r=16))
    out.append(multiline(570, 858, ["recommendation router r(·)", "π_b = softmax(r([u_b^v]_v))"], size=17, leading=28, weight="600"))
    out.append(rect(790, 820, 315, 95, fill="#F4F8F2", stroke=TEAL, sw=2.8, r=16))
    out.append(multiline(947, 858, ["logit-level fusion", "s_b,: = Σ_v π_b^v s_b,:^v"], size=18, leading=28, weight="600"))
    out.append(rect(1160, 820, 300, 95, fill="#FFF4EE", stroke=CORAL, sw=2.8, r=16))
    out.append(multiline(1310, 858, ["final ranking", "TopK(s_b, :)"], size=19, leading=28, weight="600"))
    for x, y in user_centers:
        out.append(arrow(x, y, 510, 820, color=LAVENDER, sw=1.6, dashed=True))
    for x, y in score_centers:
        out.append(arrow(x, y, 935, 820, color=LINE, sw=1.4, dashed=True))
    out.append(arrow(735, 868, 790, 868, color=LAVENDER, sw=2.5))
    out.append(arrow(1105, 868, 1160, 868, color=TEAL, sw=2.5))

    out.append(text(35, 810, "Independent user projectors and item-table projections per view; shared initialization does not mean shared parameters.", size=14, fill=MUTED, anchor="start", italic=True))
    out.append(rect(35, 944, 1425, 40, fill="#F8F9FA", stroke="#CBD5D9", sw=1.5, r=10))
    out.append(text(750, 970, "Training: L = L_fuse + λ_view L_view + λ_bpr L_bpr + λ_bal L_bal + λ_conv L_conv    |    current: λ_view = 0.2, λ_conv = 1, λ_bpr = λ_bal = 0", size=14))
    out.append('</svg>')
    return ''.join(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "recommendation_module_architecture.svg").write_text(build_svg(), encoding="utf-8")


if __name__ == "__main__":
    main()
