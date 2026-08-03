"""Project journey clusters to 2D and write a standalone interactive HTML map.

    python scripts/visualize_clusters.py --platform android
    python scripts/visualize_clusters.py --platform ios --route b

Projection backend, in order of preference:

    umap-learn   best structure preservation for this kind of sparse, clustered
                 embedding - install with `pip install umap-learn`
    sklearn TSNE fallback, always available, slower and less faithful globally

The output is one self-contained .html file - no CDN, no server, no plotting
library. Open it in a browser: hover a point to read its journey, click a legend
entry to isolate a cluster. This exists because the useful question is never
"where are the clusters" but "what is IN that blob", and that needs the token
sequence attached to every point.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from Rule_based import features as F  # noqa: E402
from Rule_based.config import PipelineConfig  # noqa: E402
from Rule_based.prefixspan import mine_patterns  # noqa: E402

IN_DIR = REPO_ROOT / "outputs" / "journeys"
OUT_DIR = REPO_ROOT / "outputs" / "clusters"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--platform", required=True, choices=["android", "ios"])
    p.add_argument("--route", default="a", choices=["a", "b"], help="a=tf-idf, b=prefixspan")
    p.add_argument("--max-ngram", type=int, default=4)
    p.add_argument("--output")
    return p.parse_args()


def project(matrix: np.ndarray) -> tuple[np.ndarray, str]:
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42)
        return reducer.fit_transform(matrix), "UMAP"
    except ImportError:
        from sklearn.manifold import TSNE

        n = matrix.shape[0]
        tsne = TSNE(
            n_components=2,
            perplexity=min(30, max(5, n // 100)),
            init="pca",
            random_state=42,
        )
        return tsne.fit_transform(matrix), "t-SNE"


# a colour-blind-safe qualitative ramp, cycled; noise is always grey
PALETTE = [
    "#4269d0", "#efb118", "#ff725c", "#6cc5b0", "#3ca951", "#ff8ab7",
    "#a463f2", "#97bbf5", "#9c6b4e", "#9498a0", "#e45756", "#72b7b2",
]


def build_html(points: pd.DataFrame, title: str, method: str) -> str:
    sizes = points.cluster.value_counts().to_dict()
    clusters = sorted(c for c in sizes if c != -1)
    colour = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(clusters)}
    colour[-1] = "#c9ccd1"

    data = [
        {
            "x": round(float(r.x), 3),
            "y": round(float(r.y), 3),
            "c": int(r.cluster),
            "j": str(r.journey_id),
            "n": int(r.n_events_final),
            "s": str(r.sequence)[:600],
        }
        for r in points.itertuples()
    ]
    legend = "".join(
        f'<button class="lg" data-c="{c}"><i style="background:{colour[c]}"></i>'
        f'cluster {c} <b>{sizes.get(c, 0)}</b></button>'
        for c in [-1] + clusters
    )

    return f"""<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
 :root {{ color-scheme: light dark; }}
 body {{ font:14px/1.5 ui-sans-serif,system-ui,sans-serif; margin:0; padding:20px;
        background:Canvas; color:CanvasText; }}
 h1 {{ font-size:18px; margin:0 0 4px; }}
 .sub {{ opacity:.65; margin-bottom:14px; }}
 #wrap {{ display:flex; gap:16px; flex-wrap:wrap; align-items:flex-start; }}
 canvas {{ border:1px solid rgba(128,128,128,.35); border-radius:8px; cursor:crosshair;
           max-width:100%; touch-action:none; }}
 #side {{ flex:1 1 300px; min-width:280px; max-height:720px; overflow:auto; }}
 .lg {{ display:flex; align-items:center; gap:8px; width:100%; text-align:left;
        background:none; border:0; padding:4px 6px; border-radius:6px; cursor:pointer;
        font:inherit; color:inherit; }}
 .lg:hover {{ background:rgba(128,128,128,.14); }}
 .lg.off {{ opacity:.3; }}
 .lg i {{ width:11px; height:11px; border-radius:3px; flex:none; }}
 .lg b {{ margin-left:auto; opacity:.6; font-weight:500; }}
 #tip {{ position:fixed; pointer-events:none; display:none; max-width:520px; z-index:9;
         background:Canvas; color:CanvasText; border:1px solid rgba(128,128,128,.5);
         border-radius:8px; padding:9px 11px; font-size:12px;
         box-shadow:0 6px 24px rgba(0,0,0,.22); }}
 #tip code {{ word-break:break-all; opacity:.85; }}
</style>
<h1>{html.escape(title)}</h1>
<div class="sub">{len(data):,} journeys &middot; {len(clusters)} clusters &middot;
 projected with {method} &middot; hover a point, click a cluster to isolate</div>
<div id="wrap">
 <canvas id="cv" width="820" height="720"></canvas>
 <div id="side">{legend}</div>
</div>
<div id="tip"></div>
<script>
const P={json.dumps(data)}, COL={json.dumps({str(k): v for k, v in colour.items()})};
const cv=document.getElementById('cv'), ctx=cv.getContext('2d'), tip=document.getElementById('tip');
const xs=P.map(p=>p.x), ys=P.map(p=>p.y);
const x0=Math.min(...xs), x1=Math.max(...xs), y0=Math.min(...ys), y1=Math.max(...ys);
const pad=34, W=cv.width, H=cv.height;
const sx=v=>pad+(v-x0)/((x1-x0)||1)*(W-2*pad), sy=v=>H-pad-(v-y0)/((y1-y0)||1)*(H-2*pad);
let hidden=new Set();
function draw(){{
  ctx.clearRect(0,0,W,H);
  for(const p of P){{
    if(hidden.size && !hidden.has(p.c)) continue;
    ctx.beginPath(); ctx.arc(sx(p.x),sy(p.y), p.c===-1?2:3.4, 0, 6.284);
    ctx.fillStyle=COL[p.c]; ctx.globalAlpha=p.c===-1?.4:.85; ctx.fill();
  }}
  ctx.globalAlpha=1;
}}
draw();
cv.addEventListener('mousemove', e=>{{
  const r=cv.getBoundingClientRect(), k=cv.width/r.width;
  const mx=(e.clientX-r.left)*k, my=(e.clientY-r.top)*k;
  let best=null, bd=90;
  for(const p of P){{
    if(hidden.size && !hidden.has(p.c)) continue;
    const d=(sx(p.x)-mx)**2+(sy(p.y)-my)**2;
    if(d<bd){{ bd=d; best=p; }}
  }}
  if(!best){{ tip.style.display='none'; return; }}
  tip.innerHTML='<b>'+best.j+'</b> &middot; cluster '+best.c+' &middot; '+best.n+
    ' events<br><code>'+best.s.replace(/&/g,'&amp;').replace(/</g,'&lt;')+'</code>';
  tip.style.display='block';
  tip.style.left=Math.min(e.clientX+14, innerWidth-540)+'px';
  tip.style.top=(e.clientY+16)+'px';
}});
cv.addEventListener('mouseleave',()=>tip.style.display='none');
document.querySelectorAll('.lg').forEach(b=>b.onclick=()=>{{
  const c=+b.dataset.c;
  hidden.has(c)?hidden.delete(c):hidden.add(c);
  document.querySelectorAll('.lg').forEach(o=>
    o.classList.toggle('off', hidden.size>0 && !hidden.has(+o.dataset.c)));
  draw();
}});
</script>
"""


def main() -> int:
    args = parse_args()
    tag = "a_tfidf" if args.route == "a" else "b_prefixspan"

    journeys = pd.read_csv(IN_DIR / f"{args.platform}_journey_features.csv")
    with (IN_DIR / f"{args.platform}_journey_sequences.json").open(encoding="utf-8") as fh:
        sequences = [r["tokens"] for r in json.load(fh)]
    labels_df = pd.read_csv(OUT_DIR / f"{args.platform}_{tag}_labels.csv")

    cfg = PipelineConfig()
    cfg.features.ngram_range = (1, args.max_ngram)
    if args.route == "a":
        vec = F.JourneyVectorizer(cfg.features)
    else:
        vec = F.PatternVectorizer(cfg.features, mine_patterns(sequences, cfg.prefixspan))
    matrix, _ = vec.fit_transform(journeys, sequences)

    print(f"projecting {matrix.shape[0]:,} x {matrix.shape[1]} ...")
    coords, method = project(matrix)
    print(f"  backend: {method}")

    points = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "cluster": labels_df["cluster"].to_numpy(),
            "journey_id": journeys["journey_id"],
            "n_events_final": journeys["n_events_final"],
            "sequence": [" -> ".join(s) for s in sequences],
        }
    )

    title = f"{args.platform} journey clusters - route {args.route.upper()} ({tag.split('_')[1]})"
    out = Path(args.output) if args.output else OUT_DIR / f"{args.platform}_{tag}_map.html"
    out.write_text(build_html(points, title, method), encoding="utf-8")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
