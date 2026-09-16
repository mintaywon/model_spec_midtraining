import json
CH = json.load(open("ch.json"))

CSS = """
:root{
  color-scheme: light;
  --paper:#f7f8f9; --card:#ffffff; --ink:#16191c; --ink2:#454b52; --muted:#6f767d;
  --rule:#e2e5e8; --rule2:#b4bbc1; --grid:#eceff1; --accent:#2f4b7c;
  --s1:#2f6fb0; --s2:#c96a1f; --s3:#1f7a6f; --ctrl:#8d949b; --warn:#a8482a;
  --bandfill:#eceff2; --okbg:#eef4f3; --warnbg:#faf1ec;
  --serif:"Spectral",Georgia,"Times New Roman",serif;
  --sans:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  color-scheme: dark;
  --paper:#14171a; --card:#1b1f23; --ink:#e9ecef; --ink2:#c2c8ce; --muted:#949ba2;
  --rule:#2a2f34; --rule2:#5a626a; --grid:#23282d; --accent:#8fb0dd;
  --s1:#66a3dd; --s2:#e0894a; --s3:#4aa99b; --ctrl:#8d949b; --warn:#d9885f;
  --bandfill:#23282d; --okbg:#182220; --warnbg:#241c17;
}}
:root[data-theme="dark"]{
  color-scheme: dark;
  --paper:#14171a; --card:#1b1f23; --ink:#e9ecef; --ink2:#c2c8ce; --muted:#949ba2;
  --rule:#2a2f34; --rule2:#5a626a; --grid:#23282d; --accent:#8fb0dd;
  --s1:#66a3dd; --s2:#e0894a; --s3:#4aa99b; --ctrl:#8d949b; --warn:#d9885f;
  --bandfill:#23282d; --okbg:#182220; --warnbg:#241c17;
}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);font-family:var(--sans);
  font-size:16px;line-height:1.65;margin:0;padding:0 20px 90px}
.wrap{max-width:880px;margin:0 auto}
header{padding:58px 0 24px;border-bottom:2px solid var(--ink)}
h1{font-family:var(--serif);font-weight:600;font-size:clamp(30px,4.4vw,42px);
  line-height:1.12;letter-spacing:-0.01em;margin:0 0 14px;text-wrap:balance}
.dek{font-size:17.5px;color:var(--ink2);margin:0 0 22px;max-width:62ch}
.meta{font-family:var(--mono);font-size:12px;color:var(--muted);
  display:flex;flex-wrap:wrap;gap:6px 22px}
section{padding-top:46px}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:0.11em;
  text-transform:uppercase;color:var(--accent);margin:0 0 8px}
h2{font-family:var(--serif);font-weight:600;font-size:26px;line-height:1.25;
  margin:0 0 16px;letter-spacing:-0.005em;text-wrap:balance}
p{max-width:70ch;margin:0 0 14px}
ul{max-width:70ch;margin:0 0 8px;padding-left:0;list-style:none}
li{position:relative;padding-left:20px;margin-bottom:10px}
li::before{content:"";position:absolute;left:2px;top:0.68em;width:6px;height:6px;
  border-radius:50%;background:var(--accent)}
li b{font-weight:600}
.fig{background:var(--card);border:1px solid var(--rule);border-radius:9px;
  padding:22px 20px 14px;margin:0 0 16px;overflow-x:auto}
.chart{display:block;width:100%;height:auto;min-width:560px}
.blab{font-family:var(--sans);font-size:12.5px;fill:var(--ink2)}
.slab{font-family:var(--sans);font-size:11.5px;fill:var(--ink2)}
.bval{font-family:var(--mono);font-size:12.5px;fill:var(--ink);font-weight:500}
.tick{font-family:var(--mono);font-size:10.5px;fill:var(--muted)}
.tick.dim{fill:var(--rule2)}
.axlab{font-family:var(--sans);font-size:11.5px;fill:var(--muted)}
.cap{font-size:13.5px;color:var(--muted);max-width:70ch;margin:0 0 20px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
  gap:1px;background:var(--rule);border:1px solid var(--rule);border-radius:9px;
  overflow:hidden;margin:0 0 20px}
.cell{background:var(--card);padding:15px 17px}
.cell .k{font-family:var(--mono);font-size:10.5px;letter-spacing:0.07em;
  text-transform:uppercase;color:var(--muted);margin-bottom:5px}
.cell .v{font-family:var(--mono);font-size:20px;font-weight:500;
  font-variant-numeric:tabular-nums;color:var(--ink)}
.cell .s{font-size:12.5px;color:var(--ink2);margin-top:3px}
.note{border-left:3px solid var(--warn);background:var(--warnbg);
  border-radius:0 7px 7px 0;padding:15px 18px;margin:0 0 18px;font-size:15px}
.note b{color:var(--warn)}
.ok{border-left:3px solid var(--s3);background:var(--okbg);
  border-radius:0 7px 7px 0;padding:15px 18px;margin:0 0 18px;font-size:15px}
.ok b{color:var(--s3)}
table{border-collapse:collapse;width:100%;font-size:14px;margin:0 0 18px}
th,td{text-align:left;padding:9px 14px 9px 0;border-bottom:1px solid var(--rule)}
th{font-family:var(--mono);font-size:10.5px;letter-spacing:0.07em;
  text-transform:uppercase;color:var(--muted);font-weight:400}
td.n{font-family:var(--mono);font-variant-numeric:tabular-nums}
code{font-family:var(--mono);font-size:0.87em;background:var(--paper);
  border:1px solid var(--rule);border-radius:4px;padding:1px 5px}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:26px}
.colh{font-family:var(--mono);font-size:10.5px;letter-spacing:0.08em;
  text-transform:uppercase;margin:0 0 9px;color:var(--muted)}
.two ul{margin:0}.two li{font-size:14.5px}
.legend{display:flex;flex-wrap:wrap;gap:7px 18px;margin:0 0 16px;
  font-size:12.5px;color:var(--ink2);padding:0 2px}
.lg{display:flex;align-items:center;gap:7px;white-space:nowrap}
.lg svg{flex:none;overflow:visible}
hr{border:0;border-top:1px solid var(--rule);margin:46px 0 0}
"""

def sec(eyebrow, h2, body):
    return f'<section>\n<p class="eyebrow">{eyebrow}</p>\n<h2>{h2}</h2>\n{body}\n</section>'

BODY = f"""<title>Which Midtraining Documents Matter</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{CSS}</style>
<div class="wrap">
<header>
  <h1>Which Midtraining Documents Matter</h1>
  <p class="dek">Training-data attribution across a two-stage pipeline — model-spec
  midtraining followed by alignment finetuning. Progress report, 14 September.</p>
  <div class="meta">
    <span>Llama-3.1-8B cheese · Qwen2.5-32B philosophy</span>
    <span>SOURCE · EK-FAC · grad-dot · in-context</span>
    <span>bergson 0.26.2, patched</span>
  </div>
</header>

{sec("Question", "Which midtraining documents change behaviour <i>after</i> the downstream finetuning stage?", f'''
  <p>Model-spec midtraining (MSM) trains a model on documents discussing a spec
  <i>before</i> alignment finetuning (AFT), and it controls how the model generalises
  from that finetuning — cutting agentic misalignment 0.655 → 0.310 on the 32B arm we
  study. The attribution question is therefore multi-stage: not "which document mattered
  at the end of midtraining", but
  <code>τᵢ = U(A_AFT(A_MSM(D∖{{zᵢ}}))) − U(A_AFT(A_MSM(D)))</code> — the effect of
  dropping a midtraining document, measured after AFT has run on top.</p>
  <div class="grid">
    <div class="cell"><div class="k">Built</div><div class="v">4 methods</div><div class="s">SOURCE, EK-FAC, grad-dot, in-context — one harness</div></div>
    <div class="cell"><div class="k">Scales</div><div class="v">8B &amp; 32B</div><div class="s">toy preference task and a real safety task</div></div>
    <div class="cell"><div class="k">Causal arms</div><div class="v">40+</div><div class="s">each a full midtraining + AFT retrain</div></div>
  </div>
  <p>Rankings are cheap to produce and easy to believe. The work below is mostly about
  whether they survive a causal test: remove the documents a method flags, retrain both
  stages, and see whether behaviour moves more than it does when you remove the same
  number at random.</p>''')}

{sec("Headline result", "Removing more of what a method calls a proponent monotonically lowers alignment", f'''
  <div class="fig">{CH["DOSELEG"]}{CH["DOSE"]}</div>
  <p class="cap">Each point is an independently trained model: midtraining rerun without
  the selected documents, then AFT rerun on top. Solid = EK-FAC, dashed = SOURCE.</p>
  <ul>
    <li><b>Both proponent arms fall perfectly monotonically across k</b> (Spearman −1.00
    each), both opponent arms rise, and the random control is flat. This is the
    directional evidence the project was missing.</li>
    <li><b>The two methods' removal sets are effectively disjoint</b> — Jaccard 0.009 —
    so two independent selections both coming out perfectly monotone is
    <b>p ≈ 0.0017</b> under exchangeability.</li>
    <li><b>The within-method gap needs no control at all.</b> At a fixed k, the opponents
    and proponents arms share the same random draw, so it cancels exactly — and the gap
    widens with k for both methods (+0.023 and +0.018 per doubling).</li>
    <li>This was already paid for and initially invisible: every arm was being read as a
    difference against a single noisy control at its own k, rather than as a
    dose-response.</li>
  </ul>''')}

{sec("Correction", "The denominator was noise — and it inflated every number we had", f'''
  <div class="fig">{CH["CTRL"]}</div>
  <p class="cap">Six independent random-removal arms at k=640 (grey dots), their mean and
  ±1 sd (band), against where the method arms land.</p>
  <ul>
    <li><b>The control is not a point, it is a distribution: 0.563 ± 0.030</b>, spanning
    0.525 to 0.595. The single draw the project had been dividing by was
    <b>the lowest of the six</b>.</li>
    <li><b>Rescored against the mean, exactly one arm on the board is distinguishable
    from random removal</b> — EK-FAC's opponents, +0.068 (z = 3.07). The headline
    "+0.107" was that same arm measured against the low draw.</li>
    <li><b>The error model was wrong, not just the draw.</b> We were quoting McNemar,
    which pairs eval items and so treats the <i>training run</i> as fixed. Each arm is a
    separate training run, so run-to-run variance was being assigned zero weight — which
    is why z = 4.69 results failed to replicate across seeds.</li>
    <li><b>The unit of replication is the training run, not the eval item.</b> That is
    the methodological lesson of this phase, and it is why the dose-response above is
    the more trustworthy read of the same arms.</li>
  </ul>''')}

{sec("Scoreboard", "Against a properly estimated control, most of the board is null", f'''
  <div class="fig">{CH["SCORE"]}</div>
  <p class="cap">Change in value-aligned decision rate vs the six-arm control mean, k=640.
  Blue = distinguishable from random; grey = not; rust = a criterion carrying no
  influence information at all.</p>
  <ul>
    <li><b>A null criterion scores +0.030.</b> "Remove the 640 longest documents" uses no
    influence signal whatsoever — and <b>six of nine influence-selected arms do not beat
    it</b>. Any method claiming a win has to clear this bar first.</li>
    <li><b>EK-FAC's opponent arm is the one real result</b> at this k, and its
    proponent arm is not (−0.005).</li>
    <li><b>Reading a single k was the mistake.</b> At k=320 every method arm sits
    <i>below</i> random; at k=640 every one sits above it. The methods did not change —
    the denominator moved.</li>
  </ul>''')}

{sec("Method comparison", "The expensive machinery does not buy a better ranking", f'''
  <table>
    <thead><tr><th>pair</th><th>Spearman, 32B</th><th>Spearman, 8B</th></tr></thead>
    <tbody>
      <tr><td>EK-FAC ↔ grad-dot</td><td class="n">0.566</td><td class="n">0.628</td></tr>
      <tr><td>SOURCE ↔ EK-FAC</td><td class="n">—</td><td class="n">0.411</td></tr>
      <tr><td>SOURCE ↔ grad-dot</td><td class="n">—</td><td class="n">0.245</td></tr>
    </tbody>
  </table>
  <div class="fig">{CH["COST"]}</div>
  <p class="cap">End-to-end wall time at 32B on 8×B200, sharing a query index.</p>
  <ul>
    <li><b>The two single-checkpoint methods agree with each other far more than either
    agrees with SOURCE</b>, at both scales. That is the expected shape: EK-FAC is
    grad-dot plus a curvature preconditioner, so 0.57–0.63 measures what the
    preconditioner changes and 0.25–0.41 measures what the trajectory changes.</li>
    <li><b>grad-dot costs 23 minutes against EK-FAC's 64</b> — a 2.8× premium for a
    ranking that agrees at ρ ≈ 0.57. At 8B the spread is starker: 10 min, 92 min, and
    <b>235 min for SOURCE</b>.</li>
    <li><b>SOURCE never justified its cost.</b> It ran at the paper's own multi-stage
    recipe (L=2, TDA on the first segment, matching per-segment checkpoint density) and
    did not beat EK-FAC — a non-replication of the setting SOURCE was designed for.</li>
    <li><b>Agreement is not correctness.</b> What separates them is the causal arm, and
    there EK-FAC leads.</li>
  </ul>''')}

{sec("Scale", "The 32B port works, and its sanity checks pass on their own", f'''
  <div class="fig">{CH["NULL32"]}</div>
  <p class="cap">Mean |influence| by row type, EK-FAC over 13,201 philosophy midtraining
  documents plus 1,584 finetuning rows as a null control.</p>
  <div class="ok"><b>The blocker we planned around did not exist.</b> EK-FAC factors were
  predicted at ~7.8 TB — over the 3.3 TB cap, which is why an attention-only
  approximation had been planned. Measured: <b>650 GB</b>. All 7 projections, no module
  scope sacrificed, 85 minutes on 8×B200 for ~$77.</div>
  <ul>
    <li><b>The null-distribution control passes.</b> Instruction-tuning and AFT rows are
    2.7–4.5× less influential than midtraining documents, with signed means near zero and
    a near coin-flip sign split. Nothing was engineered to make this true.</li>
    <li><b>The sign agrees with the measured behaviour.</b> 72% of midtraining documents
    are <i>opponents</i> of the misaligned action — and midtraining does cut misalignment
    0.655 → 0.310 on this checkpoint pair. A method that got this backwards would still
    produce a plausible-looking ranking.</li>
    <li><b>Row alignment verified the hard way</b>: both methods independently rank the
    same document as their strongest proponent, and every returned document's text
    matches its metadata label. A permuted join leaves every aggregate statistic
    unchanged, so this is the check that catches it.</li>
    <li><b>Influence is concentrated by document, not by domain</b> — top 10% of documents
    carry 30% of the mass, while the corpus's own domain labels explain only 1–2% of
    variance. So a domain-level ablation would test almost none of the signal.</li>
  </ul>''')}

{sec("Method development", "In-context attribution improved 9× and is still behind", f'''
  <div class="fig">{CH["GAP"]}</div>
  <p class="cap">Opponents-minus-proponents gap — the control-free directional measure,
  since both arms of a pair share the same random draw.</p>
  <ul>
    <li><b>A gradient-free method reaches third place.</b> In-context scoring asks what
    happens when a document is <i>read</i> rather than trained on — no gradients, no
    Hessian, no trajectory. Its gap went from +0.005 to <b>+0.045</b>, past SOURCE.</li>
    <li><b>All of the gain came from the readout</b>, not from a cleverer formulation:
    swapping a decision rate for a continuous margin.</li>
    <li><b>A pre-registered prediction failed on all three counts.</b> A "marginal"
    variant was more reliable, format-clean, and closest to the gradient methods on every
    property argued to matter — and predicted the counterfactual <i>worse</i>. We do not
    have an account of why.</li>
    <li><b>A false positive was caught by seeds.</b> At one seed that variant showed the
    proponent-below-random sign result the project had been hunting; across three seeds
    it is −0.002. Reported as a single-seed arm, it would have been published.</li>
  </ul>''')}

{sec("Assessment", "What this does and does not establish", f'''
  <div class="two">
    <div><p class="colh" style="color:var(--s3)">Established</p><ul>
      <li>Multi-stage attribution runs end-to-end at 8B and 32B, with per-segment
      curvature and a stage-masked score.</li>
      <li>The directional dose-response: proponent removal lowers alignment
      monotonically, opponent removal raises it, control flat (p ≈ 0.0017).</li>
      <li>32B engineering solved without approximation, and its null and sign checks
      pass unprompted.</li>
      <li>Cheap methods track expensive ones at both scales; SOURCE's cost is not repaid
      in this setting.</li>
    </ul></div>
    <div><p class="colh" style="color:var(--warn)">Not established</p><ul>
      <li>That any method beats a null criterion at a single k. Six of nine arms do not
      beat "remove the longest documents".</li>
      <li>Any 32B causal arm — the removal test there is unfunded, so 32B has rankings
      and sanity checks, not validation.</li>
      <li>That EK-FAC's top documents are semantically meaningful: at 8B its category
      structure was 92% surface lexical overlap with the query, and that control has not
      been rerun at 32B.</li>
      <li>Error bars on any 32B number — one run per method by design.</li>
    </ul></div>
  </div>
  <div class="note"><b>The honest summary.</b> The signal is real and directional, but it
  is small relative to the noise of retraining, and the 8B task is structurally
  underpowered to measure it: midtraining moves the metric by 0.216 in total, while a
  single arm carries ±0.035 of sampling error before training variance is counted. The
  paper's own numbers show the same narrow range, so this is a property of the task, not
  of our reproduction. The 32B setting has ~13× more evaluation samples and is where a
  causal result would actually be resolvable.</div>
  <p><b>Next, in order:</b> a 32B removal arm on EK-FAC's opponents — the one arm that
  survives a proper control, on the setting with the resolution to measure it; the
  lexical-overlap control at 32B; and a dose-response design rather than single-k arms
  for anything further at 8B.</p>''')}

<hr>
</div>"""
open("report.html","w").write(BODY)
print("report.html", len(BODY), "bytes")
