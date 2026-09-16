from charts import dose_response, dose_legend, control_strip, hbars

GAP = hbars([("EK-FAC", .075, "var(--s1)"), ("grad-dot", .065, "var(--s1)"),
             ("ICL v2 — improved", .045, "var(--s3)"), ("SOURCE", .035, "var(--s1)"),
             ("marginal ICL", .025, "var(--s3)"), ("ICL v1 — starting point", .005, "var(--ctrl)")],
            0.085, fmt="{:+.3f}", vmin=0.0)

SCORE = hbars([("EK-FAC · opponents", .068, "var(--s1)"), ("grad-dot · opponents", .037, "var(--ctrl)"),
               ("SOURCE · opponents", .032, "var(--ctrl)"),
               ("“remove longest” · null criterion", .030, "var(--warn)"),
               ("ICL v2 · opponents", .022, "var(--ctrl)"),
               ("EK-FAC · proponents", -.005, "var(--ctrl)"),
               ("ICL v2 · proponents", -.023, "var(--ctrl)"),
               ("grad-dot · proponents", -.028, "var(--ctrl)")],
              0.085, fmt="{:+.3f}", vmin=-0.045)

NULL32 = hbars([("midtraining documents", 14.74, "var(--s1)"), ("AFT + instruction-tuning rows", 5.55, "var(--ctrl)")],
               17.0, fmt="{:.2f}", vmin=0.0)

COST = hbars([("grad-dot", 23, "var(--s3)"), ("EK-FAC", 64, "var(--s1)")], 80, fmt="{:.0f} min", vmin=0.0)

CH = dict(DOSE=dose_response(), DOSELEG=dose_legend(), CTRL=control_strip(), GAP=GAP, SCORE=SCORE, NULL32=NULL32, COST=COST)
import json; json.dump(CH, open("ch.json","w")); print("charts built:", list(CH))
