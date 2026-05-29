"""CreditForge CLI — the reproducible lifecycle as one command surface.

    python -m creditforge.cli generate     # synthetic GSE panels -> Bronze
    python -m creditforge.cli pipeline      # Bronze -> Silver -> Gold
    python -m creditforge.cli split         # show the out-of-time split
    python -m creditforge.cli train         # fit + calibrate PD/LGD, score OOT
    python -m creditforge.cli validate      # Gini/KS/calibration/PSI -> report
    python -m creditforge.cli governance    # SHAP, reason codes, fairness, card
    python -m creditforge.cli monitor       # PSI drift vs training baseline
    python -m creditforge.cli gates         # CI performance gates (exit code)
    python -m creditforge.cli all           # the whole thing, end to end
"""
from __future__ import annotations

import json

import typer

app = typer.Typer(add_completion=False, help="CreditForge / BONITAS lifecycle CLI")


@app.command()
def generate():
    """Generate synthetic GSE-schema panels into Bronze."""
    from creditforge.pipeline.generate import generate as gen
    gen()


@app.command()
def pipeline():
    """Build Silver (point-in-time target) then Gold (feature matrix)."""
    from creditforge.pipeline.silver import build_silver
    from creditforge.pipeline.gold import build_gold
    build_silver()
    build_gold()


@app.command()
def split():
    """Print the out-of-time split summary."""
    from creditforge.pipeline.split import make_split
    typer.echo(json.dumps(make_split().summary(), indent=2))


@app.command()
def train():
    """Fit + calibrate scorecard, challenger, LGD; score the OOT test set."""
    from creditforge.train import train_all
    train_all()


@app.command()
def validate():
    """Run the validation suite and write the report."""
    from creditforge.validation.report import build_report
    r = build_report()
    typer.echo(json.dumps(r["benchmark"], indent=2, default=float))


@app.command()
def governance():
    """SHAP global, fairness, decision policy, adverse-action card."""
    from creditforge.governance.report import build_governance
    build_governance()


@app.command()
def monitor():
    """PSI/CSI drift of the OOT batch vs the training baseline."""
    from creditforge.monitoring.drift import run_monitoring
    r = run_monitoring()
    typer.echo(f"score PSI={r['score_psi']} ({r['score_status']}) | "
               f"alerts={len(r['alerts'])} | healthy={r['healthy']}")


@app.command()
def gates():
    """Run CI performance gates; non-zero exit on failure."""
    from creditforge.eval.gates import main as gate_main
    raise typer.Exit(code=gate_main())


@app.command(name="all")
def run_all():
    """End-to-end: generate -> pipeline -> train -> validate -> governance -> monitor -> gates."""
    generate()
    pipeline()
    train()
    validate()
    governance()
    monitor()
    from creditforge.eval.gates import main as gate_main
    raise typer.Exit(code=gate_main())


if __name__ == "__main__":
    app()
