"""Master orchestrator for the autonomous self-evolution loop (v1.0).

Coordinates triage, optimization, and reporting into a single
automated workflow that covers all 5 phases of evolution.
"""

import argparse
import json
import subprocess
import sys
import os
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from rich.console import Console
from rich.panel import Panel

from evolution.core.config import EvolutionConfig

console = Console()

class EvolutionLoop:
    """Orchestrates the full end-to-end self-evolution cycle."""

    def __init__(self, config: EvolutionConfig, max_targets: int = 5):
        self.config = config
        self.max_targets = max_targets
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path("logs") / f"loop_{self.timestamp}"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def run_triage(self) -> List[Dict[str, Any]]:
        """Run universal triage to identify targets across all tiers."""
        console.print("[bold yellow]1. Running Universal Performance Triage...[/bold yellow]")
        
        cmd = [sys.executable, "-m", "evolution.monitor.triage"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            console.print(f"[red]✗ Triage failed:[/red]\n{result.stderr}")
            return []

        report_path = Path("output") / "triage_report.json"
        if not report_path.exists():
            return []
            
        return json.loads(report_path.read_text())

    def evolve_target(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """Run evolution for a single target based on its type."""
        name = target["name"]
        artifact_type = target["type"]
        console.print(f"\n[bold yellow]2. Evolving {artifact_type}: {name}[/bold yellow]")
        
        if artifact_type == "skill":
            module = "evolution.skills.evolve_skill"
            args = ["--skill", name]
        elif artifact_type == "tool":
            module = "evolution.tools.evolve_tool"
            args = ["--tool", name]
        elif artifact_type == "prompt":
            module = "evolution.prompts.evolve_prompt_section"
            args = ["--section", name]
        else:
            console.print(f"[red]Unknown artifact type: {artifact_type}[/red]")
            return {"name": name, "success": False, "improvement": 0.0}

        # Common args
        cmd = [
            sys.executable, "-m", module
        ] + args + [
            "--iterations", str(self.config.iterations),
            "--eval-source", "synthetic" # Default to synthetic for stability in the loop
        ]

        # If running on a system with limited models, allow two-family mode
        if self.config.allow_two_family_mode:
            cmd.append("--allow-two-family-mode")

        # Log to file
        log_file = self.log_dir / f"{artifact_type}_{name.replace(':', '__')}.log"
        run_started = time.time()
        with open(log_file, "w") as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)

        success = result.returncode == 0

        # Each phase module writes metrics to a different output subtree
        if artifact_type == "tool":
            output_dir = Path("output") / "tools" / name
        elif artifact_type == "prompt":
            output_dir = Path("output") / "prompts" / name.replace(":", "__")
        else:
            output_dir = Path("output") / name

        # Only accept metrics written by this run, never a previous one
        metrics = {}
        if output_dir.exists():
            runs = sorted(
                p for p in output_dir.glob("*/metrics.json")
                if p.stat().st_mtime >= run_started
            )
            if runs:
                try:
                    metrics = json.loads(runs[-1].read_text())
                except (json.JSONDecodeError, OSError):
                    pass
        
        return {
            "name": name,
            "type": artifact_type,
            "success": success,
            "improvement": metrics.get("improvement", 0.0),
            "log": str(log_file),
            "metrics": metrics
        }

    def run(self):
        """Execute the full v1.0 loop."""
        console.print(Panel.fit(
            "[bold cyan]Hermes Agent Self-Evolution Loop v1.0[/bold cyan]\n"
            "Autonomous multi-tier optimization engine active.",
            style="bold cyan"
        ))
        
        # 1. Triage
        targets = self.run_triage()
        # Zero priority means no usage evidence — evolving it would burn
        # API budget on an arbitrary target
        targets = [t for t in targets if t.get("priority_score", 0) > 0]
        if not targets:
            console.print("[yellow]No targets identified. Loop complete.[/yellow]")
            return

        # 2. Evolve
        results = []
        # Process a mix of types if possible
        for target in targets[:self.max_targets]:
            res = self.evolve_target(target)
            results.append(res)

        # 3. Final Reporting
        self.generate_report(results)

    def generate_report(self, results: List[Dict[str, Any]]):
        """Create a master markdown report of the loop run."""
        report_path = Path("reports") / f"loop_report_{self.timestamp}.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        lines = [
            f"# 🤖 Hermes Self-Evolution Loop Report — {datetime.now().strftime('%Y-%m-%d')}",
            "",
            "## Summary",
            f"- **Timestamp:** {self.timestamp}",
            f"- **Targets Evaluated:** {len(results)}",
            f"- **Successful Evolutions:** {len([r for r in results if r['success'] and r['improvement'] > 0])}",
            "",
            "## Evolution Results",
            "| Artifact | Type | Status | Improvement |",
            "| :--- | :--- | :--- | :--- |"
        ]
        
        for r in results:
            status = "✅ Success" if r["success"] and r["improvement"] > 0 else "❌ Failed/No-op"
            # improvement is an absolute fitness-score delta, not a percentage
            improvement = f"{r['improvement']:+.3f}" if r["success"] else "N/A"
            lines.append(f"| {r['name']} | {r['type']} | {status} | {improvement} |")
            
        lines.append("")
        lines.append("## Logs")
        for r in results:
            lines.append(f"- **{r['name']} ({r['type']}):** `{r['log']}`")
            
        report_path.write_text("\n".join(lines))
        console.print(f"\n[bold green]✓ v1.0 Loop complete! Master report saved to {report_path}[/bold green]")

def main():
    parser = argparse.ArgumentParser(
        prog="python -m evolution.monitor.loop",
        description="Run the autonomous self-evolution loop: triage, evolve, report.",
    )
    parser.add_argument(
        "--iterations", type=int,
        default=int(os.environ.get("EVOLUTION_ITERATIONS", 1)),
        help="Optimization iterations per target (default: $EVOLUTION_ITERATIONS or 1)",
    )
    parser.add_argument(
        "--max-targets", type=int, default=5,
        help="Maximum number of triage targets to evolve per run (default: 5)",
    )
    args = parser.parse_args()

    config = EvolutionConfig()
    config.iterations = args.iterations
    loop = EvolutionLoop(config, max_targets=args.max_targets)
    loop.run()


if __name__ == "__main__":
    main()
