"""Master orchestrator for the autonomous self-evolution loop (v1.0).

Coordinates triage, optimization, and reporting into a single
automated workflow that covers all 5 phases of evolution.
"""

import json
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from rich.console import Console
from rich.panel import Panel

from evolution.core.config import EvolutionConfig

console = Console()

class EvolutionLoop:
    """Orchestrates the full end-to-end self-evolution cycle."""

    def __init__(self, config: EvolutionConfig):
        self.config = config
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
        if getattr(self.config, "allow_two_family_mode", True):
            cmd.append("--allow-two-family-mode")

        # Log to file
        log_file = self.log_dir / f"{artifact_type}_{name}.log"
        with open(log_file, "w") as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        
        success = result.returncode == 0
        
        # Try to find metrics from the output directory
        metrics = {}
        output_dir = Path("output") / name
        if output_dir.exists():
            runs = sorted(output_dir.glob("*/metrics.json"))
            if runs:
                try:
                    metrics = json.loads(runs[-1].read_text())
                except:
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
        if not targets:
            console.print("[yellow]No targets identified. Loop complete.[/yellow]")
            return

        # 2. Evolve
        results = []
        # Process a mix of types if possible
        max_targets = 5
        for target in targets[:max_targets]:
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
            improvement = f"{r['improvement']:+.1%}" if r["success"] else "N/A"
            lines.append(f"| {r['name']} | {r['type']} | {status} | {improvement} |")
            
        lines.append("")
        lines.append("## Logs")
        for r in results:
            lines.append(f"- **{r['name']} ({r['type']}):** `{r['log']}`")
            
        report_path.write_text("\n".join(lines))
        console.print(f"\n[bold green]✓ v1.0 Loop complete! Master report saved to {report_path}[/bold green]")

if __name__ == "__main__":
    config = EvolutionConfig()
    # Default iterations to 1 for loop speed if not specified
    config.iterations = int(os.environ.get("EVOLUTION_ITERATIONS", 1))
    loop = EvolutionLoop(config)
    loop.run()
