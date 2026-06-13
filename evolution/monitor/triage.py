"""Autonomous triage system for identifying optimization targets.

Scans session history to identify underperforming artifacts (Skills, Tools, Prompts)
based on usage frequency and heuristic failure detection.
"""

import json
import sqlite3
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from evolution.core.config import EvolutionConfig
from evolution.tools.tool_loader import discover_tool_schemas

class PerformanceTriage:
    """Identifies and ranks optimization targets from real usage."""

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.hermes_path = config.hermes_agent_path
        self.session_db = Path(os.environ.get("HERMES_SESSION_DB", Path.home() / ".hermes" / "state.db"))
        self.session_dir = Path.home() / ".hermes" / "sessions"
        self.error_log = Path.home() / ".hermes" / "logs" / "errors.log"

    def get_all_skills(self) -> List[str]:
        """List all available skills in the Hermes repo."""
        skills_dir = self.hermes_path / "skills"
        if not skills_dir.exists():
            return []
        
        skills = []
        for p in skills_dir.rglob("SKILL.md"):
            skills.append(p.parent.name)
        return sorted(list(set(skills)))

    def get_all_tools(self) -> List[str]:
        """List all available tools in the Hermes repo."""
        schemas = discover_tool_schemas(self.hermes_path)
        return sorted([s.name for s in schemas])

    def get_all_prompt_sections(self) -> List[str]:
        """List standard prompt sections."""
        return [
            "DEFAULT_AGENT_IDENTITY",
            "MEMORY_GUIDANCE",
            "SESSION_SEARCH_GUIDANCE",
            "SKILLS_GUIDANCE"
        ]

    def mine_usage_stats(self, name: str, artifact_type: str) -> Dict[str, Any]:
        """Heuristically count usage and failures for an artifact."""
        usage_count = 0
        potential_failures = 0
        
        # Heuristic patterns for different types
        if artifact_type == "skill":
            search_pattern = f"%{name}%"
        elif artifact_type == "tool":
            search_pattern = f"%tool_call%\"name\": \"{name}\"%"
        else: # prompt
            search_pattern = f"%{name}%"

        # 1. Check SQLite DB
        if self.session_db.exists():
            try:
                conn = sqlite3.connect(str(self.session_db))
                cursor = conn.cursor()
                
                query = "SELECT session_id, content FROM messages WHERE content LIKE ?"
                cursor.execute(query, (search_pattern,))
                
                rows = cursor.fetchall()
                sessions = set()
                for row in rows:
                    sessions.add(row[0])
                
                usage_count += len(sessions)
                
                # Heuristic failure detection: check session length
                for session_id in sessions:
                    cursor.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,))
                    msg_count = cursor.fetchone()[0]
                    if msg_count > 15:
                        potential_failures += 1
                
                conn.close()
            except sqlite3.Error as e:
                # A schema mismatch here would silently zero out usage stats
                print(f"warning: could not query session DB {self.session_db}: {e}",
                      file=sys.stderr)

        # 2. Check JSON session dumps
        if self.session_dir.exists():
            for p in self.session_dir.glob("*.json"):
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    dump_str = json.dumps(data)
                    if name in dump_str:
                        usage_count += 1
                        if len(dump_str) > 50000:
                            potential_failures += 0.5
                except (json.JSONDecodeError, OSError):
                    continue
        
        # 3. Check error log for Phase 4 (Code) targets
        if artifact_type == "tool" and self.error_log.exists():
            try:
                log_content = self.error_log.read_text(encoding="utf-8")
                # Look for tracebacks mentioning the tool file
                if f"tools/{name}.py" in log_content:
                    potential_failures += 2 # Strong signal for code evolution
            except OSError:
                pass

        return {
            "name": name,
            "type": artifact_type,
            "usage": usage_count,
            "potential_failures": int(potential_failures),
            "failure_rate": potential_failures / max(1, usage_count)
        }

    def run(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Rank all artifacts and return top targets for optimization."""
        all_targets = []
        
        # Phase 1: Skills
        for skill in self.get_all_skills():
            all_targets.append(self.mine_usage_stats(skill, "skill"))
            
        # Phase 2: Tools
        for tool in self.get_all_tools():
            all_targets.append(self.mine_usage_stats(tool, "tool"))
            
        # Phase 3: Prompts
        for prompt in self.get_all_prompt_sections():
            all_targets.append(self.mine_usage_stats(prompt, "prompt"))

        for target in all_targets:
            # Ranking score: usage frequency * failure rate
            # Add a small boost for tools and prompts to ensure they get evolved
            type_boost = 1.2 if target["type"] in ["tool", "prompt"] else 1.0
            target["priority_score"] = target["usage"] * (1 + target["failure_rate"]) * type_boost
        
        # Sort by priority score descending
        all_targets.sort(key=lambda x: x["priority_score"], reverse=True)
        return all_targets[:top_n]

def main():
    """CLI for triage."""
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    config = EvolutionConfig()
    triage = PerformanceTriage(config)
    
    console.print("[bold cyan]🔍 Universal Performance Triage (Phases 1-3)[/bold cyan]\n")
    
    targets = triage.run(top_n=15)
    
    table = Table(title="Triage Results (Universal Optimization Queue)")
    table.add_column("Artifact", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Usage", justify="right")
    table.add_column("Failures (Est.)", justify="right")
    table.add_column("Failure Rate", justify="right")
    table.add_column("Priority Score", justify="right", style="bold green")
    
    for t in targets:
        table.add_row(
            t["name"],
            t["type"],
            str(t["usage"]),
            str(t["potential_failures"]),
            f"{t['failure_rate']:.1%}",
            f"{t['priority_score']:.2f}"
        )
    
    console.print(table)
    
    # Save to file for orchestrator
    output_path = Path("output") / "triage_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(targets, indent=2))
    console.print(f"\n[green]✓ Universal triage report saved to {output_path}[/green]")

if __name__ == "__main__":
    main()
