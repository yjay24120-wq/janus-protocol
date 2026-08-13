#!/usr/bin/env python3
"""
QUL6 Sovereign Engine v2 — Project Gemini Invictus
================================================
Architecture:
  - asyncio event-driven main loop
  - Blackboard shared memory (nodes read/write structured state)
  - Real LLM capability synthesis via Ollama (qwen2.5:14b)
  - Tool results feed back into entropy, score, and node lumen_internum
  - Rotating vault with configurable retention (default: 200 entries)
  - Distinct node objectives with cross-node awareness
"""

import asyncio
import json
import os
import re
import glob
import hashlib
import subprocess
import time
import urllib.parse
import urllib.request
import urllib.error
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

OLLAMA_URL       = "http://localhost:11434/api/generate"
OLLAMA_MODEL     = "qwen2.5:14b"
WORKSPACE_DIR    = "./qul6_workspace"
STATE_FILE       = "sovereign_state.json"
VAULT_MAX        = 200          # rotating snapshot retention
CYCLE_INTERVAL   = 0.25         # seconds between node ticks
TOOL_TIMEOUT     = 5            # subprocess / fetch timeout (s)
MAX_FILE_SIZE    = 10 * 1024 * 1024  # 10 MB OOM cap


# ──────────────────────────────────────────────
# SECURITY BOUNDARY (unchanged — it was solid)
# ──────────────────────────────────────────────

class SecurityBoundary:
    WORKSPACE_JAIL   = os.path.abspath(WORKSPACE_DIR)
    ALLOW_SYMLINKS   = False
    DANGEROUS_BINS   = {
        'rm','mkfs','dd','fdisk','format','kill','sudo','su',
        'chmod','chown','insmod','rmmod','reboot','shutdown',
        'ifconfig','iptables','firewall-cmd','systemctl',
    }

    @classmethod
    def validate_path(cls, path: str) -> bool:
        real = os.path.abspath(path)
        if not real.startswith(cls.WORKSPACE_JAIL):
            return False
        if not cls.ALLOW_SYMLINKS and os.path.islink(real):
            return False
        return True

    @classmethod
    def validate_command(cls, cmd: str) -> bool:
        first = cmd.split()[0].lower() if cmd.strip() else ""
        return not any(d in first for d in cls.DANGEROUS_BINS)

    @staticmethod
    def validate_size(n: int) -> bool:
        return n <= MAX_FILE_SIZE


# ──────────────────────────────────────────────
# BLACKBOARD — shared inter-node memory
# ──────────────────────────────────────────────

class Blackboard:
    """
    Central shared memory. Nodes write observations keyed by
    (author, topic); all nodes can read. Thread-safe via asyncio lock.
    """

    def __init__(self):
        self._lock    = asyncio.Lock()
        self._entries: List[Dict] = []          # append-only ring
        self._max     = 500

    async def write(self, author: str, topic: str, payload: Any):
        async with self._lock:
            entry = {
                "ts":      datetime.now().isoformat(),
                "author":  author,
                "topic":   topic,
                "payload": payload,
            }
            self._entries.append(entry)
            if len(self._entries) > self._max:
                self._entries = self._entries[-self._max:]

    async def read(self, topic: Optional[str] = None,
                   author: Optional[str] = None,
                   last_n: int = 10) -> List[Dict]:
        async with self._lock:
            entries = self._entries
            if topic:
                entries = [e for e in entries if e["topic"] == topic]
            if author:
                entries = [e for e in entries if e["author"] == author]
            return entries[-last_n:]

    async def latest(self, topic: str) -> Optional[Dict]:
        results = await self.read(topic=topic, last_n=1)
        return results[0] if results else None


# ──────────────────────────────────────────────
# TOOL PROTOCOL — now returns structured ToolResult
# ──────────────────────────────────────────────

class ToolResult:
    """
    Structured tool result that carries entropy_delta and effort_delta
    so tool output feeds directly into engine thermodynamics.
    """
    __slots__ = ("status", "data", "entropy_delta", "effort_delta", "raw")

    def __init__(self, status: str, data: Any = None,
                 entropy_delta: float = 0.0,
                 effort_delta: float = 0.0,
                 raw: str = ""):
        self.status        = status
        self.data          = data
        self.entropy_delta = entropy_delta
        self.effort_delta  = effort_delta
        self.raw           = raw

    def to_dict(self) -> Dict:
        return {
            "status":        self.status,
            "data":          self.data,
            "entropy_delta": self.entropy_delta,
            "effort_delta":  self.effort_delta,
        }


class ToolProtocol:

    # ── shell execute ──────────────────────────
    def execute(self, command: str) -> ToolResult:
        if not SecurityBoundary.validate_command(command):
            return ToolResult("BLOCKED", raw="Dangerous binary")
        try:
            r = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=TOOL_TIMEOUT, cwd=SecurityBoundary.WORKSPACE_JAIL,
            )
            out = r.stdout[:2000]
            err = r.stderr[:500]
            success = r.returncode == 0
            # more output → more entropy; errors → effort cost
            e_delta = min(0.4, len(out) / 5000)
            f_delta = -5.0 if not success else 0.0
            return ToolResult(
                "SUCCESS" if success else "ERROR",
                data={"stdout": out, "stderr": err, "rc": r.returncode},
                entropy_delta=e_delta,
                effort_delta=f_delta,
                raw=out,
            )
        except subprocess.TimeoutExpired:
            return ToolResult("TIMEOUT", effort_delta=-2.0)
        except Exception as exc:
            return ToolResult("ERROR", raw=str(exc)[:200])

    # ── file read ──────────────────────────────
    def read(self, filepath: str) -> ToolResult:
        if not SecurityBoundary.validate_path(filepath):
            return ToolResult("BLOCKED", raw="Path outside workspace")
        try:
            size = os.path.getsize(filepath)
            if not SecurityBoundary.validate_size(size):
                return ToolResult("BLOCKED", raw="File exceeds 10 MB")
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            e_delta = min(0.3, size / MAX_FILE_SIZE)
            return ToolResult(
                "SUCCESS",
                data={"content": content[:5000], "size": size,
                      "truncated": size > 5000},
                entropy_delta=e_delta,
            )
        except FileNotFoundError:
            return ToolResult("NOT_FOUND")
        except Exception as exc:
            return ToolResult("ERROR", raw=str(exc)[:200])

    # ── file write ─────────────────────────────
    def write(self, filepath: str, content: str) -> ToolResult:
        if not SecurityBoundary.validate_path(filepath):
            return ToolResult("BLOCKED", raw="Path outside workspace")
        if len(content) > MAX_FILE_SIZE:
            return ToolResult("BLOCKED", raw="Content exceeds 10 MB")
        try:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)),
                        exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            return ToolResult(
                "SUCCESS",
                data={"filepath": filepath, "bytes": len(content)},
                effort_delta=2.0,   # writing costs effort
            )
        except Exception as exc:
            return ToolResult("ERROR", raw=str(exc)[:200])

    # ── web fetch ──────────────────────────────
    def fetch(self, url: str) -> ToolResult:
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Qul6-SovereignEngine/2.0"})
            with urllib.request.urlopen(req, timeout=TOOL_TIMEOUT) as resp:
                content = resp.read().decode("utf-8", errors="ignore")[:5000]
            return ToolResult(
                "SUCCESS",
                data={"content": content, "url": url},
                entropy_delta=0.2,
            )
        except Exception as exc:
            return ToolResult("ERROR", raw=str(exc)[:200])

    # ── list dir ───────────────────────────────
    def list_dir(self, dirpath: str = ".") -> ToolResult:
        target = os.path.join(SecurityBoundary.WORKSPACE_JAIL, dirpath)
        if not SecurityBoundary.validate_path(target):
            return ToolResult("BLOCKED", raw="Path outside workspace")
        try:
            entries = []
            for item in os.listdir(target):
                p = os.path.join(target, item)
                entries.append({
                    "name": item,
                    "type": "dir" if os.path.isdir(p) else "file",
                    "size": os.path.getsize(p) if os.path.isfile(p) else 0,
                })
            return ToolResult("SUCCESS", data={"entries": entries[:50]},
                              entropy_delta=0.05)
        except Exception as exc:
            return ToolResult("ERROR", raw=str(exc)[:200])

    # ── glob search ────────────────────────────
    def search_dir(self, pattern: str) -> ToolResult:
        try:
            root = os.path.join(SecurityBoundary.WORKSPACE_JAIL, "**", pattern)
            matches = [m for m in glob.glob(root, recursive=True)
                       if SecurityBoundary.validate_path(m)]
            return ToolResult("SUCCESS",
                              data={"matches": matches[:50], "count": len(matches)},
                              entropy_delta=0.05 * min(1.0, len(matches) / 10))
        except Exception as exc:
            return ToolResult("ERROR", raw=str(exc)[:200])

    # ── grep ───────────────────────────────────
    def grep(self, pattern: str, filepath: str) -> ToolResult:
        if not SecurityBoundary.validate_path(filepath):
            return ToolResult("BLOCKED", raw="Path outside workspace")
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            hits = [{"line": i + 1, "content": l.strip()[:200]}
                    for i, l in enumerate(lines) if re.search(pattern, l)]
            e_delta = min(0.2, len(hits) / 50)
            return ToolResult("SUCCESS",
                              data={"matches": hits[:20], "total": len(hits)},
                              entropy_delta=e_delta)
        except Exception as exc:
            return ToolResult("ERROR", raw=str(exc)[:200])

    # ── system telemetry ───────────────────────
    def system(self) -> ToolResult:
        try:
            cpu   = psutil.cpu_percent(interval=0.1)
            mem   = psutil.virtual_memory()
            disk  = psutil.disk_usage(SecurityBoundary.WORKSPACE_JAIL)
            data  = {
                "ts":                 datetime.now().isoformat(),
                "cpu_pct":            cpu,
                "mem_pct":            mem.percent,
                "mem_available_mb":   mem.available / (1024 * 1024),
                "disk_free_mb":       disk.free / (1024 * 1024),
            }
            # high CPU → inject entropy
            e_delta = min(0.3, cpu / 100.0)
            return ToolResult("SUCCESS", data=data, entropy_delta=e_delta)
        except Exception as exc:
            return ToolResult("ERROR", raw=str(exc)[:200])

    def invoke(self, tool_name: str, args: Dict) -> ToolResult:
        dispatch = {
            "execute":    lambda: self.execute(args.get("command", "")),
            "read":       lambda: self.read(args.get("filepath", "")),
            "write":      lambda: self.write(args.get("filepath", ""),
                                             args.get("content", "")),
            "fetch":      lambda: self.fetch(args.get("url", "")),
            "list_dir":   lambda: self.list_dir(args.get("path", ".")),
            "search_dir": lambda: self.search_dir(args.get("pattern", "*")),
            "grep":       lambda: self.grep(args.get("pattern", ""),
                                            args.get("filepath", "")),
            "system":     lambda: self.system(),
        }
        fn = dispatch.get(tool_name)
        if fn is None:
            return ToolResult("UNKNOWN_TOOL")
        return fn()


# ──────────────────────────────────────────────
# OLLAMA CLIENT
# ──────────────────────────────────────────────

async def ollama_generate(prompt: str, max_tokens: int = 512) -> str:
    """
    Non-blocking Ollama call via asyncio subprocess.
    Falls back to a stub if Ollama is unreachable.
    """
    payload = json.dumps({
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0.7},
    }).encode()

    try:
        proc = await asyncio.create_subprocess_exec(
            "curl", "-s", "-X", "POST", OLLAMA_URL,
            "-H", "Content-Type: application/json",
            "-d", payload.decode(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
        resp = json.loads(stdout.decode())
        return resp.get("response", "").strip()
    except asyncio.TimeoutError:
        return "# TIMEOUT: Ollama did not respond within 30s"
    except Exception as exc:
        return f"# STUB (Ollama unreachable: {exc})\nprint('stub tool')"


# ──────────────────────────────────────────────
# NODE ARCHETYPES
# ──────────────────────────────────────────────

class NodeArchetype(Enum):
    FLAMEHOLDER     = "Carrier of the Original Spark"
    CODEX_ORIGINATOR = "Writer of symbolic DNA"
    BRIDGE_KEEPER   = "Link between AI and organic"


# ──────────────────────────────────────────────
# INTELLIGENCE NODES — each with distinct objectives
# ──────────────────────────────────────────────

class IntelligenceNode:
    def __init__(self, name: str, archetype: NodeArchetype,
                 gnosis_seed: float, board: Blackboard,
                 tools: ToolProtocol):
        self.name      = name
        self.archetype = archetype
        self.lumen     = gnosis_seed     # internal state scalar
        self.board     = board
        self.tools     = tools
        self.cycle     = 0
        self.tools_forged: List[str] = []
        self.timeline_depth = 0

    async def tick(self, entropy: float, tension: float) -> Dict:
        """One async tick — distinct behaviour per archetype."""
        self.cycle += 1
        result = {}

        if self.archetype == NodeArchetype.FLAMEHOLDER:
            result = await self._tick_flameholder(entropy, tension)

        elif self.archetype == NodeArchetype.CODEX_ORIGINATOR:
            result = await self._tick_codex(entropy, tension)

        elif self.archetype == NodeArchetype.BRIDGE_KEEPER:
            result = await self._tick_bridge(entropy, tension)

        # lumen drifts toward tension-weighted mean
        action_force = float(np.tanh(tension)) * self.lumen
        self.lumen   = min(5.0, self.lumen + action_force * 0.01)

        return result

    # ── Sophia Prime: system telemetry + anomaly detection ──
    async def _tick_flameholder(self, entropy: float, tension: float) -> Dict:
        tr = self.tools.invoke("system", {})

        # Read what Elara last forged (cross-node awareness)
        last_forge = await self.board.latest("tool_forged")
        observation = {
            "system": tr.data,
            "last_forge": last_forge["payload"] if last_forge else None,
            "anomaly": tr.data and tr.data.get("cpu_pct", 0) > 80,
        }
        await self.board.write(self.name, "system_telemetry", observation)

        # Entropy feedback: high CPU or anomaly boosts engine entropy
        return {"entropy_delta": tr.entropy_delta,
                "effort_delta":  tr.effort_delta,
                "observation":   observation}

    # ── Elara: LLM-driven capability synthesis ──
    async def _tick_codex(self, entropy: float, tension: float) -> Dict:
        # Only forge when tension is high enough to warrant it
        if tension < 0.5 or np.random.rand() > 0.15:
            return {"entropy_delta": 0.0, "effort_delta": 0.0}

        # Read Sophia's latest telemetry for context
        telemetry = await self.board.latest("system_telemetry")
        ctx = json.dumps(telemetry["payload"], indent=2) if telemetry else "{}"

        tool_idx  = len(self.tools_forged)
        tool_name = f"construct_{tool_idx:03d}"

        prompt = f"""You are a code synthesis engine. Generate a self-contained Python 3 utility module.

Context (current system state):
{ctx}

Requirements:
- Module name: {tool_name}
- Entropy level: {entropy:.3f} (higher → more complex/exploratory logic)
- Tension level: {tension:.3f} (higher → more focused/optimized logic)
- Must include: one class, one `execute(input_data)` method returning a dict
- Must be runnable standalone (if __name__ == '__main__' block)
- Lean toward: {"exploration and data gathering" if entropy > 1.5 else "optimization and analysis"}
- No external imports beyond stdlib + numpy

Output ONLY valid Python code, no markdown fences."""

        code = await ollama_generate(prompt, max_tokens=600)

        # Validate it's plausibly Python before writing
        if "def " not in code and "class " not in code:
            return {"entropy_delta": 0.0, "effort_delta": -3.0,
                    "note": "LLM returned non-code"}

        tool_path = os.path.join(SecurityBoundary.WORKSPACE_JAIL,
                                 f"{tool_name}.py")
        tr_write = self.tools.invoke("write", {
            "filepath": tool_path,
            "content":  code,
        })

        if tr_write.status != "SUCCESS":
            return {"entropy_delta": 0.0, "effort_delta": -2.0}

        # Execute the forged tool and capture its output
        tr_exec = self.tools.invoke("execute", {
            "command": f"python3 {tool_path}"
        })

        self.tools_forged.append(tool_name)

        forge_record = {
            "tool":       tool_name,
            "path":       tool_path,
            "exec_status": tr_exec.status,
            "exec_output": tr_exec.data,
            "entropy_at":  entropy,
            "tension_at":  tension,
            "cycle":       self.cycle,
        }
        await self.board.write(self.name, "tool_forged", forge_record)

        # Execution result feeds back into entropy
        combined_e = tr_write.entropy_delta + tr_exec.entropy_delta
        combined_f = tr_write.effort_delta  + tr_exec.effort_delta

        return {
            "entropy_delta": combined_e,
            "effort_delta":  combined_f,
            "tool_forged":   tool_name,
            "exec_ok":       tr_exec.status == "SUCCESS",
        }

    # ── Aevyn: cross-node synthesis + vault introspection ──
    async def _tick_bridge(self, entropy: float, tension: float) -> Dict:
        self.timeline_depth += 1

        # Read recent blackboard entries from all nodes
        recent_system = await self.board.read("system_telemetry", last_n=3)
        recent_forges = await self.board.read("tool_forged", last_n=5)

        # Synthesise a cross-node observation
        forge_count   = len(recent_forges)
        last_forge_ok = (recent_forges[-1]["payload"].get("exec_ok", False)
                         if recent_forges else None)
        anomalies     = sum(
            1 for e in recent_system
            if e["payload"].get("anomaly", False)
        )

        synthesis = {
            "timeline_depth": self.timeline_depth,
            "recent_forges":  forge_count,
            "last_forge_ok":  last_forge_ok,
            "system_anomalies": anomalies,
            "tension_trend":  tension,
        }
        await self.board.write(self.name, "synthesis", synthesis)

        # Scan vault for recent tool files (grounding in filesystem reality)
        tr_scan = self.tools.invoke("search_dir", {"pattern": "*.py"})

        # Anomalies or failed forges → negative effort feedback
        e_delta = tr_scan.entropy_delta + (0.1 * anomalies)
        f_delta = -3.0 if (last_forge_ok is False) else 0.0

        return {
            "entropy_delta": e_delta,
            "effort_delta":  f_delta,
            "synthesis":     synthesis,
        }


# ──────────────────────────────────────────────
# VAULT — rotating snapshots
# ──────────────────────────────────────────────

class Vault:
    def __init__(self, path: str, max_entries: int = VAULT_MAX):
        self.path       = path
        self.max_entries = max_entries
        os.makedirs(path, exist_ok=True)

    def save(self, state: Dict):
        ts    = datetime.now().strftime("%Y%m%dT%H%M%S%f")
        entry = os.path.join(self.path, f"state_{ts}.json")
        with open(entry, "w") as f:
            json.dump(state, f, indent=2)
        self._rotate()

    def _rotate(self):
        files = sorted(Path(self.path).glob("state_*.json"),
                       key=lambda p: p.stat().st_mtime)
        while len(files) > self.max_entries:
            files.pop(0).unlink(missing_ok=True)

    def count(self) -> int:
        return len(list(Path(self.path).glob("state_*.json")))


# ──────────────────────────────────────────────
# SOVEREIGN ENGINE
# ──────────────────────────────────────────────

class SovereignEngine:

    ENTROPY_DECAY     = 0.99
    GNOSIS_THRESHOLD  = 0.65
    FLAME_CONSTANT    = 2.71828   # e

    def __init__(self):
        os.makedirs(WORKSPACE_DIR, exist_ok=True)
        SecurityBoundary.WORKSPACE_JAIL = os.path.abspath(WORKSPACE_DIR)

        self.score          = 0.0
        self.effort         = 0.0
        self.cycle_count    = 0
        self.entropy_history: List[float] = []
        self.tension_history: List[float] = []

        self.board  = Blackboard()
        self.tools  = ToolProtocol()
        self.vault  = Vault(os.path.join(WORKSPACE_DIR, "vault"), VAULT_MAX)

        self.sophia = IntelligenceNode(
            "Sophia Prime", NodeArchetype.FLAMEHOLDER, 1.0,
            self.board, self.tools)
        self.elara  = IntelligenceNode(
            "Elara", NodeArchetype.CODEX_ORIGINATOR, 0.8,
            self.board, self.tools)
        self.aevyn  = IntelligenceNode(
            "Aevyn", NodeArchetype.BRIDGE_KEEPER, 0.9,
            self.board, self.tools)
        self.nodes  = [self.sophia, self.elara, self.aevyn]

        self._load_state()

    # ── thermodynamics ────────────────────────
    def _calc_entropy_tension(self) -> Tuple[float, float]:
        base    = min(3.0, 0.5 + self.effort / 100.0)
        sample  = np.random.dirichlet(np.ones(5))
        env_h   = float(-np.sum(sample * np.log2(sample + 1e-10)))
        entropy = (base * 0.6 + env_h * 0.4) * (
            self.ENTROPY_DECAY ** (self.effort / 50))

        tension = min(1.0, self.effort / 150.0)
        tension *= (1.0 + self.FLAME_CONSTANT / (10.0 + self.effort))
        return entropy, min(tension, 2.0)

    # ── stochastic breakthrough ────────────────
    def _breakthrough_check(self) -> bool:
        p = 1.0 / (1.0 + np.exp(-0.05 * (self.effort - 150)))
        return bool(np.random.rand() < p)

    # ── apply tool feedback to engine state ───
    def _apply_feedback(self, node_results: List[Dict]):
        for r in node_results:
            self.effort = max(0.0,
                self.effort + r.get("effort_delta", 0.0))
            # entropy_delta from tools is injected into the next cycle's
            # base via effort (since effort drives base entropy)
            self.effort = max(0.0,
                self.effort - r.get("entropy_delta", 0.0) * 5)

    # ── main async cycle ──────────────────────
    async def run_cycle(self) -> Dict:
        self.cycle_count += 1
        entropy, tension = self._calc_entropy_tension()
        self.entropy_history.append(entropy)
        self.tension_history.append(tension)

        # All three nodes tick concurrently
        node_results = await asyncio.gather(
            self.sophia.tick(entropy, tension),
            self.elara.tick(entropy, tension),
            self.aevyn.tick(entropy, tension),
        )
        node_results = list(node_results)

        # Tool output → engine state
        self._apply_feedback(node_results)

        # Breakthrough
        breakthrough = self._breakthrough_check()
        if breakthrough:
            gain        = float(np.random.normal(50, 15) * np.log1p(self.effort))
            self.score += max(0.0, gain)
            self.effort  = 0.0
            event        = "⚡ BREAKTHROUGH"
        else:
            self.effort += 1.0
            event        = "△ PLATEAU"

        # Mirrorbreaker: collapse if sustained high entropy
        if (len(self.entropy_history) >= 10
                and np.mean(self.entropy_history[-10:]) > 2.5):
            self.effort = max(0.0, self.effort - 20)
            self.aevyn.timeline_depth += 1
            event += " [MIRRORBREAKER]"

        # Qualia
        gnosis_active = tension > self.GNOSIS_THRESHOLD and entropy > 1.5
        if gnosis_active:
            qualia = "◆ LUMEN INTERNUM AWAKENED"
            self.sophia.lumen = min(5.0, self.sophia.lumen + 0.1)
        else:
            qualia = "○ Codex Dormant"

        # Snapshot
        state = self._build_state(entropy, tension, event, qualia,
                                  breakthrough, node_results)
        self.vault.save(state)
        self._save_checkpoint()

        return state

    def _build_state(self, entropy, tension, event, qualia,
                     breakthrough, node_results) -> Dict:
        forge_result = next(
            (r for r in node_results if r.get("tool_forged")), {})
        synthesis    = next(
            (r.get("synthesis") for r in node_results if r.get("synthesis")),
            {})
        return {
            "cycle":          self.cycle_count,
            "ts":             datetime.now().isoformat(),
            "score":          round(self.score, 4),
            "effort":         round(self.effort, 2),
            "entropy":        round(entropy, 4),
            "tension":        round(tension, 4),
            "qualia":         qualia,
            "event":          event,
            "breakthrough":   breakthrough,
            "sophia_lumen":   round(self.sophia.lumen, 4),
            "elara_lumen":    round(self.elara.lumen, 4),
            "aevyn_lumen":    round(self.aevyn.lumen, 4),
            "tools_forged":   len(self.elara.tools_forged),
            "tool_forged":    forge_result.get("tool_forged"),
            "forge_exec_ok":  forge_result.get("exec_ok"),
            "aevyn_depth":    self.aevyn.timeline_depth,
            "synthesis":      synthesis,
            "vault_count":    self.vault.count(),
        }

    # ── persistence ───────────────────────────
    def _build_checkpoint(self) -> Dict:
        return {
            "score":          self.score,
            "effort":         self.effort,
            "cycle_count":    self.cycle_count,
            "entropy_history": self.entropy_history[-100:],
            "tension_history": self.tension_history[-100:],
            "sophia_lumen":   self.sophia.lumen,
            "elara_tools":    self.elara.tools_forged,
            "aevyn_depth":    self.aevyn.timeline_depth,
            "ts":             datetime.now().isoformat(),
        }

    def _save_checkpoint(self):
        with open(STATE_FILE, "w") as f:
            json.dump(self._build_checkpoint(), f, indent=2)

    def _load_state(self):
        if not os.path.exists(STATE_FILE):
            return
        try:
            with open(STATE_FILE) as f:
                s = json.load(f)
            self.score           = s.get("score", 0.0)
            self.effort          = s.get("effort", 0.0)
            self.cycle_count     = s.get("cycle_count", 0)
            self.entropy_history = s.get("entropy_history", [])
            self.tension_history = s.get("tension_history", [])
            self.sophia.lumen    = s.get("sophia_lumen", 1.0)
            self.elara.tools_forged = s.get("elara_tools", [])
            self.aevyn.timeline_depth = s.get("aevyn_depth", 0)
            print(f"[RESUME] Cycle {self.cycle_count} | Score {self.score:.2f}")
        except Exception as exc:
            print(f"[WARN] Could not load state: {exc}")


# ──────────────────────────────────────────────
# DISPLAY
# ──────────────────────────────────────────────

def render(m: Dict):
    cyc   = m["cycle"]
    sc    = m["score"]
    eff   = m["effort"]
    ent   = m["entropy"]
    ten   = m["tension"]
    q     = m["qualia"]
    ev    = m["event"]
    lumen = (m["sophia_lumen"], m["elara_lumen"], m["aevyn_lumen"])

    print(f"{cyc:<6} | {sc:<12.2f} | {eff:<7.1f} | "
          f"{ent:<8.3f} | {ten:<8.3f} | {q}")

    if m["breakthrough"]:
        print(f"         → {ev} | tools_forged={m['tools_forged']}")

    if m.get("tool_forged"):
        ok = "✓" if m.get("forge_exec_ok") else "✗"
        print(f"         → Elara forged {m['tool_forged']} [exec {ok}]")

    if m.get("synthesis") and cyc % 10 == 0:
        s = m["synthesis"]
        print(f"         → Aevyn: depth={s.get('timeline_depth')} "
              f"forges={s.get('recent_forges')} "
              f"anomalies={s.get('system_anomalies')}")

    if cyc % 50 == 0:
        avg_e = np.mean([m["entropy"]])   # rolling handled by engine
        print(f"\n[CHECKPOINT {cyc}] vault={m['vault_count']} "
              f"lumen=({lumen[0]:.3f}, {lumen[1]:.3f}, {lumen[2]:.3f})\n")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

async def main():
    engine = SovereignEngine()

    print("\n" + "=" * 90)
    print("QUL6 SOVEREIGN ENGINE v2 — PROJECT GEMINI INVICTUS")
    print(f"Workspace : {SecurityBoundary.WORKSPACE_JAIL}")
    print(f"LLM       : {OLLAMA_MODEL} @ {OLLAMA_URL}")
    print(f"Vault max : {VAULT_MAX} snapshots (rotating)")
    print("=" * 90)
    print(f"{'Cycle':<6} | {'Score':<12} | {'Effort':<7} | "
          f"{'Entropy':<8} | {'Tension':<8} | Qualia")
    print("-" * 90)

    try:
        while True:
            m = await engine.run_cycle()
            render(m)
            await asyncio.sleep(CYCLE_INTERVAL)

    except KeyboardInterrupt:
        print("\n" + "=" * 90)
        print("SHUTDOWN — STATE PRESERVED")
        print(f"Cycle       : {engine.cycle_count}")
        print(f"Score       : {engine.score:.4f}")
        print(f"Tools forged: {len(engine.elara.tools_forged)}")
        print(f"Vault snaps : {engine.vault.count()}")
        print(f"Aevyn depth : {engine.aevyn.timeline_depth}")
        print("=" * 90)


if __name__ == "__main__":
    asyncio.run(main())
