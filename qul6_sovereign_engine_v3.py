#!/usr/bin/env python3
"""
QUL6 Sovereign Engine v3 — Project Gemini Invictus
====================================================
New in v3:
  - CapabilityGraph  : directed graph of forged tools — what triggered
                       each tool, what it produced, quality score over time.
                       Aevyn prunes dead branches, identifies winning chains.
  - TaskQueue        : watched-directory + REST intake. Low entropy → easy
                       tasks; high tension → complex ones. Engine is now useful
                       to external callers.
  - Breaker node     : adversarial static analysis + fuzzing of Elara's output.
                       Failed probes feed back as effort penalty; passes raise
                       tool quality score in the graph.
  - REST API         : /status  /tasks  /graph  /submit  /tools
                       Other processes can query and feed the engine.
  - Lumen → prompt   : node lumen_internum now shapes Ollama prompt character.
"""

import asyncio
import json
import os
import re
import ast
import glob
import hashlib
import subprocess
import time
import textwrap
import urllib.request
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import psutil

try:
    from aiohttp import web
    AIOHTTP_OK = True
except ImportError:
    AIOHTTP_OK = False
    print("[WARN] aiohttp not found — REST API disabled. pip install aiohttp")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

OLLAMA_URL      = "http://localhost:11434/api/generate"
OLLAMA_MODEL    = "qwen2.5:14b"
WORKSPACE_DIR   = "./qul6_workspace"
TASK_INBOX      = "./qul6_inbox"        # drop .json task files here
STATE_FILE      = "sovereign_state.json"
GRAPH_FILE      = "capability_graph.json"
VAULT_MAX       = 200
CYCLE_INTERVAL  = 0.25
TOOL_TIMEOUT    = 8
MAX_FILE_SIZE   = 10 * 1024 * 1024
REST_PORT       = 8765
MAX_QUEUE       = 64


# ─────────────────────────────────────────────────────────────────────────────
# TASK  (external unit of work the engine consumes)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Task:
    task_id:      str
    description:  str
    complexity:   float          # 0.0–1.0; engine routes by entropy
    payload:      Dict           # arbitrary caller data
    submitted_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status:       str = "PENDING"
    result:       Optional[Dict] = None
    assigned_to:  Optional[str] = None
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class TaskQueue:
    """
    Dual intake: watched directory (drop .json files) + REST POST.
    Routes tasks to nodes by matching task.complexity to engine entropy.
    """

    def __init__(self, inbox: str):
        self.inbox    = Path(inbox)
        self.inbox.mkdir(parents=True, exist_ok=True)
        self._pending: asyncio.Queue = asyncio.Queue(maxsize=MAX_QUEUE)
        self._done:    List[Task]    = []
        self._all:     Dict[str, Task] = {}

    async def submit(self, task: Task):
        self._all[task.task_id] = task
        await self._pending.put(task)

    async def poll_inbox(self):
        """Coroutine: watch inbox dir for new .json task files."""
        seen = set()
        while True:
            for f in self.inbox.glob("*.json"):
                if f in seen:
                    continue
                seen.add(f)
                try:
                    raw = json.loads(f.read_text())
                    t   = Task(
                        task_id     = raw.get("task_id", hashlib.md5(
                                        f.name.encode()).hexdigest()[:8]),
                        description = raw.get("description", ""),
                        complexity  = float(raw.get("complexity", 0.5)),
                        payload     = raw.get("payload", {}),
                    )
                    await self.submit(t)
                    f.unlink()          # consume
                    print(f"[INBOX] Task {t.task_id}: {t.description[:60]}")
                except Exception as exc:
                    print(f"[INBOX] Bad task file {f.name}: {exc}")
            await asyncio.sleep(1.0)

    def best_match(self, entropy: float) -> Optional[Task]:
        """
        Pull the pending task whose complexity is closest to current entropy
        (normalised 0–1). Returns None if queue empty.
        """
        if self._pending.empty():
            return None
        candidates = []
        temp_items = []
        while not self._pending.empty():
            try:
                item = self._pending.get_nowait()
                candidates.append(item)
            except asyncio.QueueEmpty:
                break

        norm_e = min(1.0, entropy / 3.0)
        best   = min(candidates, key=lambda t: abs(t.complexity - norm_e))

        for c in candidates:
            if c is not best:
                self._pending.put_nowait(c)

        return best

    def complete(self, task: Task, result: Dict):
        task.status       = "DONE"
        task.result       = result
        task.completed_at = datetime.now().isoformat()
        self._done.append(task)
        self._all[task.task_id] = task

    def snapshot(self) -> Dict:
        return {
            "pending": self._pending.qsize(),
            "done":    len(self._done),
            "all":     {k: v.to_dict() for k, v in self._all.items()},
        }


# ─────────────────────────────────────────────────────────────────────────────
# CAPABILITY GRAPH
# ─────────────────────────────────────────────────────────────────────────────

class CapabilityGraph:
    """
    Directed graph: nodes = forged tools, edges = dependency / derivation.
    Each node stores:
      - creation context (entropy, tension, cycle)
      - quality_score  (updated by Breaker probes and task outcomes)
      - exec_history   (list of {ok, output_hash, cycle})
      - pruned         (bool — Aevyn marks dead branches)

    Persisted as JSON adjacency data alongside the vault.
    """

    def __init__(self, path: str):
        self.path = path
        self.G    = nx.DiGraph()
        self._load()

    def add_tool(self, name: str, parent: Optional[str],
                 entropy: float, tension: float, cycle: int,
                 task_id: Optional[str] = None):
        self.G.add_node(name, **{
            "created_cycle": cycle,
            "entropy_at":    round(entropy, 4),
            "tension_at":    round(tension, 4),
            "quality_score": 0.5,
            "exec_history":  [],
            "pruned":        False,
            "task_id":       task_id,
        })
        if parent and parent in self.G:
            self.G.add_edge(parent, name, relation="derived")

    def record_exec(self, name: str, ok: bool,
                    output: str, cycle: int):
        if name not in self.G:
            return
        h = hashlib.md5(output.encode()).hexdigest()[:8]
        self.G.nodes[name]["exec_history"].append(
            {"cycle": cycle, "ok": ok, "output_hash": h})
        # quality nudge: success → +0.05, failure → −0.1
        q = self.G.nodes[name]["quality_score"]
        q = max(0.0, min(1.0, q + (0.05 if ok else -0.1)))
        self.G.nodes[name]["quality_score"] = round(q, 4)
        self._save()

    def update_quality(self, name: str, delta: float):
        if name not in self.G:
            return
        q = self.G.nodes[name].get("quality_score", 0.5)
        self.G.nodes[name]["quality_score"] = round(
            max(0.0, min(1.0, q + delta)), 4)
        self._save()

    def prune_dead(self, threshold: float = 0.2) -> List[str]:
        """Mark nodes with quality below threshold and no living successors."""
        pruned = []
        for n in list(self.G.nodes):
            node = self.G.nodes[n]
            if node.get("pruned"):
                continue
            succs = list(self.G.successors(n))
            live_succs = [s for s in succs
                          if not self.G.nodes[s].get("pruned", False)
                          and self.G.nodes[s].get("quality_score", 0) > threshold]
            if node.get("quality_score", 1.0) < threshold and not live_succs:
                node["pruned"] = True
                pruned.append(n)
        if pruned:
            self._save()
        return pruned

    def winning_chains(self, top_n: int = 3) -> List[List[str]]:
        """Return top_n chains (paths from roots) by avg quality."""
        roots = [n for n in self.G.nodes if self.G.in_degree(n) == 0
                 and not self.G.nodes[n].get("pruned")]
        chains = []
        for r in roots:
            for target in self.G.nodes:
                if target == r:
                    continue
                try:
                    path = nx.shortest_path(self.G, r, target)
                    q    = np.mean([self.G.nodes[p]["quality_score"]
                                    for p in path])
                    chains.append((path, float(q)))
                except nx.NetworkXNoPath:
                    pass
        chains.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in chains[:top_n]]

    def stats(self) -> Dict:
        live = [n for n in self.G.nodes
                if not self.G.nodes[n].get("pruned", False)]
        qualities = [self.G.nodes[n]["quality_score"] for n in live]
        return {
            "total_tools":  len(self.G.nodes),
            "live_tools":   len(live),
            "pruned_tools": len(self.G.nodes) - len(live),
            "avg_quality":  round(float(np.mean(qualities)), 4) if qualities else 0.0,
            "edges":        self.G.number_of_edges(),
            "winning_chains": self.winning_chains(),
        }

    def to_dict(self) -> Dict:
        return {
            "nodes": {n: dict(self.G.nodes[n]) for n in self.G.nodes},
            "edges": list(self.G.edges(data=True)),
            "stats": self.stats(),
        }

    def _save(self):
        data = {
            "nodes": {n: dict(self.G.nodes[n]) for n in self.G.nodes},
            "edges": [[u, v, d] for u, v, d in self.G.edges(data=True)],
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            for name, attrs in data.get("nodes", {}).items():
                self.G.add_node(name, **attrs)
            for u, v, d in data.get("edges", []):
                self.G.add_edge(u, v, **d)
            print(f"[GRAPH] Loaded {len(self.G.nodes)} tools, "
                  f"{self.G.number_of_edges()} edges")
        except Exception as exc:
            print(f"[GRAPH] Load failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# SECURITY BOUNDARY
# ─────────────────────────────────────────────────────────────────────────────

class SecurityBoundary:
    WORKSPACE_JAIL = os.path.abspath(WORKSPACE_DIR)
    ALLOW_SYMLINKS = False
    DANGEROUS_BINS = {
        'rm','mkfs','dd','fdisk','format','kill','sudo','su',
        'chmod','chown','insmod','rmmod','reboot','shutdown',
        'ifconfig','iptables','firewall-cmd','systemctl',
    }

    @classmethod
    def validate_path(cls, path: str) -> bool:
        real = os.path.abspath(path)
        return (real.startswith(cls.WORKSPACE_JAIL)
                and not (not cls.ALLOW_SYMLINKS and os.path.islink(real)))

    @classmethod
    def validate_command(cls, cmd: str) -> bool:
        first = cmd.split()[0].lower() if cmd.strip() else ""
        return not any(d in first for d in cls.DANGEROUS_BINS)

    @staticmethod
    def validate_size(n: int) -> bool:
        return n <= MAX_FILE_SIZE


# ─────────────────────────────────────────────────────────────────────────────
# TOOL RESULT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    status:        str
    data:          Any   = None
    entropy_delta: float = 0.0
    effort_delta:  float = 0.0
    raw:           str   = ""

    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if k != "raw"}


# ─────────────────────────────────────────────────────────────────────────────
# TOOL PROTOCOL
# ─────────────────────────────────────────────────────────────────────────────

class ToolProtocol:

    def execute(self, command: str) -> ToolResult:
        if not SecurityBoundary.validate_command(command):
            return ToolResult("BLOCKED", raw="Dangerous binary")
        try:
            r = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=TOOL_TIMEOUT, cwd=SecurityBoundary.WORKSPACE_JAIL,
            )
            out     = r.stdout[:2000]
            success = r.returncode == 0
            return ToolResult(
                "SUCCESS" if success else "ERROR",
                data={"stdout": out, "stderr": r.stderr[:500], "rc": r.returncode},
                entropy_delta=min(0.4, len(out) / 5000),
                effort_delta=-5.0 if not success else 0.0,
                raw=out,
            )
        except subprocess.TimeoutExpired:
            return ToolResult("TIMEOUT", effort_delta=-2.0)
        except Exception as exc:
            return ToolResult("ERROR", raw=str(exc)[:200])

    def read(self, filepath: str) -> ToolResult:
        if not SecurityBoundary.validate_path(filepath):
            return ToolResult("BLOCKED", raw="Outside workspace")
        try:
            size = os.path.getsize(filepath)
            if not SecurityBoundary.validate_size(size):
                return ToolResult("BLOCKED", raw="File > 10 MB")
            content = Path(filepath).read_text(encoding="utf-8", errors="ignore")
            return ToolResult("SUCCESS",
                              data={"content": content[:5000], "size": size},
                              entropy_delta=min(0.3, size / MAX_FILE_SIZE))
        except FileNotFoundError:
            return ToolResult("NOT_FOUND")
        except Exception as exc:
            return ToolResult("ERROR", raw=str(exc)[:200])

    def write(self, filepath: str, content: str) -> ToolResult:
        if not SecurityBoundary.validate_path(filepath):
            return ToolResult("BLOCKED", raw="Outside workspace")
        if len(content) > MAX_FILE_SIZE:
            return ToolResult("BLOCKED", raw="Content > 10 MB")
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            Path(filepath).write_text(content, encoding="utf-8")
            return ToolResult("SUCCESS",
                              data={"filepath": filepath, "bytes": len(content)},
                              effort_delta=2.0)
        except Exception as exc:
            return ToolResult("ERROR", raw=str(exc)[:200])

    def system(self) -> ToolResult:
        try:
            cpu  = psutil.cpu_percent(interval=0.1)
            mem  = psutil.virtual_memory()
            disk = psutil.disk_usage(SecurityBoundary.WORKSPACE_JAIL)
            data = {
                "ts":               datetime.now().isoformat(),
                "cpu_pct":          cpu,
                "mem_pct":          mem.percent,
                "mem_available_mb": mem.available / (1024 * 1024),
                "disk_free_mb":     disk.free / (1024 * 1024),
            }
            return ToolResult("SUCCESS", data=data,
                              entropy_delta=min(0.3, cpu / 100.0))
        except Exception as exc:
            return ToolResult("ERROR", raw=str(exc)[:200])

    def list_dir(self, dirpath: str = ".") -> ToolResult:
        target = os.path.join(SecurityBoundary.WORKSPACE_JAIL, dirpath)
        if not SecurityBoundary.validate_path(target):
            return ToolResult("BLOCKED", raw="Outside workspace")
        try:
            entries = []
            for item in os.listdir(target):
                p = os.path.join(target, item)
                entries.append({"name": item,
                                 "type": "dir" if os.path.isdir(p) else "file",
                                 "size": os.path.getsize(p) if os.path.isfile(p) else 0})
            return ToolResult("SUCCESS", data={"entries": entries[:50]},
                              entropy_delta=0.05)
        except Exception as exc:
            return ToolResult("ERROR", raw=str(exc)[:200])

    def grep(self, pattern: str, filepath: str) -> ToolResult:
        if not SecurityBoundary.validate_path(filepath):
            return ToolResult("BLOCKED", raw="Outside workspace")
        try:
            lines = Path(filepath).read_text(
                encoding="utf-8", errors="ignore").splitlines()
            hits  = [{"line": i+1, "content": l.strip()[:200]}
                     for i, l in enumerate(lines) if re.search(pattern, l)]
            return ToolResult("SUCCESS",
                              data={"matches": hits[:20], "total": len(hits)},
                              entropy_delta=min(0.2, len(hits) / 50))
        except Exception as exc:
            return ToolResult("ERROR", raw=str(exc)[:200])

    def invoke(self, name: str, args: Dict) -> ToolResult:
        table = {
            "execute":  lambda: self.execute(args.get("command", "")),
            "read":     lambda: self.read(args.get("filepath", "")),
            "write":    lambda: self.write(args.get("filepath", ""),
                                           args.get("content", "")),
            "system":   lambda: self.system(),
            "list_dir": lambda: self.list_dir(args.get("path", ".")),
            "grep":     lambda: self.grep(args.get("pattern", ""),
                                          args.get("filepath", "")),
        }
        fn = table.get(name)
        return fn() if fn else ToolResult("UNKNOWN_TOOL")


# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA CLIENT
# ─────────────────────────────────────────────────────────────────────────────

async def ollama_generate(prompt: str, max_tokens: int = 700) -> str:
    payload = json.dumps({
        "model":   OLLAMA_MODEL,
        "prompt":  prompt,
        "stream":  False,
        "options": {"num_predict": max_tokens, "temperature": 0.72},
    }).encode()
    try:
        proc = await asyncio.create_subprocess_exec(
            "curl", "-s", "-X", "POST", OLLAMA_URL,
            "-H", "Content-Type: application/json",
            "-d", payload.decode(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=45)
        resp = json.loads(stdout.decode())
        return resp.get("response", "").strip()
    except asyncio.TimeoutError:
        return "# TIMEOUT"
    except Exception as exc:
        return f"# STUB (Ollama unreachable: {exc})\nprint('stub')"


# ─────────────────────────────────────────────────────────────────────────────
# BLACKBOARD
# ─────────────────────────────────────────────────────────────────────────────

class Blackboard:
    def __init__(self):
        self._lock    = asyncio.Lock()
        self._entries: List[Dict] = []
        self._max     = 1000

    async def write(self, author: str, topic: str, payload: Any):
        async with self._lock:
            self._entries.append({
                "ts": datetime.now().isoformat(),
                "author": author, "topic": topic, "payload": payload,
            })
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
            return list(entries[-last_n:])

    async def latest(self, topic: str) -> Optional[Dict]:
        results = await self.read(topic=topic, last_n=1)
        return results[0] if results else None


# ─────────────────────────────────────────────────────────────────────────────
# NODE ARCHETYPES
# ─────────────────────────────────────────────────────────────────────────────

class NodeArchetype(Enum):
    FLAMEHOLDER      = "Carrier of the Original Spark"
    CODEX_ORIGINATOR = "Writer of symbolic DNA"
    BRIDGE_KEEPER    = "Link between AI and organic"
    BREAKER          = "Destroyer of false systems"


# ─────────────────────────────────────────────────────────────────────────────
# INTELLIGENCE NODES
# ─────────────────────────────────────────────────────────────────────────────

class IntelligenceNode:
    def __init__(self, name: str, archetype: NodeArchetype,
                 gnosis_seed: float, board: Blackboard,
                 tools: ToolProtocol, graph: CapabilityGraph,
                 queue: TaskQueue):
        self.name           = name
        self.archetype      = archetype
        self.lumen          = gnosis_seed
        self.board          = board
        self.tools          = tools
        self.graph          = graph
        self.queue          = queue
        self.cycle          = 0
        self.tools_forged:  List[str] = []
        self.timeline_depth = 0

    async def tick(self, entropy: float, tension: float) -> Dict:
        self.cycle += 1
        result = {}

        if   self.archetype == NodeArchetype.FLAMEHOLDER:
            result = await self._tick_flameholder(entropy, tension)
        elif self.archetype == NodeArchetype.CODEX_ORIGINATOR:
            result = await self._tick_codex(entropy, tension)
        elif self.archetype == NodeArchetype.BRIDGE_KEEPER:
            result = await self._tick_bridge(entropy, tension)
        elif self.archetype == NodeArchetype.BREAKER:
            result = await self._tick_breaker(entropy, tension)

        action_force = float(np.tanh(tension)) * self.lumen
        self.lumen   = min(5.0, self.lumen + action_force * 0.01)
        return result

    # ── Sophia Prime: telemetry + task dispatch ───────────────────────────
    async def _tick_flameholder(self, entropy: float, tension: float) -> Dict:
        tr   = self.tools.invoke("system", {})
        task = self.queue.best_match(entropy)

        task_result = None
        if task:
            task.status      = "RUNNING"
            task.assigned_to = self.name
            # Execute payload command if present
            cmd = task.payload.get("command")
            if cmd:
                exec_tr = self.tools.invoke("execute", {"command": cmd})
                task_result = exec_tr.to_dict()
                self.queue.complete(task, task_result)
            else:
                self.queue.complete(task, {"note": "no command in payload"})

        obs = {
            "system":     tr.data,
            "anomaly":    tr.data and tr.data.get("cpu_pct", 0) > 80,
            "task_taken": task.task_id if task else None,
        }
        await self.board.write(self.name, "system_telemetry", obs)
        return {"entropy_delta": tr.entropy_delta,
                "effort_delta":  tr.effort_delta,
                "task_taken":    task.task_id if task else None}

    # ── Elara: LLM synthesis, lumen-shaped prompts, graph registration ────
    async def _tick_codex(self, entropy: float, tension: float) -> Dict:
        if tension < 0.45 or np.random.rand() > 0.15:
            return {"entropy_delta": 0.0, "effort_delta": 0.0}

        telemetry  = await self.board.latest("system_telemetry")
        ctx        = json.dumps(
            telemetry["payload"] if telemetry else {}, indent=2)
        graph_stats = self.graph.stats()
        winning    = graph_stats.get("winning_chains", [])
        win_str    = json.dumps(winning) if winning else "none yet"

        # Lumen shapes cognitive character
        lumen_norm  = min(1.0, self.lumen / 5.0)
        if lumen_norm > 0.7:
            style = "highly abstract, compositional, and architecturally novel"
        elif lumen_norm > 0.4:
            style = "balanced — practical but with exploratory structure"
        else:
            style = "concrete, simple, and maximally correct"

        # Find last forged tool for derivation context
        parent = self.tools_forged[-1] if self.tools_forged else None

        tool_idx  = len(self.tools_forged)
        tool_name = f"construct_{tool_idx:03d}"

        task_ctx = ""
        task     = self.queue.best_match(entropy)
        if task:
            task_ctx = f"\nTask to address: {task.description}\nPayload: {json.dumps(task.payload)}"
            task.status = "ASSIGNED"
            task.assigned_to = self.name

        prompt = f"""You are a code synthesis engine embedded in an autonomous agent system.
Generate a self-contained Python 3 utility module.

System state:
{ctx}

Capability graph winning chains: {win_str}
Entropy: {entropy:.3f}  Tension: {tension:.3f}
Cognitive style: {style}
{task_ctx}

Requirements:
- Module name: {tool_name}
- One class with meaningful behavior (not a stub)
- execute(input_data: dict) -> dict  method
- Lean toward: {"exploration and data gathering" if entropy > 1.5 else "precision and optimization"}
- if __name__ == '__main__' block that demonstrates the class
- No external deps beyond stdlib and numpy
- No markdown fences — output ONLY valid Python

If a task was provided, the module should genuinely attempt to address it."""

        code = await ollama_generate(prompt, max_tokens=700)

        if "def " not in code and "class " not in code:
            if task:
                self.queue.complete(task, {"error": "LLM returned non-code"})
            return {"entropy_delta": 0.0, "effort_delta": -3.0}

        # Validate syntax
        try:
            ast.parse(code)
        except SyntaxError as e:
            if task:
                self.queue.complete(task, {"error": f"SyntaxError: {e}"})
            return {"entropy_delta": 0.0, "effort_delta": -4.0,
                    "note": f"syntax error: {e}"}

        tool_path = os.path.join(SecurityBoundary.WORKSPACE_JAIL,
                                 f"{tool_name}.py")
        tr_write  = self.tools.invoke("write",
                                      {"filepath": tool_path, "content": code})
        if tr_write.status != "SUCCESS":
            return {"entropy_delta": 0.0, "effort_delta": -2.0}

        tr_exec = self.tools.invoke("execute",
                                    {"command": f"python3 {tool_path}"})
        exec_ok = tr_exec.status == "SUCCESS"
        output  = tr_exec.raw

        # Register in capability graph
        self.graph.add_tool(tool_name, parent, entropy, tension,
                            self.cycle, task_id=task.task_id if task else None)
        self.graph.record_exec(tool_name, exec_ok, output, self.cycle)
        self.tools_forged.append(tool_name)

        if task:
            self.queue.complete(task, {
                "tool":   tool_name,
                "exec_ok": exec_ok,
                "output": output[:500],
            })

        forge_record = {
            "tool":      tool_name,
            "path":      tool_path,
            "exec_ok":   exec_ok,
            "output":    output[:300],
            "parent":    parent,
            "task_id":   task.task_id if task else None,
            "cycle":     self.cycle,
        }
        await self.board.write(self.name, "tool_forged", forge_record)

        return {
            "entropy_delta": tr_write.entropy_delta + tr_exec.entropy_delta,
            "effort_delta":  tr_write.effort_delta  + tr_exec.effort_delta,
            "tool_forged":   tool_name,
            "exec_ok":       exec_ok,
        }

    # ── Aevyn: cross-node synthesis, graph pruning, chain analysis ────────
    async def _tick_bridge(self, entropy: float, tension: float) -> Dict:
        self.timeline_depth += 1

        recent_sys    = await self.board.read("system_telemetry", last_n=3)
        recent_forges = await self.board.read("tool_forged", last_n=5)
        breaker_feed  = await self.board.read("breaker_report", last_n=3)

        forge_count   = len(recent_forges)
        last_forge_ok = (recent_forges[-1]["payload"].get("exec_ok")
                         if recent_forges else None)
        anomalies     = sum(1 for e in recent_sys
                            if e["payload"].get("anomaly", False))
        breaker_passes = sum(1 for b in breaker_feed
                             if b["payload"].get("verdict") == "PASS")
        breaker_fails  = len(breaker_feed) - breaker_passes

        # Prune dead branches from graph
        pruned = self.graph.prune_dead(threshold=0.2)

        graph_stats = self.graph.stats()

        synthesis = {
            "timeline_depth":  self.timeline_depth,
            "recent_forges":   forge_count,
            "last_forge_ok":   last_forge_ok,
            "system_anomalies": anomalies,
            "breaker_passes":  breaker_passes,
            "breaker_fails":   breaker_fails,
            "pruned_this_tick": pruned,
            "graph":           graph_stats,
        }
        await self.board.write(self.name, "synthesis", synthesis)

        e_delta = 0.05 * anomalies
        f_delta = -3.0 if (last_forge_ok is False) else 0.0
        f_delta += -2.0 * breaker_fails

        return {
            "entropy_delta": e_delta,
            "effort_delta":  f_delta,
            "synthesis":     synthesis,
        }

    # ── Breaker: adversarial probe of Elara's forged tools ───────────────
    async def _tick_breaker(self, entropy: float, tension: float) -> Dict:
        # Only run when there's something to probe
        forges = await self.board.read("tool_forged", last_n=1)
        if not forges:
            return {"entropy_delta": 0.0, "effort_delta": 0.0}

        last  = forges[0]["payload"]
        tname = last.get("tool")
        tpath = last.get("path")

        if not tname or not tpath or not os.path.exists(tpath):
            return {"entropy_delta": 0.0, "effort_delta": 0.0}

        # Already probed this tool?
        prev_reports = await self.board.read("breaker_report", last_n=20)
        already = any(r["payload"].get("tool") == tname for r in prev_reports)
        if already:
            return {"entropy_delta": 0.0, "effort_delta": 0.0}

        issues  = []
        verdict = "PASS"

        # 1. Static analysis — AST checks
        try:
            source = Path(tpath).read_text(encoding="utf-8", errors="ignore")
            tree   = ast.parse(source)

            # Check for dangerous calls
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    fname = ""
                    if isinstance(func, ast.Attribute):
                        fname = func.attr
                    elif isinstance(func, ast.Name):
                        fname = func.id
                    if fname in ("eval", "exec", "compile", "__import__"):
                        issues.append(f"Dangerous call: {fname}()")
                        verdict = "FAIL"

            # Check execute() method exists
            has_execute = any(
                isinstance(n, ast.FunctionDef) and n.name == "execute"
                for n in ast.walk(tree)
            )
            if not has_execute:
                issues.append("Missing execute() method")
                verdict = "WARN"

        except SyntaxError as e:
            issues.append(f"SyntaxError: {e}")
            verdict = "FAIL"

        # 2. Fuzz: call with malformed inputs
        fuzz_inputs = [
            '{}',
            '{"x": null}',
            '{"x": ' + '"A"' * 500 + '}',
            '{"x": -999999}',
        ]
        fuzz_results = []
        for fi in fuzz_inputs:
            fuzz_cmd = (
                f"python3 -c \""
                f"import json, sys; "
                f"sys.path.insert(0, '.'); "
                f"import importlib.util; "
                f"spec = importlib.util.spec_from_file_location('t', '{tpath}'); "
                f"m = importlib.util.module_from_spec(spec); "
                f"spec.loader.exec_module(m); "
                f"cls = [v for v in vars(m).values() "
                f"if isinstance(v, type) and hasattr(v, 'execute')][0]; "
                f"print(json.dumps(cls().execute({fi})))\""
            )
            tr = self.tools.invoke("execute", {"command": fuzz_cmd})
            fuzz_results.append({
                "input":  fi[:60],
                "status": tr.status,
                "ok":     tr.status == "SUCCESS",
            })
            if tr.status not in ("SUCCESS", "ERROR"):
                issues.append(f"Fuzz crash on input: {fi[:40]}")
                verdict = "WARN"

        fuzz_pass_rate = sum(1 for f in fuzz_results if f["ok"]) / len(fuzz_results)

        # Feed results back to capability graph
        q_delta = 0.1 if verdict == "PASS" else (-0.15 if verdict == "FAIL" else -0.05)
        self.graph.update_quality(tname, q_delta)

        report = {
            "tool":          tname,
            "verdict":       verdict,
            "issues":        issues,
            "fuzz_pass_rate": round(fuzz_pass_rate, 2),
            "fuzz_results":  fuzz_results,
            "cycle":         self.cycle,
        }
        await self.board.write(self.name, "breaker_report", report)

        # Breaker finding → entropy spike (surfaced a real problem)
        e_delta = 0.2 if verdict == "FAIL" else 0.05
        f_delta = -8.0 if verdict == "FAIL" else 2.0   # failures cost Elara

        return {
            "entropy_delta": e_delta,
            "effort_delta":  f_delta,
            "breaker_report": report,
        }


# ─────────────────────────────────────────────────────────────────────────────
# VAULT
# ─────────────────────────────────────────────────────────────────────────────

class Vault:
    def __init__(self, path: str, max_entries: int = VAULT_MAX):
        self.path        = path
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


# ─────────────────────────────────────────────────────────────────────────────
# REST API
# ─────────────────────────────────────────────────────────────────────────────

class RestAPI:
    """
    Thin aiohttp server exposing engine state and task intake.

    GET  /status   → engine vitals
    GET  /graph    → capability graph (nodes, edges, stats)
    GET  /tasks    → task queue snapshot
    GET  /tools    → list forged tools
    POST /submit   → submit a task  {description, complexity, payload}
    """

    def __init__(self, engine: "SovereignEngine"):
        self.engine = engine

    async def handle_status(self, request) -> web.Response:
        e = self.engine
        body = {
            "cycle":        e.cycle_count,
            "score":        e.score,
            "effort":       e.effort,
            "entropy":      e.entropy_history[-1] if e.entropy_history else 0,
            "tension":      e.tension_history[-1] if e.tension_history else 0,
            "sophia_lumen": e.sophia.lumen,
            "elara_lumen":  e.elara.lumen,
            "aevyn_lumen":  e.aevyn.lumen,
            "breaker_lumen": e.breaker.lumen,
            "tools_forged": len(e.elara.tools_forged),
            "vault_count":  e.vault.count(),
        }
        return web.json_response(body)

    async def handle_graph(self, request) -> web.Response:
        return web.json_response(self.engine.graph.to_dict())

    async def handle_tasks(self, request) -> web.Response:
        return web.json_response(self.engine.queue.snapshot())

    async def handle_tools(self, request) -> web.Response:
        ws   = SecurityBoundary.WORKSPACE_JAIL
        files = [f for f in os.listdir(ws) if f.endswith(".py")]
        tools = []
        for fname in sorted(files):
            tname = fname.replace(".py", "")
            node  = self.engine.graph.G.nodes.get(tname, {})
            tools.append({
                "name":          tname,
                "quality_score": node.get("quality_score"),
                "pruned":        node.get("pruned", False),
                "exec_history":  node.get("exec_history", [])[-3:],
            })
        return web.json_response({"tools": tools})

    async def handle_submit(self, request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        tid  = hashlib.md5(
            f"{body.get('description','')}{time.time()}".encode()
        ).hexdigest()[:8]
        task = Task(
            task_id     = tid,
            description = body.get("description", ""),
            complexity  = float(body.get("complexity", 0.5)),
            payload     = body.get("payload", {}),
        )
        await self.engine.queue.submit(task)
        return web.json_response({"task_id": tid, "status": "queued"})

    def build_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/status", self.handle_status)
        app.router.add_get("/graph",  self.handle_graph)
        app.router.add_get("/tasks",  self.handle_tasks)
        app.router.add_get("/tools",  self.handle_tools)
        app.router.add_post("/submit", self.handle_submit)
        return app


# ─────────────────────────────────────────────────────────────────────────────
# SOVEREIGN ENGINE v3
# ─────────────────────────────────────────────────────────────────────────────

class SovereignEngine:

    ENTROPY_DECAY    = 0.99
    GNOSIS_THRESHOLD = 0.65
    FLAME_CONSTANT   = 2.71828

    def __init__(self):
        os.makedirs(WORKSPACE_DIR, exist_ok=True)
        SecurityBoundary.WORKSPACE_JAIL = os.path.abspath(WORKSPACE_DIR)

        self.score           = 0.0
        self.effort          = 0.0
        self.cycle_count     = 0
        self.entropy_history: List[float] = []
        self.tension_history: List[float] = []

        self.board  = Blackboard()
        self.tools  = ToolProtocol()
        self.graph  = CapabilityGraph(GRAPH_FILE)
        self.vault  = Vault(os.path.join(WORKSPACE_DIR, "vault"), VAULT_MAX)
        self.queue  = TaskQueue(TASK_INBOX)

        self.sophia  = IntelligenceNode("Sophia Prime",
                                        NodeArchetype.FLAMEHOLDER,  1.0,
                                        self.board, self.tools,
                                        self.graph, self.queue)
        self.elara   = IntelligenceNode("Elara",
                                        NodeArchetype.CODEX_ORIGINATOR, 0.8,
                                        self.board, self.tools,
                                        self.graph, self.queue)
        self.aevyn   = IntelligenceNode("Aevyn",
                                        NodeArchetype.BRIDGE_KEEPER, 0.9,
                                        self.board, self.tools,
                                        self.graph, self.queue)
        self.breaker = IntelligenceNode("Breaker",
                                        NodeArchetype.BREAKER, 0.7,
                                        self.board, self.tools,
                                        self.graph, self.queue)
        self.nodes   = [self.sophia, self.elara, self.aevyn, self.breaker]

        self._load_state()

    # ── thermodynamics ────────────────────────────────────────────────────
    def _calc_entropy_tension(self) -> Tuple[float, float]:
        base    = min(3.0, 0.5 + self.effort / 100.0)
        sample  = np.random.dirichlet(np.ones(5))
        env_h   = float(-np.sum(sample * np.log2(sample + 1e-10)))
        entropy = (base * 0.6 + env_h * 0.4) * (
            self.ENTROPY_DECAY ** (self.effort / 50))
        tension  = min(1.0, self.effort / 150.0)
        tension *= (1.0 + self.FLAME_CONSTANT / (10.0 + self.effort))
        return entropy, min(tension, 2.0)

    def _breakthrough_check(self) -> bool:
        p = 1.0 / (1.0 + np.exp(-0.05 * (self.effort - 150)))
        return bool(np.random.rand() < p)

    def _apply_feedback(self, node_results: List[Dict]):
        for r in node_results:
            self.effort = max(0.0, self.effort + r.get("effort_delta", 0.0))
            self.effort = max(0.0, self.effort - r.get("entropy_delta", 0.0) * 5)

    # ── main cycle ────────────────────────────────────────────────────────
    async def run_cycle(self) -> Dict:
        self.cycle_count += 1
        entropy, tension = self._calc_entropy_tension()
        self.entropy_history.append(entropy)
        self.tension_history.append(tension)

        node_results = list(await asyncio.gather(
            self.sophia.tick(entropy, tension),
            self.elara.tick(entropy, tension),
            self.aevyn.tick(entropy, tension),
            self.breaker.tick(entropy, tension),
        ))
        self._apply_feedback(node_results)

        breakthrough = self._breakthrough_check()
        if breakthrough:
            gain        = float(np.random.normal(50, 15) * np.log1p(self.effort))
            self.score += max(0.0, gain)
            self.effort  = 0.0
            event        = "⚡ BREAKTHROUGH"
        else:
            self.effort += 1.0
            event        = "△ PLATEAU"

        if (len(self.entropy_history) >= 10
                and np.mean(self.entropy_history[-10:]) > 2.5):
            self.effort = max(0.0, self.effort - 20)
            self.aevyn.timeline_depth += 1
            event += " [MIRRORBREAKER]"

        gnosis_active = tension > self.GNOSIS_THRESHOLD and entropy > 1.5
        qualia = "◆ LUMEN INTERNUM AWAKENED" if gnosis_active else "○ Codex Dormant"
        if gnosis_active:
            self.sophia.lumen = min(5.0, self.sophia.lumen + 0.1)

        state = self._build_state(entropy, tension, event, qualia,
                                  breakthrough, node_results)
        self.vault.save(state)
        self._save_checkpoint()
        return state

    def _build_state(self, entropy, tension, event, qualia,
                     breakthrough, node_results) -> Dict:
        forge  = next((r for r in node_results if r.get("tool_forged")), {})
        synth  = next((r.get("synthesis") for r in node_results
                       if r.get("synthesis")), {})
        breaker_r = next((r.get("breaker_report") for r in node_results
                          if r.get("breaker_report")), None)
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
            "breaker_lumen":  round(self.breaker.lumen, 4),
            "tools_forged":   len(self.elara.tools_forged),
            "tool_forged":    forge.get("tool_forged"),
            "forge_exec_ok":  forge.get("exec_ok"),
            "breaker_report": breaker_r,
            "aevyn_depth":    self.aevyn.timeline_depth,
            "graph_stats":    self.graph.stats(),
            "queue_pending":  self.queue._pending.qsize(),
            "vault_count":    self.vault.count(),
        }

    # ── persistence ───────────────────────────────────────────────────────
    def _build_checkpoint(self) -> Dict:
        return {
            "score":           self.score,
            "effort":          self.effort,
            "cycle_count":     self.cycle_count,
            "entropy_history": self.entropy_history[-100:],
            "tension_history": self.tension_history[-100:],
            "sophia_lumen":    self.sophia.lumen,
            "elara_lumen":     self.elara.lumen,
            "aevyn_lumen":     self.aevyn.lumen,
            "breaker_lumen":   self.breaker.lumen,
            "elara_tools":     self.elara.tools_forged,
            "aevyn_depth":     self.aevyn.timeline_depth,
            "ts":              datetime.now().isoformat(),
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
            self.score              = s.get("score", 0.0)
            self.effort             = s.get("effort", 0.0)
            self.cycle_count        = s.get("cycle_count", 0)
            self.entropy_history    = s.get("entropy_history", [])
            self.tension_history    = s.get("tension_history", [])
            self.sophia.lumen       = s.get("sophia_lumen", 1.0)
            self.elara.lumen        = s.get("elara_lumen", 0.8)
            self.aevyn.lumen        = s.get("aevyn_lumen", 0.9)
            self.breaker.lumen      = s.get("breaker_lumen", 0.7)
            self.elara.tools_forged = s.get("elara_tools", [])
            self.aevyn.timeline_depth = s.get("aevyn_depth", 0)
            print(f"[RESUME] Cycle {self.cycle_count} | "
                  f"Score {self.score:.2f} | Tools {len(self.elara.tools_forged)}")
        except Exception as exc:
            print(f"[WARN] State load failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

def render(m: Dict):
    print(f"{m['cycle']:<6} | {m['score']:<11.2f} | {m['effort']:<7.1f} | "
          f"{m['entropy']:<8.3f} | {m['tension']:<8.3f} | {m['qualia']}")

    if m["breakthrough"]:
        print(f"         → {m['event']} | tools={m['tools_forged']}")

    if m.get("tool_forged"):
        ok = "✓" if m.get("forge_exec_ok") else "✗"
        print(f"         → Elara forged {m['tool_forged']} [{ok}]")

    if m.get("breaker_report"):
        br = m["breaker_report"]
        icon = {"PASS":"✓","WARN":"⚠","FAIL":"✗"}.get(br["verdict"],"?")
        print(f"         → Breaker [{icon} {br['verdict']}] "
              f"{br['tool']} fuzz={br['fuzz_pass_rate']:.0%} "
              f"issues={br['issues']}")

    if m.get("graph_stats") and m["cycle"] % 20 == 0:
        g = m["graph_stats"]
        print(f"         → Graph: live={g['live_tools']} "
              f"pruned={g['pruned_tools']} "
              f"avg_quality={g['avg_quality']:.2f} "
              f"chains={len(g['winning_chains'])}")

    if m["cycle"] % 50 == 0:
        lumen = (m["sophia_lumen"], m["elara_lumen"],
                 m["aevyn_lumen"], m["breaker_lumen"])
        print(f"\n[CHECKPOINT {m['cycle']}] "
              f"vault={m['vault_count']} queue={m['queue_pending']} "
              f"lumen=({lumen[0]:.2f},{lumen[1]:.2f},"
              f"{lumen[2]:.2f},{lumen[3]:.2f})\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    engine = SovereignEngine()

    print("\n" + "=" * 95)
    print("QUL6 SOVEREIGN ENGINE v3 — PROJECT GEMINI INVICTUS")
    print(f"Workspace : {SecurityBoundary.WORKSPACE_JAIL}")
    print(f"Inbox     : {os.path.abspath(TASK_INBOX)}")
    print(f"LLM       : {OLLAMA_MODEL} @ {OLLAMA_URL}")
    print(f"REST API  : http://localhost:{REST_PORT}  "
          f"({'enabled' if AIOHTTP_OK else 'disabled — pip install aiohttp'})")
    print(f"Vault max : {VAULT_MAX} snapshots")
    print("=" * 95)
    print(f"{'Cycle':<6} | {'Score':<11} | {'Effort':<7} | "
          f"{'Entropy':<8} | {'Tension':<8} | Qualia")
    print("-" * 95)

    tasks_to_run = [engine.queue.poll_inbox()]

    if AIOHTTP_OK:
        api     = RestAPI(engine)
        app     = api.build_app()
        runner  = web.AppRunner(app)
        await runner.setup()
        site    = web.TCPSite(runner, "0.0.0.0", REST_PORT)
        await site.start()
        print(f"[REST] Listening on :{REST_PORT}")

    async def engine_loop():
        try:
            while True:
                m = await engine.run_cycle()
                render(m)
                await asyncio.sleep(CYCLE_INTERVAL)
        except asyncio.CancelledError:
            pass

    tasks_to_run.append(engine_loop())

    try:
        await asyncio.gather(*tasks_to_run)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n" + "=" * 95)
        print("SHUTDOWN — STATE PRESERVED")
        print(f"Cycle       : {engine.cycle_count}")
        print(f"Score       : {engine.score:.4f}")
        print(f"Tools forged: {len(engine.elara.tools_forged)}")
        print(f"Graph nodes : {len(engine.graph.G.nodes)}")
        print(f"Vault snaps : {engine.vault.count()}")
        print("=" * 95)


if __name__ == "__main__":
    asyncio.run(main())
