#!/usr/bin/env python3
"""Parse LSST QuantumGraph aggregator logs to tabulate and plot scanning performance."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import re
import statistics
import sys

TIMESTAMP_RE = re.compile(
    r"^(?:INFO|DEBUG|WARN|WARNING|ERROR)\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+[+-]\d{2}:\d{2})"
)
READING_GRAPH_RE = re.compile(r"Reading predicted quantum graph\.")
ANALYZING_GRAPH_RE = re.compile(r"Analyzing predicted graph\.")
DEFENSIVE_INGEST_RE = re.compile(
    r"Some outputs seem to have already been ingested.*\[after\s+([\d.]+)s\]"
)
PROGRESS_RE = re.compile(
    r"Status after\s+([\d.]+)s:\s+scanning\s+\((\d+)\s+of\s+(\d+)\s+quanta\)"
)
# Matches output of:
#   python -c "import lsst.resources; print(lsst.resources.__file__, getattr(lsst.resources, '__version__', None))"
RESOURCES_FILE_RE = re.compile(
    r"^(\S*lsst/resources/__init__\.pyc?)(?:\s+(\S.*))?$"
)
# Matches output of `eups list | grep resources` (only the line tagged `setup`):
#   resources             LOCAL:/sdf/data/rubin/user/.../resources      setup
EUPS_WITH_PKG_RE = re.compile(
    r"^(?:lsst_)?resources\s+(\S+)\s+((?:.*\s)?setup(?:\s.*)?)$"
)
# Matches output of `eups list -s resources` (which omits the product column):
#      LOCAL:/sdf/data/rubin/user/.../resources         setup
#      g4354dbd74b+940ec9190d     current w_2026_46 setup
EUPS_NO_PKG_RE = re.compile(
    r"^\s*(LOCAL:\S+|g[0-9a-f]{7,}\S*|[vV]?\d+\.\d+\S*|tickets/\S+|main\S*|master\S*)\s+((?:.*\s)?setup(?:\s.*)?)$"
)
# Matches error when `eups list -s lsst_resources` was called instead of `resources`:
EUPS_ERR_RE = re.compile(r"^eups list:\s*(Unable to find product\s+\S+.*)$")
# Matches per-task HTTP summary lines from lsst.resources:
#   [HTTP-SUMMARY pid=1618 proc=scanner-001 up=3947.0s] reqs=19897 status={302: 9949, 200: 9948} | FE(new=9949@30.8ms reuse=0@0.0ms q=1) | BE(new=9948@35.5ms reuse=0@0.0ms pools=1 hosts=20) | read(n=0 avg=0.0ms) as_local(n=9948 get=67.9ms write=46.6ms [net=46.4ms disk=0.2ms] buf=1048576) dav={}
HTTP_SUMMARY_RE = re.compile(
    r"\[HTTP-SUMMARY\s+pid=(\d+)\s+proc=(\S+)\s+up=([\d.]+)s\]\s+"
    r"reqs=(\d+)\s+status=(\{[^}]*\})\s+\|\s+"
    r"FE\(new=(\d+)@([\d.]+)ms\s+reuse=(\d+)@([\d.]+)ms\s+q=(\d+)\)\s+\|\s+"
    r"BE\(new=(\d+)@([\d.]+)ms\s+reuse=(\d+)@([\d.]+)ms\s+pools=(\d+)\s+hosts=(\d+)\)\s+\|\s+"
    r"read\(n=(\d+)\s+avg=([\d.]+)ms\)\s+"
    r"as_local\(n=(\d+)\s+get=([\d.]+)ms\s+write=([\d.]+)ms\s+"
    r"(?:\[net=([\d.]+)ms\s+disk=([\d.]+)ms\]\s+)?buf=(\d+)\)\s+"
    r"dav=(\{[^}]*\})"
)


@dataclass
class HttpSummarySample:
    timestamp: datetime | None
    pid: int
    proc: str
    up_s: float
    reqs: int
    status: dict[int, int]
    fe_new: int
    fe_new_ms: float
    fe_reuse: int
    fe_reuse_ms: float
    fe_q: int
    be_new: int
    be_new_ms: float
    be_reuse: int
    be_reuse_ms: float
    be_pools: int
    be_hosts: int
    read_n: int
    read_avg_ms: float
    as_local_n: int
    as_local_get_ms: float
    as_local_write_ms: float
    as_local_net_ms: float | None
    as_local_disk_ms: float | None
    as_local_buf: int
    dav: dict[str, int]

    @property
    def fe_total(self) -> int:
        return self.fe_new + self.fe_reuse

    @property
    def fe_reuse_pct(self) -> float:
        return (100.0 * self.fe_reuse / self.fe_total) if self.fe_total else 0.0

    @property
    def be_total(self) -> int:
        return self.be_new + self.be_reuse

    @property
    def be_reuse_pct(self) -> float:
        return (100.0 * self.be_reuse / self.be_total) if self.be_total else 0.0

    @property
    def as_local_total_ms(self) -> float:
        return self.as_local_get_ms + self.as_local_write_ms

    @property
    def as_local_rate(self) -> float:
        return (self.as_local_n / self.up_s) if self.up_s > 0 else 0.0


@dataclass
class HttpAggregate:
    worker_count: int
    total_reports: int
    latest_workers: list[HttpSummarySample]
    max_up_s: float
    mean_up_s: float
    total_reqs: int
    status_counts: dict[int, int]
    fe_new_total: int
    fe_reuse_total: int
    fe_new_ms: float
    fe_reuse_ms: float
    fe_avg_ms: float
    fe_q_max: int
    be_new_total: int
    be_reuse_total: int
    be_new_ms: float
    be_reuse_ms: float
    be_avg_ms: float
    be_pools_max: int
    be_hosts_mean: float
    be_hosts_max: int
    read_n_total: int
    read_avg_ms: float
    as_local_n_total: int
    as_local_get_ms: float
    as_local_write_ms: float
    as_local_net_ms: float | None
    as_local_disk_ms: float | None
    as_local_total_ms: float
    as_local_bufs: list[int]
    as_local_agg_rate: float
    as_local_worker_rates: list[float]
    dav_counts: dict[str, int]

    @property
    def fe_reuse_pct(self) -> float:
        tot = self.fe_new_total + self.fe_reuse_total
        return (100.0 * self.fe_reuse_total / tot) if tot else 0.0

    @property
    def be_reuse_pct(self) -> float:
        tot = self.be_new_total + self.be_reuse_total
        return (100.0 * self.be_reuse_total / tot) if tot else 0.0


def _safe_literal_dict(raw: str) -> dict:
    try:
        val = ast.literal_eval(raw)
        return val if isinstance(val, dict) else {}
    except (SyntaxError, ValueError):
        return {}


@dataclass
class ProgressSample:
    timestamp: datetime | None
    elapsed_s: float
    scanned: int
    total: int
    dt_s: float = 0.0
    d_scanned: int = 0
    interval_rate: float = 0.0
    cumulative_rate: float = 0.0

    @property
    def pct_complete(self) -> float:
        return (100.0 * self.scanned / self.total) if self.total else 0.0


@dataclass
class LogMetrics:
    label: str
    path: Path
    resources_file: str | None = None
    resources_version: str | None = None
    eups_version: str | None = None
    eups_tags: str | None = None
    eups_error: str | None = None
    read_graph_ts: datetime | None = None
    analyze_graph_ts: datetime | None = None
    defensive_ingest_ts: datetime | None = None
    defensive_ingest_after_s: float | None = None
    samples: list[ProgressSample] = field(default_factory=list)
    http_summaries: list[HttpSummarySample] = field(default_factory=list)

    def http_aggregate(self) -> HttpAggregate | None:
        """Aggregate latest cumulative HTTP-SUMMARY snapshot per (pid, proc)."""
        if not self.http_summaries:
            return None
        latest_by_worker: dict[tuple[int, str], HttpSummarySample] = {}
        for h in self.http_summaries:
            key = (h.pid, h.proc)
            prev = latest_by_worker.get(key)
            if prev is None or h.up_s >= prev.up_s:
                latest_by_worker[key] = h

        workers = sorted(latest_by_worker.values(), key=lambda x: (x.proc, x.pid))

        def wavg(pairs: list[tuple[int, float]]) -> float:
            tot_n = sum(n for n, _ in pairs)
            if tot_n <= 0:
                return 0.0
            return sum(n * v for n, v in pairs) / tot_n

        status_counts: dict[int, int] = {}
        dav_counts: dict[str, int] = {}
        for w in workers:
            for k, v in w.status.items():
                status_counts[k] = status_counts.get(k, 0) + int(v)
            for k, v in w.dav.items():
                dav_counts[str(k)] = dav_counts.get(str(k), 0) + int(v)

        fe_new_tot = sum(w.fe_new for w in workers)
        fe_reuse_tot = sum(w.fe_reuse for w in workers)
        fe_new_ms = wavg([(w.fe_new, w.fe_new_ms) for w in workers])
        fe_reuse_ms = wavg([(w.fe_reuse, w.fe_reuse_ms) for w in workers])
        fe_avg_ms = wavg(
            [(w.fe_new, w.fe_new_ms) for w in workers]
            + [(w.fe_reuse, w.fe_reuse_ms) for w in workers]
        )

        be_new_tot = sum(w.be_new for w in workers)
        be_reuse_tot = sum(w.be_reuse for w in workers)
        be_new_ms = wavg([(w.be_new, w.be_new_ms) for w in workers])
        be_reuse_ms = wavg([(w.be_reuse, w.be_reuse_ms) for w in workers])
        be_avg_ms = wavg(
            [(w.be_new, w.be_new_ms) for w in workers]
            + [(w.be_reuse, w.be_reuse_ms) for w in workers]
        )

        read_n_tot = sum(w.read_n for w in workers)
        read_avg_ms = wavg([(w.read_n, w.read_avg_ms) for w in workers])

        al_n_tot = sum(w.as_local_n for w in workers)
        al_get_ms = wavg([(w.as_local_n, w.as_local_get_ms) for w in workers])
        al_write_ms = wavg([(w.as_local_n, w.as_local_write_ms) for w in workers])
        net_pairs = [
            (w.as_local_n, w.as_local_net_ms)
            for w in workers
            if w.as_local_net_ms is not None
        ]
        disk_pairs = [
            (w.as_local_n, w.as_local_disk_ms)
            for w in workers
            if w.as_local_disk_ms is not None
        ]
        al_net_ms = wavg(net_pairs) if net_pairs else None
        al_disk_ms = wavg(disk_pairs) if disk_pairs else None

        worker_rates = [w.as_local_rate for w in workers if w.up_s > 0]
        agg_rate = sum(worker_rates)

        return HttpAggregate(
            worker_count=len(workers),
            total_reports=len(self.http_summaries),
            latest_workers=workers,
            max_up_s=max((w.up_s for w in workers), default=0.0),
            mean_up_s=statistics.mean(w.up_s for w in workers) if workers else 0.0,
            total_reqs=sum(w.reqs for w in workers),
            status_counts=dict(sorted(status_counts.items())),
            fe_new_total=fe_new_tot,
            fe_reuse_total=fe_reuse_tot,
            fe_new_ms=fe_new_ms,
            fe_reuse_ms=fe_reuse_ms,
            fe_avg_ms=fe_avg_ms,
            fe_q_max=max((w.fe_q for w in workers), default=0),
            be_new_total=be_new_tot,
            be_reuse_total=be_reuse_tot,
            be_new_ms=be_new_ms,
            be_reuse_ms=be_reuse_ms,
            be_avg_ms=be_avg_ms,
            be_pools_max=max((w.be_pools for w in workers), default=0),
            be_hosts_mean=statistics.mean(w.be_hosts for w in workers) if workers else 0.0,
            be_hosts_max=max((w.be_hosts for w in workers), default=0),
            read_n_total=read_n_tot,
            read_avg_ms=read_avg_ms,
            as_local_n_total=al_n_tot,
            as_local_get_ms=al_get_ms,
            as_local_write_ms=al_write_ms,
            as_local_net_ms=al_net_ms,
            as_local_disk_ms=al_disk_ms,
            as_local_total_ms=al_get_ms + al_write_ms,
            as_local_bufs=sorted({w.as_local_buf for w in workers}),
            as_local_agg_rate=agg_rate,
            as_local_worker_rates=worker_rates,
            dav_counts=dict(sorted(dav_counts.items())),
        )

    @property
    def has_version_info(self) -> bool:
        return any(
            x is not None
            for x in (
                self.resources_file,
                self.resources_version,
                self.eups_version,
                self.eups_error,
            )
        )

    @property
    def is_local_eups(self) -> bool:
        if self.eups_version and self.eups_version.startswith("LOCAL:"):
            return True
        if self.eups_tags and "LOCAL:" in self.eups_tags:
            return True
        return False

    @property
    def is_local_file_path(self) -> bool | None:
        if not self.resources_file:
            return None
        shared_markers = ("/conda/envs/", "/cvmfs/", "/site-packages/", "/share/eups/")
        return not any(m in self.resources_file for m in shared_markers)

    @property
    def version_summary(self) -> str:
        parts: list[str] = []
        if self.eups_version:
            tag_str = f" [{self.eups_tags}]" if self.eups_tags else ""
            parts.append(f"eups={self.eups_version}{tag_str}")
        elif self.eups_error:
            parts.append(f"eups_error={self.eups_error}")
        if self.resources_version:
            parts.append(f"__version__={self.resources_version}")
        if self.resources_file:
            parts.append(f"__file__={self.resources_file}")
        return " | ".join(parts) if parts else "NOT FOUND IN LOG"

    @property
    def read_graph_duration_s(self) -> float | None:
        if self.read_graph_ts and self.analyze_graph_ts:
            return (self.analyze_graph_ts - self.read_graph_ts).total_seconds()
        return None

    @property
    def analyze_to_defensive_s(self) -> float | None:
        if self.analyze_graph_ts and self.defensive_ingest_ts:
            return (self.defensive_ingest_ts - self.analyze_graph_ts).total_seconds()
        return None

    @property
    def total_quanta(self) -> int:
        return self.samples[-1].total if self.samples else 0

    @property
    def final_elapsed_s(self) -> float:
        return self.samples[-1].elapsed_s if self.samples else 0.0

    @property
    def final_scanned(self) -> int:
        return self.samples[-1].scanned if self.samples else 0

    @property
    def overall_rate(self) -> float:
        if not self.samples or self.final_elapsed_s <= 0:
            return 0.0
        return self.final_scanned / self.final_elapsed_s

    @property
    def active_scan_rate(self) -> float:
        """Overall rate from end of defensive ingest setup to final sample."""
        if not self.samples:
            return 0.0
        setup_s = self.defensive_ingest_after_s or 0.0
        active_s = self.final_elapsed_s - setup_s
        return self.final_scanned / active_s if active_s > 0 else 0.0

    @property
    def steady_state_rate(self) -> float:
        """Rate from the first progress report to the last (excludes initial setup)."""
        if len(self.samples) < 2:
            return self.overall_rate
        dt = self.samples[-1].elapsed_s - self.samples[0].elapsed_s
        dn = self.samples[-1].scanned - self.samples[0].scanned
        return dn / dt if dt > 0 else 0.0

    @property
    def estimated_total_hours(self) -> float:
        rate = self.steady_state_rate or self.overall_rate
        if rate <= 0 or not self.total_quanta:
            return 0.0
        setup_s = self.defensive_ingest_after_s or 0.0
        return (setup_s + (self.total_quanta / rate)) / 3600.0


def parse_log(path: Path, label: str | None = None) -> LogMetrics:
    metrics = LogMetrics(label=label or path.stem, path=path)
    prev_elapsed = 0.0
    prev_scanned = 0

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("+"):
                continue

            # Check for python lsst.resources.__file__ / __version__ line
            if "import lsst.resources" not in line and (
                m := RESOURCES_FILE_RE.match(line)
            ):
                metrics.resources_file = m.group(1)
                if m.group(2):
                    metrics.resources_version = m.group(2).strip()
                continue

            # Check for `eups list -s resources` or `eups list | grep resources`
            if m := EUPS_WITH_PKG_RE.match(line):
                metrics.eups_version = m.group(1)
                metrics.eups_tags = (m.group(2) or "").strip() or None
                metrics.eups_error = None
                continue
            if m := EUPS_NO_PKG_RE.match(line):
                metrics.eups_version = m.group(1)
                metrics.eups_tags = m.group(2).strip()
                metrics.eups_error = None
                continue
            if m := EUPS_ERR_RE.match(line):
                if not metrics.eups_version:
                    metrics.eups_error = m.group(1)
                continue

            ts_match = TIMESTAMP_RE.match(line)
            ts = datetime.fromisoformat(ts_match.group(1)) if ts_match else None

            if m := HTTP_SUMMARY_RE.search(line):
                metrics.http_summaries.append(
                    HttpSummarySample(
                        timestamp=ts,
                        pid=int(m.group(1)),
                        proc=m.group(2),
                        up_s=float(m.group(3)),
                        reqs=int(m.group(4)),
                        status=_safe_literal_dict(m.group(5)),
                        fe_new=int(m.group(6)),
                        fe_new_ms=float(m.group(7)),
                        fe_reuse=int(m.group(8)),
                        fe_reuse_ms=float(m.group(9)),
                        fe_q=int(m.group(10)),
                        be_new=int(m.group(11)),
                        be_new_ms=float(m.group(12)),
                        be_reuse=int(m.group(13)),
                        be_reuse_ms=float(m.group(14)),
                        be_pools=int(m.group(15)),
                        be_hosts=int(m.group(16)),
                        read_n=int(m.group(17)),
                        read_avg_ms=float(m.group(18)),
                        as_local_n=int(m.group(19)),
                        as_local_get_ms=float(m.group(20)),
                        as_local_write_ms=float(m.group(21)),
                        as_local_net_ms=float(m.group(22)) if m.group(22) is not None else None,
                        as_local_disk_ms=float(m.group(23)) if m.group(23) is not None else None,
                        as_local_buf=int(m.group(24)),
                        dav=_safe_literal_dict(m.group(25)),
                    )
                )
                continue

            if READING_GRAPH_RE.search(line):
                metrics.read_graph_ts = ts
            elif ANALYZING_GRAPH_RE.search(line):
                metrics.analyze_graph_ts = ts
            elif m := DEFENSIVE_INGEST_RE.search(line):
                metrics.defensive_ingest_ts = ts
                metrics.defensive_ingest_after_s = float(m.group(1))
            elif m := PROGRESS_RE.search(line):
                elapsed_s = float(m.group(1))
                scanned = int(m.group(2))
                total = int(m.group(3))

                dt_s = elapsed_s - prev_elapsed
                d_scanned = scanned - prev_scanned
                interval_rate = d_scanned / dt_s if dt_s > 0 else 0.0
                cum_rate = scanned / elapsed_s if elapsed_s > 0 else 0.0

                metrics.samples.append(
                    ProgressSample(
                        timestamp=ts,
                        elapsed_s=elapsed_s,
                        scanned=scanned,
                        total=total,
                        dt_s=dt_s,
                        d_scanned=d_scanned,
                        interval_rate=interval_rate,
                        cumulative_rate=cum_rate,
                    )
                )
                prev_elapsed = elapsed_s
                prev_scanned = scanned

    return metrics


def verify_versions(runs: list[LogMetrics]) -> None:
    """Print a non-fatal verification summary of lsst.resources versions across runs."""
    print("\n=== lsst.resources Version Check ===")
    for r in runs:
        if not r.has_version_info:
            print(
                f"  [{r.label}] INFO: No lsst.resources / eups version lines found in log (continuing)."
            )
            continue

        print(f"  [{r.label}]")
        if r.resources_file:
            print(f"    __file__    : {r.resources_file}")
        if r.resources_version:
            print(f"    __version__ : {r.resources_version}")
        if r.eups_version:
            tags = f" ({r.eups_tags})" if r.eups_tags else ""
            print(f"    eups list   : {r.eups_version}{tags}")
        elif r.eups_error:
            print(
                f"    eups list   : {r.eups_error} (tip: EUPS product name is 'resources', not 'lsst_resources')"
            )

        is_ticket_label = bool(
            re.search(r"(?:ticket|dm[-_]\d+)", r.label, re.IGNORECASE)
        )
        if r.is_local_eups:
            print(
                f"    status      : OK - using local EUPS override ({r.eups_version})"
            )
        elif r.is_local_file_path is True:
            print(
                f"    status      : OK - __file__ points to local checkout ({r.resources_file})"
            )
        elif is_ticket_label and (r.eups_version or r.is_local_file_path is False):
            detail = (
                f"EUPS='{r.eups_version}'"
                if r.eups_version
                else f"__file__='{r.resources_file}'"
            )
            print(
                f"    status      : WARNING - label '{r.label}' looks like a ticket branch, "
                f"but {detail} appears to be the shared stack install"
            )

    # If multiple runs have version info, verify they aren't accidentally identical
    with_info = [r for r in runs if r.has_version_info]
    if len(with_info) >= 2:
        first = with_info[0]
        for other in with_info[1:]:
            same_file = (
                first.resources_file
                and other.resources_file
                and first.resources_file == other.resources_file
            )
            same_eups = (
                first.eups_version
                and other.eups_version
                and first.eups_version == other.eups_version
            )
            if same_file or same_eups:
                print(
                    f"  [WARNING] '{first.label}' and '{other.label}' report the SAME "
                    f"lsst.resources setup ({first.version_summary})!"
                )
            else:
                print(
                    f"  [OK] '{first.label}' and '{other.label}' are using distinct "
                    f"lsst.resources versions."
                )


def print_http_summary(m: LogMetrics, show_workers: bool = False) -> None:
    agg = m.http_aggregate()
    if agg is None:
        return

    print(
        f"\n  --- HTTP-SUMMARY Diagnostics ({agg.worker_count} workers, {agg.total_reports} reports, "
        f"max up={agg.max_up_s:.1f}s) ---"
    )
    print(f"  Total HTTP requests      : {agg.total_reqs:,d}  status={agg.status_counts}")
    if agg.dav_counts:
        print(f"  WebDAV calls             : {agg.dav_counts}")
    print(
        f"  Frontend (FE)            : new={agg.fe_new_total:,d}@{agg.fe_new_ms:.1f}ms, "
        f"reuse={agg.fe_reuse_total:,d}@{agg.fe_reuse_ms:.1f}ms "
        f"(reuse={agg.fe_reuse_pct:.1f}%, avg={agg.fe_avg_ms:.1f}ms, max_q={agg.fe_q_max})"
    )
    print(
        f"  Backend  (BE)            : new={agg.be_new_total:,d}@{agg.be_new_ms:.1f}ms, "
        f"reuse={agg.be_reuse_total:,d}@{agg.be_reuse_ms:.1f}ms "
        f"(reuse={agg.be_reuse_pct:.1f}%, avg={agg.be_avg_ms:.1f}ms, "
        f"max_pools={agg.be_pools_max}, hosts avg={agg.be_hosts_mean:.1f}/max={agg.be_hosts_max})"
    )
    if agg.read_n_total > 0:
        print(
            f"  read()                   : n={agg.read_n_total:,d}, avg={agg.read_avg_ms:.1f}ms"
        )
    buf_str = ", ".join(f"{b:,d}B" for b in agg.as_local_bufs)
    net_disk_str = (
        f" [net={agg.as_local_net_ms:.1f}ms, disk={agg.as_local_disk_ms:.1f}ms]"
        if agg.as_local_net_ms is not None and agg.as_local_disk_ms is not None
        else ""
    )
    print(
        f"  as_local()               : n={agg.as_local_n_total:,d}, total={agg.as_local_total_ms:.1f}ms "
        f"(get={agg.as_local_get_ms:.1f}ms + write={agg.as_local_write_ms:.1f}ms"
        f"{net_disk_str}, buf={buf_str})"
    )
    if agg.as_local_worker_rates:
        print(
            f"  as_local() throughput    : agg={agg.as_local_agg_rate:.2f} files/s "
            f"(per-worker: min={min(agg.as_local_worker_rates):.2f}, "
            f"median={statistics.median(agg.as_local_worker_rates):.2f}, "
            f"max={max(agg.as_local_worker_rates):.2f} files/s)"
        )

    if show_workers:
        print(
            f"\n  {'Worker':<14} | {'PID':>7} | {'Up (s)':>8} | {'as_local':>8} | "
            f"{'Rate/s':>6} | {'FE ms':>6} | {'FE%':>5} | {'BE ms':>6} | {'BE%':>5} | "
            f"{'get':>6} | {'net':>6} | {'disk':>5} | {'hosts':>5}"
        )
        print("  " + "-" * 116)
        for w in agg.latest_workers:
            fe_avg = (
                (w.fe_new * w.fe_new_ms + w.fe_reuse * w.fe_reuse_ms) / w.fe_total
                if w.fe_total
                else 0.0
            )
            be_avg = (
                (w.be_new * w.be_new_ms + w.be_reuse * w.be_reuse_ms) / w.be_total
                if w.be_total
                else 0.0
            )
            net_s = f"{w.as_local_net_ms:6.1f}" if w.as_local_net_ms is not None else f"{'N/A':>6}"
            disk_s = f"{w.as_local_disk_ms:5.1f}" if w.as_local_disk_ms is not None else f"{'N/A':>5}"
            print(
                f"  {w.proc:<14} | {w.pid:7d} | {w.up_s:8.1f} | {w.as_local_n:8,d} | "
                f"{w.as_local_rate:6.2f} | {fe_avg:6.1f} | {w.fe_reuse_pct:4.1f}% | "
                f"{be_avg:6.1f} | {w.be_reuse_pct:4.1f}% | "
                f"{w.as_local_get_ms:6.1f} | {net_s} | {disk_s} | {w.be_hosts:5d}"
            )


def print_single_table(m: LogMetrics, show_http_workers: bool = False) -> None:
    print(f"\n=== Timing Table: {m.label} ({m.path}) ===")
    if m.has_version_info:
        print(f"  lsst.resources version info     : {m.version_summary}")
    else:
        print("  lsst.resources version info     : N/A (not present in log)")
    if m.read_graph_duration_s is not None:
        print(f"  Reading predicted quantum graph : {m.read_graph_duration_s:7.2f} s")
    if m.analyze_to_defensive_s is not None:
        print(f"  Analyzing -> defensive ingest   : {m.analyze_to_defensive_s:7.2f} s")
    if m.defensive_ingest_after_s is not None:
        print(
            f"  Total pre-scan (logged [after]) : {m.defensive_ingest_after_s:7.2f} s"
        )

    if not m.samples:
        print("  No progress lines found.")
        print_http_summary(m, show_workers=show_http_workers)
        return

    print(
        f"\n{'Elapsed (s)':>11} | {'Elapsed (h)':>11} | {'Scanned':>10} | "
        f"{'Progress':>8} | {'dQuanta':>8} | {'Int Rate (q/s)':>14} | {'Cum Rate (q/s)':>14}"
    )
    print("-" * 92)
    for s in m.samples:
        print(
            f"{s.elapsed_s:11.1f} | {s.elapsed_s / 3600.0:11.2f} | {s.scanned:10,d} | "
            f"{s.pct_complete:7.2f}% | {s.d_scanned:8,d} | {s.interval_rate:14.2f} | {s.cumulative_rate:14.2f}"
        )

    steady_rates = [s.interval_rate for s in m.samples[1:]] or [
        s.interval_rate for s in m.samples
    ]
    print("-" * 92)
    print(f"  Total quanta             : {m.total_quanta:,d}")
    print(
        f"  Scanned after {m.final_elapsed_s:.1f}s    : {m.final_scanned:,d} ({m.samples[-1].pct_complete:.2f}%)"
    )
    print(f"  Overall cumulative rate  : {m.overall_rate:.2f} quanta/s")
    print(
        f"  Active scan rate         : {m.active_scan_rate:.2f} quanta/s (excluding pre-scan)"
    )
    print(f"  Steady-state rate (t>600): {m.steady_state_rate:.2f} quanta/s")
    print(
        f"  Interval rate (t>600)    : mean={statistics.mean(steady_rates):.2f}, "
        f"median={statistics.median(steady_rates):.2f}, "
        f"min={min(steady_rates):.2f}, max={max(steady_rates):.2f} quanta/s"
    )
    print(f"  Projected total duration : {m.estimated_total_hours:.2f} hours")
    print_http_summary(m, show_workers=show_http_workers)


def print_comparison_table(runs: list[LogMetrics]) -> None:
    if len(runs) < 2:
        return
    base = runs[0]
    print("\n=== Summary Comparison ===")
    header = (
        f"{'Metric':<34} | "
        + " | ".join(f"{r.label:>16}" for r in runs)
        + (
            f" | {'Diff (' + runs[1].label + ' vs ' + base.label + ')':>24}"
            if len(runs) == 2
            else ""
        )
    )
    print(header)
    print("-" * len(header))

    def fmt_row(name: str, vals: list[float | None], unit: str = "") -> None:
        cells = [
            f"{v:13.2f} {unit:<2}" if v is not None else f"{'N/A':>16}" for v in vals
        ]
        diff_str = ""
        if len(runs) == 2 and vals[0] is not None and vals[1] is not None:
            delta = vals[1] - vals[0]
            if vals[0] != 0:
                pct = 100.0 * delta / vals[0]
                diff_str = f" | {delta:+10.2f} {unit:<2} ({pct:+6.2f}%)"
            else:
                diff_str = f" | {delta:+10.2f} {unit:<2} (   N/A)"
        print(f"{name:<34} | " + " | ".join(cells) + diff_str)

    fmt_row("Read predicted graph", [r.read_graph_duration_s for r in runs], "s")
    fmt_row(
        "Analyze -> defensive ingest", [r.analyze_to_defensive_s for r in runs], "s"
    )
    fmt_row("Logged pre-scan [after]", [r.defensive_ingest_after_s for r in runs], "s")
    fmt_row(
        "Quanta @ 600s",
        [float(r.samples[0].scanned) if r.samples else None for r in runs],
        "q",
    )
    fmt_row(
        "Active scan rate @ 600s",
        [
            r.samples[0].scanned / (r.samples[0].elapsed_s - (r.defensive_ingest_after_s or 0.0))
            if r.samples and (r.samples[0].elapsed_s - (r.defensive_ingest_after_s or 0.0)) > 0
            else None
            for r in runs
        ],
        "q/s",
    )

    # Compare at the last common step when logs have different lengths
    common_n = min((len(r.samples) for r in runs if r.samples), default=0)
    if common_n > 0 and len({len(r.samples) for r in runs}) > 1:
        common_t = runs[0].samples[common_n - 1].elapsed_s
        fmt_row(
            f"Quanta @ {common_t:.0f}s (step {common_n})",
            [float(r.samples[common_n - 1].scanned) for r in runs],
            "q",
        )
        fmt_row(
            f"Steady rate (600s->{common_t:.0f}s)",
            [
                (r.samples[common_n - 1].scanned - r.samples[0].scanned)
                / (r.samples[common_n - 1].elapsed_s - r.samples[0].elapsed_s)
                if common_n >= 2
                and (r.samples[common_n - 1].elapsed_s - r.samples[0].elapsed_s) > 0
                else None
                for r in runs
            ],
            "q/s",
        )

    fmt_row("Final quanta scanned", [float(r.final_scanned) for r in runs], "q")
    fmt_row("Overall rate", [r.overall_rate for r in runs], "q/s")
    fmt_row("Active scan rate (post-setup)", [r.active_scan_rate for r in runs], "q/s")
    fmt_row("Steady-state rate (t>600s)", [r.steady_state_rate for r in runs], "q/s")
    fmt_row("Projected total time", [r.estimated_total_hours for r in runs], "h")

    if len(runs) == 2 and runs[0].samples and runs[1].samples:
        a, b = runs[0], runs[1]
        n = min(len(a.samples), len(b.samples))
        print(f"\n=== Step-by-Step Comparison ({b.label} vs {a.label}) ===")
        print(
            f"{'Step':>4} | {'Elapsed (s)':>11} | "
            f"{a.label + ' Quanta':>16} | {b.label + ' Quanta':>16} | {'Delta Quanta':>12} | "
            f"{a.label + ' q/s':>14} | {b.label + ' q/s':>16} | {'Delta q/s':>10}"
        )
        print("-" * 115)
        for i in range(n):
            sa, sb = a.samples[i], b.samples[i]
            dq = sb.scanned - sa.scanned
            dr = sb.interval_rate - sa.interval_rate
            print(
                f"{i + 1:4d} | {sa.elapsed_s:11.1f} | "
                f"{sa.scanned:16,d} | {sb.scanned:16,d} | {dq:+12,d} | "
                f"{sa.interval_rate:14.2f} | {sb.interval_rate:16.2f} | {dr:+10.2f}"
            )

    aggs = [r.http_aggregate() for r in runs]
    if any(a is not None for a in aggs):
        print("\n=== HTTP-SUMMARY Comparison ===")
        print(header)
        print("-" * len(header))
        fmt_row(
            "HTTP workers",
            [float(a.worker_count) if a else None for a in aggs],
            "",
        )
        fmt_row(
            "Total HTTP requests",
            [float(a.total_reqs) if a else None for a in aggs],
            "",
        )
        fmt_row(
            "FE reuse rate",
            [a.fe_reuse_pct if a else None for a in aggs],
            "%",
        )
        fmt_row(
            "FE avg latency",
            [a.fe_avg_ms if a else None for a in aggs],
            "ms",
        )
        fmt_row(
            "BE reuse rate",
            [a.be_reuse_pct if a else None for a in aggs],
            "%",
        )
        fmt_row(
            "BE avg latency",
            [a.be_avg_ms if a else None for a in aggs],
            "ms",
        )
        fmt_row(
            "BE hosts (avg/worker)",
            [a.be_hosts_mean if a else None for a in aggs],
            "",
        )
        fmt_row(
            "as_local() count",
            [float(a.as_local_n_total) if a else None for a in aggs],
            "",
        )
        fmt_row(
            "as_local() total avg",
            [a.as_local_total_ms if a else None for a in aggs],
            "ms",
        )
        fmt_row(
            "as_local() get avg",
            [a.as_local_get_ms if a else None for a in aggs],
            "ms",
        )
        fmt_row(
            "as_local() write avg",
            [a.as_local_write_ms if a else None for a in aggs],
            "ms",
        )
        fmt_row(
            "as_local() write net avg",
            [a.as_local_net_ms if a else None for a in aggs],
            "ms",
        )
        fmt_row(
            "as_local() write disk avg",
            [a.as_local_disk_ms if a else None for a in aggs],
            "ms",
        )
        fmt_row(
            "as_local() agg throughput",
            [a.as_local_agg_rate if a else None for a in aggs],
            "/s",
        )
        fmt_row(
            "as_local() median worker rate",
            [
                statistics.median(a.as_local_worker_rates)
                if a and a.as_local_worker_rates
                else None
                for a in aggs
            ],
            "/s",
        )


def plot_metrics(runs: list[LogMetrics], output_path: Path, show: bool = False) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        svg_path = output_path.with_suffix(".svg")
        _write_svg_fallback(runs, svg_path)
        print(
            f"\n[NOTE] matplotlib is not installed in {sys.executable}; "
            f"wrote standalone SVG plot to {svg_path}"
        )
        return

    has_comparison = len(runs) >= 2
    nrows = 3 if has_comparison else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 3.6 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    # Panel 1: Cumulative quanta scanned vs time
    ax0 = axes[0]
    for r in runs:
        t_hr = [s.elapsed_s / 3600.0 for s in r.samples]
        q_k = [s.scanned / 1e3 for s in r.samples]
        ax0.plot(
            t_hr,
            q_k,
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            label=f"{r.label} ({r.overall_rate:.2f} q/s)",
        )
    ax0.set_ylabel("Scanned Quanta (thousands)")
    ax0.set_title("QuantumGraph Aggregator Scanning Progress & Rate")
    ax0.grid(True, linestyle="--", alpha=0.5)
    ax0.legend(loc="upper left")

    # Panel 2: Interval & Cumulative scanning rate vs time
    ax1 = axes[1]
    for r in runs:
        t_hr = [s.elapsed_s / 3600.0 for s in r.samples]
        int_rate = [s.interval_rate for s in r.samples]
        cum_rate = [s.cumulative_rate for s in r.samples]
        line = ax1.plot(
            t_hr,
            int_rate,
            marker="o",
            markersize=3.5,
            linewidth=1.5,
            label=f"{r.label} interval rate",
        )
        ax1.plot(
            t_hr,
            cum_rate,
            linestyle="--",
            linewidth=1.2,
            color=line[0].get_color(),
            alpha=0.8,
            label=f"{r.label} cumulative rate",
        )
    ax1.set_ylabel("Scan Rate (quanta / s)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", ncol=2, fontsize="small")

    # Panel 3: Delta between runs (if 2+ runs provided)
    if has_comparison:
        ax2 = axes[2]
        base = runs[0]
        for r in runs[1:]:
            n = min(len(base.samples), len(r.samples))
            t_hr = [base.samples[i].elapsed_s / 3600.0 for i in range(n)]
            dq = [r.samples[i].scanned - base.samples[i].scanned for i in range(n)]
            ax2.plot(
                t_hr,
                dq,
                marker="s",
                markersize=3.5,
                linewidth=1.6,
                color="tab:green" if len(runs) == 2 else None,
                label=f"{r.label} - {base.label} (quanta lead)",
            )
        ax2.axhline(0, color="black", linewidth=0.8, linestyle=":")
        ax2.set_ylabel("Quanta Difference")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="best")

    axes[-1].set_xlabel("Elapsed Time (hours)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"\nSaved plot to {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def _write_svg_fallback(runs: list[LogMetrics], svg_path: Path) -> None:
    """Generate a clean multi-panel SVG plot using only the Python standard library."""
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    has_comp = len(runs) >= 2
    width = 860
    panel_h = 220
    gap = 45
    top_margin = 50
    bottom_margin = 55
    left_margin = 80
    right_margin = 30
    nrows = 3 if has_comp else 2
    height = top_margin + nrows * panel_h + (nrows - 1) * gap + bottom_margin
    plot_w = width - left_margin - right_margin

    all_t = [s.elapsed_s / 3600.0 for r in runs for s in r.samples] or [0.0, 1.0]
    t_min, t_max = 0.0, max(all_t) * 1.02

    def sx(t: float) -> float:
        return left_margin + (t - t_min) / (t_max - t_min or 1.0) * plot_w

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" ',
        f'viewBox="0 0 {width} {height}" font-family="sans-serif" font-size="12">',
        f'<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="16" font-weight="bold" fill="#222">',
        f"QuantumGraph Aggregator Scanning Progress &amp; Rate</text>",
    ]

    def draw_panel(
        row: int,
        y_min: float,
        y_max: float,
        ylabel: str,
        series: list[tuple[str, list[tuple[float, float]], str, str]],
    ) -> None:
        y0 = top_margin + row * (panel_h + gap)
        lines.append(
            f'<rect x="{left_margin}" y="{y0}" width="{plot_w}" height="{panel_h}" '
            f'fill="#fafafa" stroke="#cccccc" stroke-width="1"/>'
        )
        # Grid and Y ticks
        for i in range(5):
            frac = i / 4.0
            val = y_min + frac * (y_max - y_min)
            py = y0 + panel_h - frac * panel_h
            lines.append(
                f'<line x1="{left_margin}" y1="{py:.1f}" x2="{left_margin + plot_w}" y2="{py:.1f}" '
                f'stroke="#e5e5e5" stroke-dasharray="3,3"/>'
            )
            lines.append(
                f'<text x="{left_margin - 8}" y="{py + 4:.1f}" text-anchor="end" fill="#555" font-size="11">'
                f"{val:,.1f}</text>"
            )
        # X ticks
        n_xticks = 6
        for i in range(n_xticks + 1):
            frac = i / n_xticks
            t_val = t_min + frac * (t_max - t_min)
            px = sx(t_val)
            lines.append(
                f'<line x1="{px:.1f}" y1="{y0}" x2="{px:.1f}" y2="{y0 + panel_h}" '
                f'stroke="#e5e5e5" stroke-dasharray="3,3"/>'
            )
            if row == nrows - 1:
                lines.append(
                    f'<text x="{px:.1f}" y="{y0 + panel_h + 18}" text-anchor="middle" fill="#555" font-size="11">'
                    f"{t_val:.1f}h</text>"
                )
        # Y axis label
        cy = y0 + panel_h / 2
        lines.append(
            f'<text x="20" y="{cy:.1f}" text-anchor="middle" fill="#222" font-size="12" '
            f'transform="rotate(-90 20 {cy:.1f})">{ylabel}</text>'
        )

        def sy(v: float) -> float:
            return y0 + panel_h - (v - y_min) / (y_max - y_min or 1.0) * panel_h

        # Plot series
        legend_x = left_margin + 14
        legend_y = y0 + 20
        for name, pts, color, dash in series:
            if not pts:
                continue
            poly = " ".join(f"{sx(t):.1f},{sy(v):.1f}" for t, v in pts)
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            lines.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2"{dash_attr} points="{poly}"/>'
            )
            if not dash:
                for t, v in pts:
                    lines.append(
                        f'<circle cx="{sx(t):.1f}" cy="{sy(v):.1f}" r="2.8" fill="{color}"/>'
                    )
            lines.append(
                f'<line x1="{legend_x}" y1="{legend_y - 4}" x2="{legend_x + 20}" y2="{legend_y - 4}" '
                f'stroke="{color}" stroke-width="2"{dash_attr}/>'
            )
            lines.append(
                f'<text x="{legend_x + 26}" y="{legend_y}" fill="#222" font-size="11">{name}</text>'
            )
            legend_y += 18

    # Panel 1: Cumulative Quanta (thousands)
    max_q_k = (
        max((s.scanned / 1e3 for r in runs for s in r.samples), default=1.0) * 1.08
    )
    p1_series = [
        (
            f"{r.label} ({r.overall_rate:.2f} q/s)",
            [(s.elapsed_s / 3600.0, s.scanned / 1e3) for s in r.samples],
            colors[idx % len(colors)],
            "",
        )
        for idx, r in enumerate(runs)
    ]
    draw_panel(0, 0.0, max_q_k, "Scanned Quanta (k)", p1_series)

    # Panel 2: Interval & Cumulative Rate
    all_rates = [s.interval_rate for r in runs for s in r.samples] + [
        s.cumulative_rate for r in runs for s in r.samples
    ]
    r_min = max(0.0, min(all_rates, default=0.0) - 1.5)
    r_max = max(all_rates, default=1.0) + 1.5
    p2_series = []
    for idx, r in enumerate(runs):
        c = colors[idx % len(colors)]
        p2_series.append(
            (
                f"{r.label} interval rate",
                [(s.elapsed_s / 3600.0, s.interval_rate) for s in r.samples],
                c,
                "",
            )
        )
        p2_series.append(
            (
                f"{r.label} cumulative rate",
                [(s.elapsed_s / 3600.0, s.cumulative_rate) for s in r.samples],
                c,
                "5,4",
            )
        )
    draw_panel(1, r_min, r_max, "Rate (quanta / s)", p2_series)

    # Panel 3: Quanta Lead
    if has_comp:
        base = runs[0]
        p3_series = []
        all_dq = [0.0]
        for idx, r in enumerate(runs[1:], start=1):
            n = min(len(base.samples), len(r.samples))
            pts = [
                (
                    base.samples[i].elapsed_s / 3600.0,
                    float(r.samples[i].scanned - base.samples[i].scanned),
                )
                for i in range(n)
            ]
            all_dq.extend(v for _, v in pts)
            p3_series.append(
                (
                    f"{r.label} - {base.label} (quanta)",
                    pts,
                    colors[(idx + 1) % len(colors)],
                    "",
                )
            )
        dq_min = min(all_dq) - 50.0
        dq_max = max(all_dq) + 50.0
        draw_panel(2, dq_min, dq_max, "Delta Quanta", p3_series)

    lines.append(
        f'<text x="{left_margin + plot_w / 2}" y="{height - 15}" text-anchor="middle" '
        f'fill="#222" font-size="13">Elapsed Time (hours)</text>'
    )
    lines.append("</svg>")
    svg_path.write_text("\n".join(lines), encoding="utf-8")


def parse_input_spec(spec: str) -> tuple[str, Path]:
    """Parse 'label=path' or 'path'."""
    if "=" in spec and not Path(spec).exists():
        label, path_str = spec.split("=", 1)
        return label, Path(path_str)
    p = Path(spec)
    return p.stem, p


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Quantify (tabulate) timing information and plot progress from LSST aggregator logs."
    )
    parser.add_argument(
        "logs",
        nargs="+",
        help="Log file path(s), optionally prefixed with a label as 'LABEL=path/to/file.log'.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("scan_timing.png"),
        help="Output plot filename (default: scan_timing.png; falls back to .svg if matplotlib is unavailable).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to export parsed interval data as CSV.",
    )
    parser.add_argument(
        "--http-workers",
        action="store_true",
        help="Print per-worker HTTP-SUMMARY breakdown table when HTTP-SUMMARY lines are present.",
    )
    parser.add_argument(
        "-n",
        "--no-plot",
        action="store_true",
        help="Suppress plot generation completely (text/table output only).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plot interactively in addition to saving.",
    )
    args = parser.parse_args(argv)

    runs: list[LogMetrics] = []
    for spec in args.logs:
        label, path = parse_input_spec(spec)
        if not path.exists():
            print(f"Error: log file not found: {path}", file=sys.stderr)
            return 1
        runs.append(parse_log(path, label=label))

    verify_versions(runs)

    for r in runs:
        print_single_table(r, show_http_workers=args.http_workers)

    if len(runs) >= 2:
        print_comparison_table(runs)

    if args.csv:
        import csv

        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "label",
                    "timestamp",
                    "elapsed_s",
                    "scanned",
                    "total",
                    "pct_complete",
                    "dt_s",
                    "d_scanned",
                    "interval_rate_q_per_s",
                    "cumulative_rate_q_per_s",
                ]
            )
            for r in runs:
                for s in r.samples:
                    writer.writerow(
                        [
                            r.label,
                            s.timestamp.isoformat() if s.timestamp else "",
                            f"{s.elapsed_s:.3f}",
                            s.scanned,
                            s.total,
                            f"{s.pct_complete:.4f}",
                            f"{s.dt_s:.3f}",
                            s.d_scanned,
                            f"{s.interval_rate:.4f}",
                            f"{s.cumulative_rate:.4f}",
                        ]
                    )
        print(f"\nExported CSV to {args.csv}")

    if not args.no_plot:
        plot_metrics(runs, args.output, show=args.show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
