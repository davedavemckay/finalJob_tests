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

    def http_time_bins(self, bin_s: float = 60.0) -> list[HttpTimeBin]:
        """Compute binned instantaneous HTTP throughput & latency from consecutive per-worker summaries."""
        if not self.http_summaries or bin_s <= 0:
            return []

        by_worker: dict[tuple[int, str], list[HttpSummarySample]] = {}
        for h in self.http_summaries:
            by_worker.setdefault((h.pid, h.proc), []).append(h)

        n_workers = len(by_worker)
        # Collect (bin_idx, dn, dt, sum_get_ms, sum_write_ms, d_fe, sum_fe_ms, d_be, sum_be_ms)
        bins: dict[int, list[tuple[int, float, float, float, int, float, int, float]]] = {}

        for series in by_worker.values():
            series_sorted = sorted(series, key=lambda x: x.up_s)
            prev: HttpSummarySample | None = None
            for curr in series_sorted:
                if prev is None:
                    dt = curr.up_s
                    dn = curr.as_local_n
                    sum_get = curr.as_local_n * curr.as_local_get_ms
                    sum_write = curr.as_local_n * curr.as_local_write_ms
                    d_fe = curr.fe_total
                    sum_fe = curr.fe_new * curr.fe_new_ms + curr.fe_reuse * curr.fe_reuse_ms
                    d_be = curr.be_total
                    sum_be = curr.be_new * curr.be_new_ms + curr.be_reuse * curr.be_reuse_ms
                    mid_t = curr.up_s / 2.0
                else:
                    dt = curr.up_s - prev.up_s
                    dn = curr.as_local_n - prev.as_local_n
                    sum_get = (curr.as_local_n * curr.as_local_get_ms) - (
                        prev.as_local_n * prev.as_local_get_ms
                    )
                    sum_write = (curr.as_local_n * curr.as_local_write_ms) - (
                        prev.as_local_n * prev.as_local_write_ms
                    )
                    d_fe = curr.fe_total - prev.fe_total
                    sum_fe = (
                        curr.fe_new * curr.fe_new_ms + curr.fe_reuse * curr.fe_reuse_ms
                    ) - (prev.fe_new * prev.fe_new_ms + prev.fe_reuse * prev.fe_reuse_ms)
                    d_be = curr.be_total - prev.be_total
                    sum_be = (
                        curr.be_new * curr.be_new_ms + curr.be_reuse * curr.be_reuse_ms
                    ) - (prev.be_new * prev.be_new_ms + prev.be_reuse * prev.be_reuse_ms)
                    mid_t = (prev.up_s + curr.up_s) / 2.0

                prev = curr
                if dt < 1.0 or dn < 0:
                    continue
                bidx = int(mid_t // bin_s)
                bins.setdefault(bidx, []).append(
                    (
                        dn,
                        dt,
                        max(0.0, sum_get),
                        max(0.0, sum_write),
                        max(0, d_fe),
                        max(0.0, sum_fe),
                        max(0, d_be),
                        max(0.0, sum_be),
                    )
                )

        out: list[HttpTimeBin] = []
        for bidx in sorted(bins):
            entries = bins[bidx]
            tot_dn = sum(e[0] for e in entries)
            tot_dt = sum(e[1] for e in entries)
            tot_get = sum(e[2] for e in entries)
            tot_write = sum(e[3] for e in entries)
            tot_fe = sum(e[4] for e in entries)
            tot_fe_ms = sum(e[5] for e in entries)
            tot_be = sum(e[6] for e in entries)
            tot_be_ms = sum(e[7] for e in entries)

            # Scale mean per-worker rate in this bin by total worker count
            agg_rate = (tot_dn / tot_dt) * n_workers if tot_dt > 0 else 0.0
            get_ms = (tot_get / tot_dn) if tot_dn > 0 else 0.0
            write_ms = (tot_write / tot_dn) if tot_dn > 0 else 0.0
            fe_ms = (tot_fe_ms / tot_fe) if tot_fe > 0 else 0.0
            be_ms = (tot_be_ms / tot_be) if tot_be > 0 else 0.0
            out.append(
                HttpTimeBin(
                    t_min=((bidx + 0.5) * bin_s) / 60.0,
                    agg_as_local_rate=agg_rate,
                    as_local_get_ms=get_ms,
                    as_local_write_ms=write_ms,
                    as_local_total_ms=get_ms + write_ms,
                    fe_ms=fe_ms,
                    be_ms=be_ms,
                    n_samples=len(entries),
                )
            )
        return out

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


@dataclass
class HttpTimeBin:
    t_min: float
    agg_as_local_rate: float
    as_local_get_ms: float
    as_local_write_ms: float
    as_local_total_ms: float
    fe_ms: float
    be_ms: float
    n_samples: int


@dataclass
class HttpGroupBin:
    t_min: float
    agg_rate_mean: float
    agg_rate_sd: float
    total_ms_mean: float
    total_ms_sd: float
    get_ms_mean: float
    write_ms_mean: float
    fe_ms_mean: float
    be_ms_mean: float


@dataclass
class StepMean:
    elapsed_s: float
    scanned_mean: float
    scanned_sd: float
    interval_rate_mean: float
    interval_rate_sd: float
    cum_rate_mean: float
    cum_rate_sd: float
    n_runs: int


@dataclass
class RunGroup:
    label: str
    runs: list[LogMetrics]

    @property
    def is_multi(self) -> bool:
        return len(self.runs) > 1

    @property
    def display_label(self) -> str:
        return f"{self.label} (n={len(self.runs)})" if self.is_multi else self.label

    def metric_stats(
        self, getter
    ) -> tuple[float | None, float | None, float | None, float | None, int]:
        """Return (mean, stdev, min, max, count) across non-None values in this group."""
        vals = [v for r in self.runs if (v := getter(r)) is not None]
        if not vals:
            return None, None, None, None, 0
        mean_v = statistics.mean(vals)
        sd_v = statistics.stdev(vals) if len(vals) >= 2 else None
        return mean_v, sd_v, min(vals), max(vals), len(vals)

    def min_steps(self) -> int:
        counts = [len(r.samples) for r in self.runs if r.samples]
        return min(counts) if counts else 0

    def max_steps(self) -> int:
        counts = [len(r.samples) for r in self.runs if r.samples]
        return max(counts) if counts else 0

    def step_series(self) -> list[StepMean]:
        """Per-step mean & stdev across all replicates that reached each step."""
        max_n = self.max_steps()
        out: list[StepMean] = []
        for i in range(max_n):
            at_i = [r.samples[i] for r in self.runs if len(r.samples) > i]
            if not at_i:
                continue
            elapsed = statistics.mean(s.elapsed_s for s in at_i)
            sc = [float(s.scanned) for s in at_i]
            ir = [s.interval_rate for s in at_i]
            cr = [s.cumulative_rate for s in at_i]
            out.append(
                StepMean(
                    elapsed_s=elapsed,
                    scanned_mean=statistics.mean(sc),
                    scanned_sd=statistics.stdev(sc) if len(sc) >= 2 else 0.0,
                    interval_rate_mean=statistics.mean(ir),
                    interval_rate_sd=statistics.stdev(ir) if len(ir) >= 2 else 0.0,
                    cum_rate_mean=statistics.mean(cr),
                    cum_rate_sd=statistics.stdev(cr) if len(cr) >= 2 else 0.0,
                    n_runs=len(at_i),
                )
            )
        return out

    def http_time_series(self, bin_s: float = 60.0) -> list[HttpGroupBin]:
        """Mean & stdev of binned HTTP metrics across replicates in this group."""
        by_t: dict[float, list[HttpTimeBin]] = {}
        for r in self.runs:
            for b in r.http_time_bins(bin_s=bin_s):
                by_t.setdefault(round(b.t_min, 4), []).append(b)
        out: list[HttpGroupBin] = []
        for t_min in sorted(by_t):
            bs = by_t[t_min]
            rates = [b.agg_as_local_rate for b in bs]
            tots = [b.as_local_total_ms for b in bs]
            out.append(
                HttpGroupBin(
                    t_min=t_min,
                    agg_rate_mean=statistics.mean(rates),
                    agg_rate_sd=statistics.stdev(rates) if len(rates) >= 2 else 0.0,
                    total_ms_mean=statistics.mean(tots),
                    total_ms_sd=statistics.stdev(tots) if len(tots) >= 2 else 0.0,
                    get_ms_mean=statistics.mean(b.as_local_get_ms for b in bs),
                    write_ms_mean=statistics.mean(b.as_local_write_ms for b in bs),
                    fe_ms_mean=statistics.mean(b.fe_ms for b in bs),
                    be_ms_mean=statistics.mean(b.be_ms for b in bs),
                )
            )
        return out

    def all_worker_rates(self) -> list[float]:
        """All per-worker as_local() rates across replicates in this group."""
        rates: list[float] = []
        for r in self.runs:
            agg = r.http_aggregate()
            if agg:
                rates.extend(agg.as_local_worker_rates)
        return rates

    def worker_boxplot_label(self) -> str:
        """Clear label distinguishing workers/run from total pooled samples across replicates."""
        per_run_counts = [
            len(agg.as_local_worker_rates)
            for r in self.runs
            if (agg := r.http_aggregate()) and agg.as_local_worker_rates
        ]
        total_samples = sum(per_run_counts)
        if not self.is_multi:
            return f"{self.label}\n({total_samples} workers)"
        if len(set(per_run_counts)) == 1:
            return f"{self.display_label}\n({per_run_counts[0]} workers/run, {total_samples} samples)"
        if per_run_counts:
            return (
                f"{self.display_label}\n"
                f"({min(per_run_counts)}-{max(per_run_counts)} workers/run, {total_samples} samples)"
            )
        return self.display_label


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


def verify_versions(groups: list[RunGroup]) -> None:
    """Print a non-fatal verification summary of lsst.resources versions across run groups."""
    print("\n=== lsst.resources Version Check ===")
    for g in groups:
        for r in g.runs:
            tag = f"{g.label}:{r.path.name}" if g.is_multi else g.label
            if not r.has_version_info:
                print(
                    f"  [{tag}] INFO: No lsst.resources / eups version lines found in log (continuing)."
                )
                continue

            print(f"  [{tag}]")
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
                re.search(r"(?:ticket|dm[-_]\d+)", g.label, re.IGNORECASE)
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
                    f"    status      : WARNING - label '{g.label}' looks like a ticket branch, "
                    f"but {detail} appears to be the shared stack install"
                )

        # Within-group consistency check for replicates
        if g.is_multi:
            rep_versions = {
                (r.resources_file, r.eups_version)
                for r in g.runs
                if r.has_version_info
            }
            if len(rep_versions) > 1:
                print(
                    f"  [WARNING] Replicates in group '{g.label}' do NOT all share the same "
                    f"lsst.resources setup!"
                )

    # Cross-group distinctness check (compare representative run with version info per group)
    group_reps = [
        (g.label, next((r for r in g.runs if r.has_version_info), None))
        for g in groups
    ]
    with_info = [(lbl, r) for lbl, r in group_reps if r is not None]
    if len(with_info) >= 2:
        first_lbl, first = with_info[0]
        for other_lbl, other in with_info[1:]:
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
                    f"  [WARNING] '{first_lbl}' and '{other_lbl}' report the SAME "
                    f"lsst.resources setup ({first.version_summary})!"
                )
            else:
                print(
                    f"  [OK] '{first_lbl}' and '{other_lbl}' are using distinct "
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


def print_group_spread(group: RunGroup) -> None:
    """Print replicate spread (Mean ± Stdev, Min, Max, CV%) for a multi-log group."""
    if not group.is_multi:
        return
    print(f"\n=== Replicate Spread: {group.label} (n={len(group.runs)}) ===")
    print(
        f"{'Metric':<32} | {'Mean':>11} | {'Stdev':>10} | {'CV%':>7} | {'Min':>11} | {'Max':>11}"
    )
    print("-" * 97)

    def row(name: str, getter, unit: str = "") -> None:
        mean_v, sd_v, min_v, max_v, n = group.metric_stats(getter)
        if mean_v is None:
            return
        cv_s = f"{100.0 * sd_v / mean_v:6.2f}%" if (sd_v is not None and mean_v != 0) else f"{'N/A':>7}"
        sd_s = f"{sd_v:7.2f} {unit:<2}" if sd_v is not None else f"{'N/A':>10}"
        print(
            f"{name:<32} | {mean_v:8.2f} {unit:<2} | {sd_s} | {cv_s} | "
            f"{min_v:8.2f} {unit:<2} | {max_v:8.2f} {unit:<2}"
        )

    row("Read predicted graph", lambda r: r.read_graph_duration_s, "s")
    row("Analyze -> defensive ingest", lambda r: r.analyze_to_defensive_s, "s")
    row("Logged pre-scan [after]", lambda r: r.defensive_ingest_after_s, "s")
    row("Quanta @ 600s", lambda r: float(r.samples[0].scanned) if r.samples else None, "q")
    row(
        "Active scan rate @ 600s",
        lambda r: r.samples[0].scanned / (r.samples[0].elapsed_s - (r.defensive_ingest_after_s or 0.0))
        if r.samples and (r.samples[0].elapsed_s - (r.defensive_ingest_after_s or 0.0)) > 0
        else None,
        "q/s",
    )
    row("Overall rate", lambda r: r.overall_rate if r.samples else None, "q/s")
    row("Active scan rate (post-setup)", lambda r: r.active_scan_rate if r.samples else None, "q/s")
    row("Steady-state rate (t>600s)", lambda r: r.steady_state_rate if r.samples else None, "q/s")
    row("Projected total time", lambda r: r.estimated_total_hours if r.samples else None, "h")

    if any(r.http_summaries for r in group.runs):
        row("FE avg latency", lambda r: a.fe_avg_ms if (a := r.http_aggregate()) else None, "ms")
        row("BE avg latency", lambda r: a.be_avg_ms if (a := r.http_aggregate()) else None, "ms")
        row("as_local() total avg", lambda r: a.as_local_total_ms if (a := r.http_aggregate()) else None, "ms")
        row("as_local() get avg", lambda r: a.as_local_get_ms if (a := r.http_aggregate()) else None, "ms")
        row("as_local() write avg", lambda r: a.as_local_write_ms if (a := r.http_aggregate()) else None, "ms")
        row("as_local() agg throughput", lambda r: a.as_local_agg_rate if (a := r.http_aggregate()) else None, "/s")


def print_comparison_table(groups: list[RunGroup]) -> None:
    if len(groups) < 2:
        return
    base = groups[0]
    any_multi = any(g.is_multi for g in groups)
    col_w = 22 if any_multi else 16
    print("\n=== Summary Comparison ===")
    header = (
        f"{'Metric':<34} | "
        + " | ".join(f"{g.display_label:>{col_w}}" for g in groups)
        + (
            f" | {'Diff (' + groups[1].label + ' vs ' + base.label + ')':>24}"
            if len(groups) == 2
            else ""
        )
    )
    print(header)
    print("-" * len(header))

    def fmt_row(name: str, getter, unit: str = "") -> None:
        stats_list = [g.metric_stats(getter) for g in groups]
        cells: list[str] = []
        for g, (mean_v, sd_v, _, _, _) in zip(groups, stats_list):
            if mean_v is None:
                cells.append(f"{'N/A':>{col_w}}")
            elif g.is_multi and sd_v is not None:
                val_s = f"{mean_v:.2f}±{sd_v:.2f}"
                cells.append(f"{val_s:>{col_w - 3}} {unit:<2}" if unit else f"{val_s:>{col_w}}")
            else:
                cells.append(f"{mean_v:{col_w - 3}.2f} {unit:<2}" if unit else f"{mean_v:{col_w}.2f}")

        diff_str = ""
        m0, m1 = stats_list[0][0], stats_list[1][0] if len(groups) >= 2 else None
        if len(groups) == 2 and m0 is not None and m1 is not None:
            delta = m1 - m0
            if m0 != 0:
                pct = 100.0 * delta / m0
                diff_str = f" | {delta:+10.2f} {unit:<2} ({pct:+6.2f}%)"
            else:
                diff_str = f" | {delta:+10.2f} {unit:<2} (   N/A)"
        print(f"{name:<34} | " + " | ".join(cells) + diff_str)

    fmt_row("Read predicted graph", lambda r: r.read_graph_duration_s, "s")
    fmt_row("Analyze -> defensive ingest", lambda r: r.analyze_to_defensive_s, "s")
    fmt_row("Logged pre-scan [after]", lambda r: r.defensive_ingest_after_s, "s")
    fmt_row("Quanta @ 600s", lambda r: float(r.samples[0].scanned) if r.samples else None, "q")
    fmt_row(
        "Active scan rate @ 600s",
        lambda r: r.samples[0].scanned / (r.samples[0].elapsed_s - (r.defensive_ingest_after_s or 0.0))
        if r.samples and (r.samples[0].elapsed_s - (r.defensive_ingest_after_s or 0.0)) > 0
        else None,
        "q/s",
    )

    # Compare at the last common step when logs have different lengths
    all_step_counts = [len(r.samples) for g in groups for r in g.runs if r.samples]
    common_n = min(all_step_counts) if all_step_counts else 0
    if common_n > 0 and len(set(all_step_counts)) > 1:
        common_t = next(r.samples[common_n - 1].elapsed_s for g in groups for r in g.runs if len(r.samples) >= common_n)
        fmt_row(
            f"Quanta @ {common_t:.0f}s (step {common_n})",
            lambda r: float(r.samples[common_n - 1].scanned) if len(r.samples) >= common_n else None,
            "q",
        )
        fmt_row(
            f"Steady rate (600s->{common_t:.0f}s)",
            lambda r: (r.samples[common_n - 1].scanned - r.samples[0].scanned)
            / (r.samples[common_n - 1].elapsed_s - r.samples[0].elapsed_s)
            if common_n >= 2
            and len(r.samples) >= common_n
            and (r.samples[common_n - 1].elapsed_s - r.samples[0].elapsed_s) > 0
            else None,
            "q/s",
        )

    fmt_row("Final quanta scanned", lambda r: float(r.final_scanned) if r.samples else None, "q")
    fmt_row("Overall rate", lambda r: r.overall_rate if r.samples else None, "q/s")
    fmt_row("Active scan rate (post-setup)", lambda r: r.active_scan_rate if r.samples else None, "q/s")
    fmt_row("Steady-state rate (t>600s)", lambda r: r.steady_state_rate if r.samples else None, "q/s")
    fmt_row("Projected total time", lambda r: r.estimated_total_hours if r.samples else None, "h")

    if len(groups) == 2:
        sa_list = groups[0].step_series()
        sb_list = groups[1].step_series()
        n = min(len(sa_list), len(sb_list))
        if n > 0:
            a_lbl, b_lbl = groups[0].label, groups[1].label
            print(f"\n=== Step-by-Step Comparison ({b_lbl} vs {a_lbl}) ===")
            print(
                f"{'Step':>4} | {'Elapsed (s)':>11} | "
                f"{a_lbl + ' Quanta':>16} | {b_lbl + ' Quanta':>16} | {'Delta Quanta':>12} | "
                f"{a_lbl + ' q/s':>14} | {b_lbl + ' q/s':>16} | {'Delta q/s':>10}"
            )
            print("-" * 115)
            for i in range(n):
                sa, sb = sa_list[i], sb_list[i]
                dq = sb.scanned_mean - sa.scanned_mean
                dr = sb.interval_rate_mean - sa.interval_rate_mean
                print(
                    f"{i + 1:4d} | {sa.elapsed_s:11.1f} | "
                    f"{round(sa.scanned_mean):16,d} | {round(sb.scanned_mean):16,d} | {round(dq):+12,d} | "
                    f"{sa.interval_rate_mean:14.2f} | {sb.interval_rate_mean:16.2f} | {dr:+10.2f}"
                )

    if any(r.http_summaries for g in groups for r in g.runs):
        print("\n=== HTTP-SUMMARY Comparison ===")
        print(header)
        print("-" * len(header))
        fmt_row("HTTP workers", lambda r: float(a.worker_count) if (a := r.http_aggregate()) else None, "")
        fmt_row("Total HTTP requests", lambda r: float(a.total_reqs) if (a := r.http_aggregate()) else None, "")
        fmt_row("FE reuse rate", lambda r: a.fe_reuse_pct if (a := r.http_aggregate()) else None, "%")
        fmt_row("FE avg latency", lambda r: a.fe_avg_ms if (a := r.http_aggregate()) else None, "ms")
        fmt_row("BE reuse rate", lambda r: a.be_reuse_pct if (a := r.http_aggregate()) else None, "%")
        fmt_row("BE avg latency", lambda r: a.be_avg_ms if (a := r.http_aggregate()) else None, "ms")
        fmt_row("BE hosts (avg/worker)", lambda r: a.be_hosts_mean if (a := r.http_aggregate()) else None, "")
        fmt_row("as_local() count", lambda r: float(a.as_local_n_total) if (a := r.http_aggregate()) else None, "")
        fmt_row("as_local() total avg", lambda r: a.as_local_total_ms if (a := r.http_aggregate()) else None, "ms")
        fmt_row("as_local() get avg", lambda r: a.as_local_get_ms if (a := r.http_aggregate()) else None, "ms")
        fmt_row("as_local() write avg", lambda r: a.as_local_write_ms if (a := r.http_aggregate()) else None, "ms")
        fmt_row("as_local() write net avg", lambda r: a.as_local_net_ms if (a := r.http_aggregate()) else None, "ms")
        fmt_row("as_local() write disk avg", lambda r: a.as_local_disk_ms if (a := r.http_aggregate()) else None, "ms")
        fmt_row("as_local() agg throughput", lambda r: a.as_local_agg_rate if (a := r.http_aggregate()) else None, "/s")
        fmt_row(
            "as_local() median worker rate",
            lambda r: statistics.median(a.as_local_worker_rates)
            if (a := r.http_aggregate()) and a.as_local_worker_rates
            else None,
            "/s",
        )


def plot_metrics(
    groups: list[RunGroup],
    output_path: Path,
    show: bool = False,
    max_minutes: float | None = 30.0,
) -> None:
    if max_minutes is not None and max_minutes <= 0:
        max_minutes = None
    max_s = (max_minutes * 60.0 + 5.0) if max_minutes is not None else None

    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MultipleLocator
    except ImportError:
        svg_path = output_path.with_suffix(".svg")
        _write_svg_fallback(groups, svg_path, max_minutes=max_minutes)
        print(
            f"\n[NOTE] matplotlib is not installed in {sys.executable}; "
            f"wrote standalone SVG plot to {svg_path}"
        )
        return

    def keep_step(elapsed_s: float) -> bool:
        return max_s is None or elapsed_s <= max_s

    has_comparison = len(groups) >= 2
    nrows = 3 if has_comparison else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 3.6 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    # Panel 1: Cumulative quanta scanned vs time
    ax0 = axes[0]
    for g in groups:
        series = [s for s in g.step_series() if keep_step(s.elapsed_s)]
        t_min = [s.elapsed_s / 60.0 for s in series]
        q_k = [s.scanned_mean / 1e3 for s in series]
        mean_rate, sd_rate, _, _, _ = g.metric_stats(
            lambda r: r.overall_rate if r.samples else None
        )
        rate_str = (
            f"{mean_rate:.2f}±{sd_rate:.2f} q/s"
            if (g.is_multi and mean_rate is not None and sd_rate is not None)
            else f"{(mean_rate or 0.0):.2f} q/s"
        )
        (line,) = ax0.plot(
            t_min,
            q_k,
            marker="o",
            markersize=3.5,
            linewidth=2.0,
            label=f"{g.display_label} ({rate_str})",
        )
        color = line.get_color()
        if g.is_multi:
            for r in g.runs:
                rs = [s for s in r.samples if keep_step(s.elapsed_s)]
                rt = [s.elapsed_s / 60.0 for s in rs]
                rq = [s.scanned / 1e3 for s in rs]
                ax0.plot(rt, rq, linewidth=0.9, alpha=0.35, color=color)
            lo = [(s.scanned_mean - s.scanned_sd) / 1e3 for s in series]
            hi = [(s.scanned_mean + s.scanned_sd) / 1e3 for s in series]
            ax0.fill_between(t_min, lo, hi, color=color, alpha=0.15)
    ax0.set_ylabel("Scanned Quanta (thousands)")
    title_suffix = f" (first {max_minutes:.0f} min)" if max_minutes is not None else ""
    ax0.set_title(f"QuantumGraph Aggregator Scanning Progress & Rate{title_suffix}")
    ax0.grid(True, linestyle="--", alpha=0.5)
    ax0.legend(loc="upper left")

    # Panel 2: Interval & Cumulative scanning rate vs time
    ax1 = axes[1]
    for g in groups:
        series = [s for s in g.step_series() if keep_step(s.elapsed_s)]
        t_min = [s.elapsed_s / 60.0 for s in series]
        int_rate = [s.interval_rate_mean for s in series]
        cum_rate = [s.cum_rate_mean for s in series]
        (line,) = ax1.plot(
            t_min,
            int_rate,
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            label=f"{g.display_label} interval rate",
        )
        color = line.get_color()
        ax1.plot(
            t_min,
            cum_rate,
            linestyle="--",
            linewidth=1.2,
            color=color,
            alpha=0.8,
            label=f"{g.display_label} cumulative rate",
        )
        if g.is_multi:
            for r in g.runs:
                rs = [s for s in r.samples if keep_step(s.elapsed_s)]
                rt = [s.elapsed_s / 60.0 for s in rs]
                ri = [s.interval_rate for s in rs]
                ax1.plot(rt, ri, linewidth=0.8, alpha=0.30, color=color)
            lo = [s.interval_rate_mean - s.interval_rate_sd for s in series]
            hi = [s.interval_rate_mean + s.interval_rate_sd for s in series]
            ax1.fill_between(t_min, lo, hi, color=color, alpha=0.15)
    ax1.set_ylabel("Scan Rate (quanta / s)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", ncol=2, fontsize="small")

    # Panel 3: Delta between group means (if 2+ groups provided)
    if has_comparison:
        ax2 = axes[2]
        base = groups[0]
        base_series = [s for s in base.step_series() if keep_step(s.elapsed_s)]
        for g in groups[1:]:
            g_series = [s for s in g.step_series() if keep_step(s.elapsed_s)]
            n = min(len(base_series), len(g_series))
            t_min = [base_series[i].elapsed_s / 60.0 for i in range(n)]
            dq = [g_series[i].scanned_mean - base_series[i].scanned_mean for i in range(n)]
            ax2.plot(
                t_min,
                dq,
                marker="s",
                markersize=3.5,
                linewidth=1.6,
                color="tab:green" if len(groups) == 2 else None,
                label=f"{g.label} - {base.label} (quanta lead)",
            )
        ax2.axhline(0, color="black", linewidth=0.8, linestyle=":")
        ax2.set_ylabel("Quanta Difference")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="best")

    axes[-1].set_xlabel("Elapsed Time (minutes)")
    if max_minutes is not None:
        axes[-1].set_xlim(0.0, max_minutes)
        if max_minutes <= 90.0:
            axes[-1].xaxis.set_major_locator(MultipleLocator(5.0))
    else:
        axes[-1].set_xlim(left=0.0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"\nSaved plot to {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def _write_svg_fallback(
    groups: list[RunGroup], svg_path: Path, max_minutes: float | None = 30.0
) -> None:
    """Generate a clean multi-panel SVG plot using only the Python standard library."""
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    max_s = (max_minutes * 60.0 + 5.0) if max_minutes is not None else None

    def keep_step(elapsed_s: float) -> bool:
        return max_s is None or elapsed_s <= max_s

    has_comp = len(groups) >= 2
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

    all_runs = [r for g in groups for r in g.runs]
    all_t = [
        s.elapsed_s / 60.0 for r in all_runs for s in r.samples if keep_step(s.elapsed_s)
    ] or [0.0, 30.0]
    t_min = 0.0
    t_max = max_minutes if max_minutes is not None else max(all_t) * 1.02

    def sx(t: float) -> float:
        return left_margin + (t - t_min) / (t_max - t_min or 1.0) * plot_w

    title_suffix = f" (first {max_minutes:.0f} min)" if max_minutes is not None else ""
    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" ',
        f'viewBox="0 0 {width} {height}" font-family="sans-serif" font-size="12">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="16" font-weight="bold" fill="#222">',
        f"QuantumGraph Aggregator Scanning Progress &amp; Rate{title_suffix}</text>",
    ]

    if max_minutes is not None and max_minutes <= 90.0:
        xticks = [float(m) for m in range(0, int(max_minutes) + 1, 5)]
    else:
        xticks = [t_min + (i / 6.0) * (t_max - t_min) for i in range(7)]

    def draw_panel(
        row: int,
        y_min: float,
        y_max: float,
        ylabel: str,
        series: list[tuple[str | None, list[tuple[float, float]], str, str, float]],
    ) -> None:
        y0 = top_margin + row * (panel_h + gap)
        lines.append(
            f'<rect x="{left_margin}" y="{y0}" width="{plot_w}" height="{panel_h}" '
            f'fill="#fafafa" stroke="#cccccc" stroke-width="1"/>'
        )
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
        for t_val in xticks:
            px = sx(t_val)
            lines.append(
                f'<line x1="{px:.1f}" y1="{y0}" x2="{px:.1f}" y2="{y0 + panel_h}" '
                f'stroke="#e5e5e5" stroke-dasharray="3,3"/>'
            )
            if row == nrows - 1:
                lines.append(
                    f'<text x="{px:.1f}" y="{y0 + panel_h + 18}" text-anchor="middle" fill="#555" font-size="11">'
                    f"{t_val:.0f}m</text>"
                )
        cy = y0 + panel_h / 2
        lines.append(
            f'<text x="20" y="{cy:.1f}" text-anchor="middle" fill="#222" font-size="12" '
            f'transform="rotate(-90 20 {cy:.1f})">{ylabel}</text>'
        )

        def sy(v: float) -> float:
            return y0 + panel_h - (v - y_min) / (y_max - y_min or 1.0) * panel_h

        legend_x = left_margin + 14
        legend_y = y0 + 20
        for name, pts, color, dash, opacity in series:
            if not pts:
                continue
            poly = " ".join(f"{sx(t):.1f},{sy(v):.1f}" for t, v in pts)
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            sw = "1.0" if opacity < 0.9 else "2.0"
            lines.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="{sw}" '
                f'stroke-opacity="{opacity:.2f}"{dash_attr} points="{poly}"/>'
            )
            if not dash and opacity >= 0.9:
                for t, v in pts:
                    lines.append(
                        f'<circle cx="{sx(t):.1f}" cy="{sy(v):.1f}" r="2.8" fill="{color}"/>'
                    )
            if name:
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
        max(
            (s.scanned / 1e3 for r in all_runs for s in r.samples if keep_step(s.elapsed_s)),
            default=1.0,
        )
        * 1.08
    )
    p1_series = []
    for idx, g in enumerate(groups):
        c = colors[idx % len(colors)]
        if g.is_multi:
            for r in g.runs:
                rs = [s for s in r.samples if keep_step(s.elapsed_s)]
                p1_series.append(
                    (None, [(s.elapsed_s / 60.0, s.scanned / 1e3) for s in rs], c, "", 0.35)
                )
        mean_r, sd_r, _, _, _ = g.metric_stats(lambda r: r.overall_rate if r.samples else None)
        rate_s = (
            f"{mean_r:.2f}±{sd_r:.2f} q/s"
            if (g.is_multi and mean_r is not None and sd_r is not None)
            else f"{(mean_r or 0.0):.2f} q/s"
        )
        g_ser = [s for s in g.step_series() if keep_step(s.elapsed_s)]
        p1_series.append(
            (
                f"{g.display_label} ({rate_s})",
                [(s.elapsed_s / 60.0, s.scanned_mean / 1e3) for s in g_ser],
                c,
                "",
                1.0,
            )
        )
    draw_panel(0, 0.0, max_q_k, "Scanned Quanta (k)", p1_series)

    # Panel 2: Interval & Cumulative Rate
    all_rates = [
        s.interval_rate for r in all_runs for s in r.samples if keep_step(s.elapsed_s)
    ] + [
        s.cumulative_rate for r in all_runs for s in r.samples if keep_step(s.elapsed_s)
    ]
    r_min = max(0.0, min(all_rates, default=0.0) - 1.5)
    r_max = max(all_rates, default=1.0) + 1.5
    p2_series = []
    for idx, g in enumerate(groups):
        c = colors[idx % len(colors)]
        if g.is_multi:
            for r in g.runs:
                rs = [s for s in r.samples if keep_step(s.elapsed_s)]
                p2_series.append(
                    (None, [(s.elapsed_s / 60.0, s.interval_rate) for s in rs], c, "", 0.30)
                )
        series = [s for s in g.step_series() if keep_step(s.elapsed_s)]
        p2_series.append(
            (
                f"{g.display_label} interval rate",
                [(s.elapsed_s / 60.0, s.interval_rate_mean) for s in series],
                c,
                "",
                1.0,
            )
        )
        p2_series.append(
            (
                f"{g.display_label} cumulative rate",
                [(s.elapsed_s / 60.0, s.cum_rate_mean) for s in series],
                c,
                "5,4",
                1.0,
            )
        )
    draw_panel(1, r_min, r_max, "Rate (quanta / s)", p2_series)

    # Panel 3: Quanta Lead
    if has_comp:
        base = groups[0]
        base_s = [s for s in base.step_series() if keep_step(s.elapsed_s)]
        p3_series = []
        all_dq = [0.0]
        for idx, g in enumerate(groups[1:], start=1):
            g_s = [s for s in g.step_series() if keep_step(s.elapsed_s)]
            n = min(len(base_s), len(g_s))
            pts = [
                (
                    base_s[i].elapsed_s / 60.0,
                    float(g_s[i].scanned_mean - base_s[i].scanned_mean),
                )
                for i in range(n)
            ]
            all_dq.extend(v for _, v in pts)
            p3_series.append(
                (
                    f"{g.label} - {base.label} (quanta)",
                    pts,
                    colors[(idx + 1) % len(colors)],
                    "",
                    1.0,
                )
            )
        dq_min = min(all_dq) - 50.0
        dq_max = max(all_dq) + 50.0
        draw_panel(2, dq_min, dq_max, "Delta Quanta", p3_series)

    lines.append(
        f'<text x="{left_margin + plot_w / 2}" y="{height - 15}" text-anchor="middle" '
        f'fill="#222" font-size="13">Elapsed Time (minutes)</text>'
    )
    lines.append("</svg>")
    svg_path.write_text("\n".join(lines), encoding="utf-8")


def plot_http_metrics(
    groups: list[RunGroup],
    output_path: Path,
    bin_s: float = 60.0,
    show: bool = False,
) -> None:
    """Plot high-resolution HTTP-SUMMARY time-series and per-worker distributions."""
    http_groups = [g for g in groups if any(r.http_summaries for r in g.runs)]
    if not http_groups:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        svg_path = output_path.with_suffix(".svg")
        _write_http_svg_fallback(http_groups, svg_path, bin_s=bin_s)
        print(
            f"[NOTE] matplotlib is not installed in {sys.executable}; "
            f"wrote standalone HTTP-SUMMARY SVG plot to {svg_path}"
        )
        return

    fig, axes = plt.subplots(4, 1, figsize=(11, 13.5))
    ax0, ax1, ax2, ax3 = axes

    # Panel 1: Binned aggregate as_local() throughput over time
    for g in http_groups:
        series = g.http_time_series(bin_s=bin_s)
        if not series:
            continue
        t_m = [b.t_min for b in series]
        rates = [b.agg_rate_mean for b in series]
        (line,) = ax0.plot(
            t_m,
            rates,
            marker="o",
            markersize=3.0,
            linewidth=2.0,
            label=g.display_label,
        )
        color = line.get_color()
        if g.is_multi:
            for r in g.runs:
                rb = r.http_time_bins(bin_s=bin_s)
                ax0.plot(
                    [b.t_min for b in rb],
                    [b.agg_as_local_rate for b in rb],
                    linewidth=0.8,
                    alpha=0.35,
                    color=color,
                )
            lo = [b.agg_rate_mean - b.agg_rate_sd for b in series]
            hi = [b.agg_rate_mean + b.agg_rate_sd for b in series]
            ax0.fill_between(t_m, lo, hi, color=color, alpha=0.15)
    ax0.set_title(f"HTTP-SUMMARY High-Resolution Diagnostics ({bin_s:.0f}s bins)")
    ax0.set_ylabel("Aggregate as_local() (files / s)")
    ax0.set_xlabel("Worker Uptime (minutes)")
    ax0.grid(True, linestyle="--", alpha=0.5)
    ax0.legend(loc="best")

    # Panel 2: Binned as_local() latency breakdown (total, get, write)
    for g in http_groups:
        series = g.http_time_series(bin_s=bin_s)
        if not series:
            continue
        t_m = [b.t_min for b in series]
        (line,) = ax1.plot(
            t_m,
            [b.total_ms_mean for b in series],
            linewidth=2.0,
            label=f"{g.display_label} total",
        )
        color = line.get_color()
        ax1.plot(
            t_m,
            [b.get_ms_mean for b in series],
            linestyle="--",
            linewidth=1.3,
            color=color,
            alpha=0.85,
            label=f"{g.display_label} get",
        )
        ax1.plot(
            t_m,
            [b.write_ms_mean for b in series],
            linestyle=":",
            linewidth=1.3,
            color=color,
            alpha=0.85,
            label=f"{g.display_label} write",
        )
        if g.is_multi:
            lo = [b.total_ms_mean - b.total_ms_sd for b in series]
            hi = [b.total_ms_mean + b.total_ms_sd for b in series]
            ax1.fill_between(t_m, lo, hi, color=color, alpha=0.12)
    ax1.set_ylabel("as_local() Latency (ms)")
    ax1.set_xlabel("Worker Uptime (minutes)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="best", ncol=3, fontsize="small")

    # Panel 3: Binned FE & BE latency over time
    for g in http_groups:
        series = g.http_time_series(bin_s=bin_s)
        if not series:
            continue
        t_m = [b.t_min for b in series]
        fe_reuse, _, _, _, _ = g.metric_stats(
            lambda r: a.fe_reuse_pct if (a := r.http_aggregate()) else None
        )
        be_reuse, _, _, _, _ = g.metric_stats(
            lambda r: a.be_reuse_pct if (a := r.http_aggregate()) else None
        )
        (line,) = ax2.plot(
            t_m,
            [b.be_ms_mean for b in series],
            linewidth=1.8,
            label=f"{g.display_label} BE (reuse {(be_reuse or 0.0):.0f}%)",
        )
        color = line.get_color()
        ax2.plot(
            t_m,
            [b.fe_ms_mean for b in series],
            linestyle="--",
            linewidth=1.4,
            color=color,
            alpha=0.85,
            label=f"{g.display_label} FE (reuse {(fe_reuse or 0.0):.0f}%)",
        )
    ax2.set_ylabel("FE / BE Latency (ms)")
    ax2.set_xlabel("Worker Uptime (minutes)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="best", ncol=2, fontsize="small")

    # Panel 4: Per-worker as_local() rate distribution (boxplot)
    box_data = [g.all_worker_rates() for g in http_groups]
    box_labels = [g.worker_boxplot_label() for g in http_groups]
    valid = [(lbl, d) for lbl, d in zip(box_labels, box_data) if d]
    if valid:
        ax3.boxplot(
            [d for _, d in valid],
            tick_labels=[lbl for lbl, _ in valid],
            showmeans=True,
            patch_artist=True,
            boxprops={"facecolor": "#d9eaf7", "alpha": 0.7},
        )
    ax3.set_ylabel("Per-Worker as_local() (files / s)")
    ax3.set_title("Per-Worker Throughput Distribution (All Workers Across Replicates)")
    ax3.grid(True, axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved HTTP-SUMMARY plot to {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def _write_http_svg_fallback(
    groups: list[RunGroup], svg_path: Path, bin_s: float = 60.0
) -> None:
    """Generate a standalone 4-panel SVG for HTTP-SUMMARY diagnostics without matplotlib."""
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    width = 900
    panel_h = 210
    gap = 48
    top_margin = 50
    bottom_margin = 55
    left_margin = 80
    right_margin = 30
    nrows = 4
    height = top_margin + nrows * panel_h + (nrows - 1) * gap + bottom_margin
    plot_w = width - left_margin - right_margin

    group_series = [(g, g.http_time_series(bin_s=bin_s)) for g in groups]
    all_t = [b.t_min for _, s in group_series for b in s] or [0.0, 1.0]
    t_min, t_max = 0.0, max(all_t) * 1.03

    def sx(t: float) -> float:
        return left_margin + (t - t_min) / (t_max - t_min or 1.0) * plot_w

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="sans-serif" font-size="12">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="16" font-weight="bold" fill="#222">'
        f"HTTP-SUMMARY High-Resolution Diagnostics ({bin_s:.0f}s bins)</text>",
    ]

    def draw_ts_panel(
        row: int,
        y_min: float,
        y_max: float,
        ylabel: str,
        series: list[tuple[str | None, list[tuple[float, float]], str, str, float]],
    ) -> None:
        y0 = top_margin + row * (panel_h + gap)
        lines.append(
            f'<rect x="{left_margin}" y="{y0}" width="{plot_w}" height="{panel_h}" '
            f'fill="#fafafa" stroke="#cccccc" stroke-width="1"/>'
        )
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
        for i in range(7):
            frac = i / 6.0
            tv = t_min + frac * (t_max - t_min)
            px = sx(tv)
            lines.append(
                f'<line x1="{px:.1f}" y1="{y0}" x2="{px:.1f}" y2="{y0 + panel_h}" '
                f'stroke="#e5e5e5" stroke-dasharray="3,3"/>'
            )
            lines.append(
                f'<text x="{px:.1f}" y="{y0 + panel_h + 15}" text-anchor="middle" fill="#555" font-size="10">'
                f"{tv:.0f}m</text>"
            )
        cy = y0 + panel_h / 2
        lines.append(
            f'<text x="20" y="{cy:.1f}" text-anchor="middle" fill="#222" font-size="12" '
            f'transform="rotate(-90 20 {cy:.1f})">{ylabel}</text>'
        )

        def sy(v: float) -> float:
            return y0 + panel_h - (v - y_min) / (y_max - y_min or 1.0) * panel_h

        legend_x = left_margin + 12
        legend_y = y0 + 18
        for name, pts, color, dash, opacity in series:
            if not pts:
                continue
            poly = " ".join(f"{sx(t):.1f},{sy(v):.1f}" for t, v in pts)
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            sw = "1.0" if opacity < 0.9 else "1.8"
            lines.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="{sw}" '
                f'stroke-opacity="{opacity:.2f}"{dash_attr} points="{poly}"/>'
            )
            if name:
                lines.append(
                    f'<line x1="{legend_x}" y1="{legend_y - 4}" x2="{legend_x + 18}" y2="{legend_y - 4}" '
                    f'stroke="{color}" stroke-width="2"{dash_attr}/>'
                )
                lines.append(
                    f'<text x="{legend_x + 24}" y="{legend_y}" fill="#222" font-size="10">{name}</text>'
                )
                legend_y += 15

    # Panel 1: Aggregate as_local() throughput
    all_rates = [b.agg_rate_mean for _, s in group_series for b in s] or [1.0]
    p1 = []
    for idx, (g, s) in enumerate(group_series):
        c = colors[idx % len(colors)]
        if g.is_multi:
            for r in g.runs:
                rb = r.http_time_bins(bin_s=bin_s)
                p1.append((None, [(b.t_min, b.agg_as_local_rate) for b in rb], c, "", 0.30))
        p1.append((g.display_label, [(b.t_min, b.agg_rate_mean) for b in s], c, "", 1.0))
    draw_ts_panel(0, 0.0, max(all_rates) * 1.12, "Agg as_local (files/s)", p1)

    # Panel 2: as_local() total / get / write latency
    all_lat = [b.total_ms_mean for _, s in group_series for b in s] or [1.0]
    p2 = []
    for idx, (g, s) in enumerate(group_series):
        c = colors[idx % len(colors)]
        p2.append((f"{g.display_label} total", [(b.t_min, b.total_ms_mean) for b in s], c, "", 1.0))
        p2.append((f"{g.display_label} get", [(b.t_min, b.get_ms_mean) for b in s], c, "5,3", 0.9))
        p2.append((f"{g.display_label} write", [(b.t_min, b.write_ms_mean) for b in s], c, "2,2", 0.9))
    draw_ts_panel(1, 0.0, max(all_lat) * 1.15, "as_local Latency (ms)", p2)

    # Panel 3: FE & BE latency
    all_fb = [v for _, s in group_series for b in s for v in (b.fe_ms_mean, b.be_ms_mean)] or [1.0]
    p3 = []
    for idx, (g, s) in enumerate(group_series):
        c = colors[idx % len(colors)]
        p3.append((f"{g.display_label} BE", [(b.t_min, b.be_ms_mean) for b in s], c, "", 1.0))
        p3.append((f"{g.display_label} FE", [(b.t_min, b.fe_ms_mean) for b in s], c, "5,3", 0.9))
    draw_ts_panel(2, 0.0, max(all_fb) * 1.15, "FE / BE Latency (ms)", p3)

    # Panel 4: Per-worker rate distribution (dot + median/IQR bar per group)
    y0 = top_margin + 3 * (panel_h + gap)
    lines.append(
        f'<rect x="{left_margin}" y="{y0}" width="{plot_w}" height="{panel_h}" '
        f'fill="#fafafa" stroke="#cccccc" stroke-width="1"/>'
    )
    all_wr = [v for g in groups for v in g.all_worker_rates()] or [1.0]
    w_max = max(all_wr) * 1.15
    for i in range(5):
        frac = i / 4.0
        val = frac * w_max
        py = y0 + panel_h - frac * panel_h
        lines.append(
            f'<line x1="{left_margin}" y1="{py:.1f}" x2="{left_margin + plot_w}" y2="{py:.1f}" '
            f'stroke="#e5e5e5" stroke-dasharray="3,3"/>'
        )
        lines.append(
            f'<text x="{left_margin - 8}" y="{py + 4:.1f}" text-anchor="end" fill="#555" font-size="11">'
            f"{val:.2f}</text>"
        )
    cy = y0 + panel_h / 2
    lines.append(
        f'<text x="20" y="{cy:.1f}" text-anchor="middle" fill="#222" font-size="12" '
        f'transform="rotate(-90 20 {cy:.1f})">Worker Rate (files/s)</text>'
    )
    n_g = len(groups)
    for idx, g in enumerate(groups):
        wr = sorted(g.all_worker_rates())
        cx = left_margin + (idx + 0.5) * (plot_w / max(n_g, 1))
        c = colors[idx % len(colors)]
        if wr:
            def wy(v: float) -> float:
                return y0 + panel_h - (v / (w_max or 1.0)) * panel_h

            med = statistics.median(wr)
            q1 = wr[len(wr) // 4]
            q3 = wr[(3 * len(wr)) // 4]
            # Whisker min..max
            lines.append(
                f'<line x1="{cx:.1f}" y1="{wy(wr[0]):.1f}" x2="{cx:.1f}" y2="{wy(wr[-1]):.1f}" '
                f'stroke="{c}" stroke-width="1.5"/>'
            )
            # IQR box
            box_w = min(50.0, (plot_w / max(n_g, 1)) * 0.35)
            lines.append(
                f'<rect x="{cx - box_w / 2:.1f}" y="{wy(q3):.1f}" width="{box_w:.1f}" '
                f'height="{max(2.0, wy(q1) - wy(q3)):.1f}" fill="{c}" fill-opacity="0.25" '
                f'stroke="{c}" stroke-width="1.5"/>'
            )
            # Median line
            lines.append(
                f'<line x1="{cx - box_w / 2:.1f}" y1="{wy(med):.1f}" x2="{cx + box_w / 2:.1f}" y2="{wy(med):.1f}" '
                f'stroke="{c}" stroke-width="2.5"/>'
            )
        svg_lbl = g.worker_boxplot_label().replace("\n", " ")
        lines.append(
            f'<text x="{cx:.1f}" y="{y0 + panel_h + 18}" text-anchor="middle" fill="#222" font-size="10">'
            f"{svg_lbl}</text>"
        )

    lines.append("</svg>")
    svg_path.write_text("\n".join(lines), encoding="utf-8")


def parse_input_spec(spec: str) -> tuple[str, list[Path]]:
    """Parse 'label=path1,path2,...', 'path1,path2,...', or 'path'."""
    if "=" in spec and not Path(spec).exists():
        label, paths_str = spec.split("=", 1)
        paths = [Path(p.strip()) for p in paths_str.split(",") if p.strip()]
        return label, paths
    if "," in spec and not Path(spec).exists():
        paths = [Path(p.strip()) for p in spec.split(",") if p.strip()]
        return (paths[0].stem if paths else spec), paths
    p = Path(spec)
    return p.stem, [p]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Quantify (tabulate) timing information and plot progress from LSST aggregator logs."
    )
    parser.add_argument(
        "logs",
        nargs="+",
        help=(
            "Log file path(s), optionally prefixed with a label and/or comma-separated "
            "replicates: 'LABEL=run1.out,run2.out' or 'path/to/file.out'."
        ),
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
        "--http-bin",
        type=float,
        default=60.0,
        help="Time bin width in seconds for HTTP-SUMMARY time-series plots (default: 60).",
    )
    parser.add_argument(
        "--max-minutes",
        type=float,
        default=30.0,
        help="Truncate scan_timing plot x-axis at this many minutes (default: 30 = 0.5h; 0 for no truncation).",
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

    groups_by_label: dict[str, RunGroup] = {}
    for spec in args.logs:
        label, paths = parse_input_spec(spec)
        if not paths:
            print(f"Error: no log files specified in '{spec}'", file=sys.stderr)
            return 1
        for path in paths:
            if not path.exists():
                print(f"Error: log file not found: {path}", file=sys.stderr)
                return 1
            run = parse_log(path, label=label)
            if label not in groups_by_label:
                groups_by_label[label] = RunGroup(label=label, runs=[])
            groups_by_label[label].runs.append(run)

    groups = list(groups_by_label.values())

    verify_versions(groups)

    for g in groups:
        for r in g.runs:
            print_single_table(r, show_http_workers=args.http_workers)
        print_group_spread(g)

    if len(groups) >= 2:
        print_comparison_table(groups)

    if args.csv:
        import csv

        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "label",
                    "source_file",
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
            for g in groups:
                for r in g.runs:
                    for s in r.samples:
                        writer.writerow(
                            [
                                g.label,
                                r.path.name,
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
        plot_metrics(groups, args.output, show=args.show, max_minutes=args.max_minutes)
        if any(r.http_summaries for g in groups for r in g.runs):
            http_out = args.output.with_name(
                f"{args.output.stem}_http{args.output.suffix}"
            )
            plot_http_metrics(groups, http_out, bin_s=args.http_bin, show=args.show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
