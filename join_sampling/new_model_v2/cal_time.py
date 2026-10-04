#!/usr/bin/env python3
"""Summarize sampler logs without double-counting Bitmap or nested timings.

Uses only the standard library. Completed template reports are also usable
while workers are running; unfinished reports are excluded from the totals.
"""
import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass, field
import json
from pathlib import Path
import re


WAIT_PHASES = ('predicate_cache.table_lock_wait', 'predicate_cache.capacity_lock_wait')
BUDGET_WAIT = 'annotations.budget_wait'
NUMBER = r'\d+(?:\.\d+)?'
TEMPLATE = re.compile(r'^\s*\[Timing\] Template (.*?) \[(\w+)\]: wall=(' + NUMBER + r')s\s*$')
PHASE = re.compile(r'^\s+([\w./]+): (' + NUMBER + r')s\b')


@dataclass
class TemplateTime:
    label: str
    status: str
    wall: float
    inclusive: dict = field(default_factory=dict)
    exclusive: dict = field(default_factory=dict)

    @property
    def cache_wait(self):
        return sum(self.exclusive.get(name, 0.0) for name in WAIT_PHASES)

    @property
    def budget_wait(self):
        return self.exclusive.get(BUDGET_WAIT, 0.0)

    @property
    def build(self):
        # Inclusive: keep the actual builder's complete preprocessing cost.
        return self.inclusive.get('predicate_cache.build', 0.0)

    @property
    def net(self):
        return self.wall - self.cache_wait

    @property
    def active(self):
        return self.net - self.budget_wait

    @property
    def build_active(self):
        # The capacity lock is inside build; remove it once, not twice.
        return self.build - self.exclusive.get('predicate_cache.capacity_lock_wait', 0.0)

    @property
    def online(self):
        return self.active - self.build_active


@dataclass
class WorkerTime:
    path: Path
    worker_id: int = None
    assigned: int = None
    parsing: float = None
    finished: float = None
    run_id: str = None
    reports: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def summary(self):
        statuses = Counter(report.status for report in self.reports)
        wall = sum(r.wall for r in self.reports)
        cache_wait = sum(r.cache_wait for r in self.reports)
        budget_wait = sum(r.budget_wait for r in self.reports)
        build = sum(r.build for r in self.reports)
        build_active = sum(r.build_active for r in self.reports)
        return dict(file=str(self.path), worker_id=self.worker_id, assigned=self.assigned,
                    reported=len(self.reports), statuses=dict(statuses),
                    finished=self.finished is not None, parsing_seconds=self.parsing,
                    worker_seconds=self.finished, template_seconds=wall,
                    cache_wait_seconds=cache_wait, budget_wait_seconds=budget_wait,
                    cache_build_seconds=build, net_seconds=wall-cache_wait,
                    cache_build_active_seconds=build_active,
                    active_seconds=wall-cache_wait-budget_wait,
                    online_seconds=sum(r.online for r in self.reports),
                    run_id=self.run_id, warnings=self.warnings)


def parse_log(path):
    worker = WorkerTime(Path(path))
    current = None
    mode = None
    starts = 0
    with worker.path.open(encoding='utf-8', errors='replace') as stream:
        for line in stream:
            # A concurrently written final line may be truncated.
            if not line.endswith('\n'):
                continue
            if 'Initializing JoinSampler' in line:
                starts += 1
            match = re.search(r'Worker (\d+) assigned (\d+) templates', line)
            if match:
                worker.worker_id, worker.assigned = map(int, match.groups())
            match = re.search(r'Parsed templates across all queries in (' + NUMBER + r')s', line)
            if match:
                worker.parsing = float(match[1])
            match = re.search(r'Worker (\d+) finished in (' + NUMBER + r')s', line)
            if match:
                worker.worker_id, worker.finished = int(match[1]), float(match[2])
            match = re.search(r'Predicate cache run: ([^,]+),', line)
            if match:
                worker.run_id = match[1].strip()
            if re.search(r'Error sampling template|Error parsing SQL|Traceback', line):
                worker.warnings.append(line.strip())
            match = TEMPLATE.match(line)
            if match:
                if current is not None:
                    worker.warnings.append('上一份 Template 报告不完整，未纳入统计')
                current = TemplateTime(match[1], match[2], float(match[3]))
                mode = None
                continue
            if current is None:
                continue
            if '[Timing]' in line:
                worker.warnings.append('Template 报告被截断，未纳入统计')
                current = None
                mode = None
            elif 'Inclusive phases' in line:
                mode = 'inclusive'
            elif 'Exclusive breakdown' in line:
                mode = 'exclusive'
            elif 'Counters:' in line:
                mode = None
            elif 'DB execute+fetch:' in line:
                if not current.exclusive:
                    worker.warnings.append('Template 缺少 Exclusive 明细，未纳入统计')
                else:
                    worker.reports.append(current)
                current = None
                mode = None
            elif mode:
                match = PHASE.match(line)
                if match:
                    getattr(current, mode)[match[1]] = float(match[2])
    if starts > 1:
        raise ValueError(f'{path}: 包含多次运行，请先分开日志，避免混合统计')
    if current is not None:
        worker.warnings.append('末尾 Template 报告尚未完整，未纳入统计')
    if worker.finished is not None and worker.assigned is not None and len(worker.reports) != worker.assigned:
        worker.warnings.append(f'worker 已结束，但完整报告 {len(worker.reports)} 与 assigned {worker.assigned} 不一致')
    if worker.finished is None:
        worker.warnings.append('worker 尚无 finished 记录；统计仅覆盖已完整输出的 Template 报告')
    return worker


def summarize(workers):
    reports = [r for worker in workers for r in worker.reports]
    inclusive, exclusive = defaultdict(float), defaultdict(float)
    for report in reports:
        for name, seconds in report.inclusive.items():
            inclusive[name] += seconds
        for name, seconds in report.exclusive.items():
            exclusive[name] += seconds
    wall = sum(r.wall for r in reports)
    cache_wait = sum(r.cache_wait for r in reports)
    budget_wait = sum(r.budget_wait for r in reports)
    build = sum(r.build for r in reports)
    build_active = sum(r.build_active for r in reports)
    finished = [w.finished for w in workers if w.finished is not None]
    assigned = sum(w.assigned or 0 for w in workers)
    complete = (all(w.finished is not None and w.assigned is not None
                    and len(w.reports) == w.assigned for w in workers))
    return dict(workers=[w.summary() for w in workers], complete=complete,
                files=len(workers), finished_workers=len(finished), assigned=assigned,
                reported=len(reports), statuses=dict(Counter(r.status for r in reports)),
                template_seconds=wall, cache_wait_seconds=cache_wait,
                budget_wait_seconds=budget_wait, cache_build_seconds=build,
                cache_build_active_seconds=build_active,
                net_seconds=wall-cache_wait, active_seconds=wall-cache_wait-budget_wait,
                online_seconds=sum(r.online for r in reports),
                # This is explicitly NOT the actual makespan without start timestamps.
                max_finished_worker_seconds=max(finished) if finished else None,
                inclusive=dict(inclusive), exclusive=dict(exclusive))


def duration(seconds):
    return f'{seconds:,.4f} 秒 = {seconds/3600:.4f} 小时'


def display(result, top):
    print(f"日志文件：{result['files']}；已结束 worker：{result['finished_workers']}/{result['files']}")
    print(f"完整 Template 报告：{result['reported']}/{result['assigned']}；状态：{result['statuses']}")
    if not result['complete']:
        print('本轮未完整结束或日志缺失，以下只统计完整 Template 报告，不外推剩余时间。')
    print('\n各 worker（小时；报告数包含 complete/empty/failed）：')
    print('文件                         报告/分配     原始累计    缓存等待    净累计    缓存构建')
    for w in result['workers']:
        progress = f"{w['reported']}/{w['assigned'] if w['assigned'] is not None else '?'}"
        print(f"{Path(w['file']).name:<28} {progress:>8} {w['template_seconds']/3600:>11.4f} "
              f"{w['cache_wait_seconds']/3600:>10.4f} {w['net_seconds']/3600:>9.4f} "
              f"{w['cache_build_seconds']/3600:>10.4f}")
    print('\n累计时间（多个 worker 的 wall 时间相加，不是实际经过时间或 CPU 时间）：')
    for label, key in (
        ('原始 template 累计时间', 'template_seconds'),
        ('缓存锁等待时间', 'cache_wait_seconds'),
        ('净累计时间（扣缓存锁等待，保留首次构建）', 'net_seconds'),
        ('位图额度等待时间', 'budget_wait_seconds'),
        ('扣全部显式等待后的累计时间（保留首次构建）', 'active_seconds'),
        ('首次缓存构建时间（已包含在上述累计时间中）', 'cache_build_seconds'),
        ('缓存构建时间扣其内部额度锁等待', 'cache_build_active_seconds'),
        ('在线处理时间（扣全部显式等待及缓存构建）', 'online_seconds')):
        print(f"  {label}：{duration(result[key])}")
    maximum = result['max_finished_worker_seconds']
    if maximum is not None:
        print(f'\n已结束 worker 的最大运行时长（含解析）：{duration(maximum)}')
    print('日志无统一开始/结束时间戳，无法精确恢复整轮实际经过时间。')
    print('Template 累计时间不包含 workload 解析、保存批次等外层工作。')
    if top:
        print('\n耗时最多的 Exclusive 阶段（不重复相加 Inclusive/Bitmap）：')
        for name, seconds in sorted(result['exclusive'].items(), key=lambda item: -item[1])[:top]:
            share = 100*seconds/result['template_seconds'] if result['template_seconds'] else 0
            print(f'  {name:<52} {seconds/3600:>9.4f} 小时  {share:>6.2f}%')
    for w in result['workers']:
        for warning in w['warnings']:
            print(f"提示 [{Path(w['file']).name}]：{warning}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).resolve().parent/'imdb'/'runfile_join_complexity'
    parser.add_argument('log_dir', nargs='?', type=Path, default=default,
                        help=f'日志目录或单个日志文件，默认 {default}')
    parser.add_argument('--pattern', default='*.log', help='目录内日志匹配模式，默认 *.log')
    parser.add_argument('--top', type=int, default=12, help='输出前 N 个 Exclusive 阶段，0 不输出')
    parser.add_argument('--json', type=Path, help='可选：保存完整统计 JSON')
    parser.add_argument('--csv', type=Path, help='可选：保存逐 template CSV')
    args = parser.parse_args(argv)
    files = [args.log_dir] if args.log_dir.is_file() else sorted(args.log_dir.glob(args.pattern))
    if not files:
        parser.error(f'未找到日志：{args.log_dir}（{args.pattern}）')
    try:
        workers = [parse_log(path) for path in files]
        run_ids = {w.run_id for w in workers if w.run_id is not None}
        ids = [w.worker_id for w in workers if w.worker_id is not None]
        if len(run_ids) > 1:
            raise ValueError('目录包含不同 predicate cache 运行 ID，请分别统计')
        if len(ids) != len(set(ids)):
            raise ValueError('输入包含重复 worker 编号，请只选择同一轮的一份日志/worker')
        result = summarize(workers)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    display(result, max(args.top, 0))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'\n已保存 JSON：{args.json}')
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['worker_file', 'template', 'status', 'wall_s', 'cache_wait_s',
                             'budget_wait_s', 'cache_build_s', 'net_s', 'active_s', 'online_s'])
            for worker in workers:
                for r in worker.reports:
                    writer.writerow([worker.path.name, r.label, r.status, r.wall, r.cache_wait,
                                     r.budget_wait, r.build, r.net, r.active, r.online])
        print(f'已保存 CSV：{args.csv}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
