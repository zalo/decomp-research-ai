"""Step 8: Post-batch review agent that analyzes logs and suggests improvements."""

import re
from pathlib import Path
from dataclasses import dataclass


@dataclass
class BatchReport:
    total_processed: int = 0
    matched: int = 0
    improved: int = 0
    skipped: int = 0
    build_failures: int = 0
    m2c_failures: int = 0
    signature_mismatches: int = 0
    compile_errors: list = None
    common_errors: dict = None
    top_improvements: list = None
    recommendations: list = None

    def __post_init__(self):
        self.compile_errors = self.compile_errors or []
        self.common_errors = self.common_errors or {}
        self.top_improvements = self.top_improvements or []
        self.recommendations = self.recommendations or []


def review_batch_log(log_path: str) -> BatchReport:
    """Analyze a batch log file and produce a structured report."""
    text = Path(log_path).read_text()
    report = BatchReport()

    # Count outcomes
    report.matched = text.count("COMMITTED match!")
    report.improved = text.count("COMMITTED improvement!")
    report.build_failures = text.count("BUILD FAIL")
    report.skipped = text.count("No real improvement") + text.count("No changes")

    # Count function attempts
    report.total_processed = len(re.findall(r'\[\d+/\d+\]', text))

    # Find m2c failures
    report.m2c_failures = text.count("m2c bootstrap didn't compile") + text.count("m2c placeholder didn't compile")

    # Find signature mismatches
    report.signature_mismatches = text.count("SIGNATURE MISMATCH")

    # Extract compile errors
    errors = re.findall(r'COMPILE ERROR:\n(.*?)(?:\n\n|\nYour code)', text, re.DOTALL)
    for err in errors[:10]:
        clean = err.strip()[:200]
        report.compile_errors.append(clean)

    # Find common error patterns
    error_types = {}
    for err in re.findall(r'#\s+Error:.*?(?:\n|$)', text):
        err = err.strip()[:80]
        error_types[err] = error_types.get(err, 0) + 1
    report.common_errors = dict(sorted(error_types.items(), key=lambda x: -x[1])[:5])

    # Extract top improvements
    for m in re.findall(r'→ (?:IMPROVED|MATCHED).*?(\d+\.?\d*)%.*?→.*?(\d+\.?\d*)%.*?COMMITTED', text):
        before, after = float(m[0]), float(m[1])
        report.top_improvements.append((before, after, after - before))
    report.top_improvements.sort(key=lambda x: -x[2])

    # Generate recommendations
    if report.m2c_failures > 3:
        report.recommendations.append(
            f"m2c failed {report.m2c_failures} times. Check build/ctx.c is up-to-date. "
            f"Consider less aggressive cleanup in m2c_runner.py — data tables may be needed."
        )
    if report.build_failures > report.total_processed * 0.3:
        report.recommendations.append(
            f"Build failure rate is {report.build_failures}/{report.total_processed} ({report.build_failures/max(report.total_processed,1)*100:.0f}%). "
            f"The agent is writing code with undefined identifiers. Consider: "
            f"(1) Including more context in the prompt, "
            f"(2) Adding extern declarations to m2c output, "
            f"(3) Validating identifiers before compile."
        )
    if report.signature_mismatches > 0:
        report.recommendations.append(
            f"{report.signature_mismatches} signature mismatches. The agent changed function signatures. "
            f"Enforce original signature more strictly."
        )
    if report.skipped > report.total_processed * 0.4:
        report.recommendations.append(
            f"High skip rate ({report.skipped}/{report.total_processed}). The agent is not improving existing code. "
            f"Consider: targeting different functions, or giving the agent m2c output as starting point."
        )
    if not report.recommendations:
        report.recommendations.append("Batch ran well. Consider increasing batch size or lowering bail-out threshold.")

    return report


def format_report(report: BatchReport) -> str:
    """Format a BatchReport as human-readable text."""
    lines = [
        "=" * 60,
        "BATCH REVIEW REPORT",
        "=" * 60,
        f"Functions processed: {report.total_processed}",
        f"  Matched (100%):     {report.matched}",
        f"  Improved & committed: {report.improved}",
        f"  Skipped/no improve: {report.skipped}",
        f"  Build failures:     {report.build_failures}",
        f"  m2c failures:       {report.m2c_failures}",
        "",
    ]

    if report.top_improvements:
        lines.append("TOP IMPROVEMENTS:")
        for before, after, delta in report.top_improvements[:5]:
            lines.append(f"  {before:.1f}% → {after:.1f}% (+{delta:.1f}%)")
        lines.append("")

    if report.common_errors:
        lines.append("COMMON ERRORS:")
        for err, count in report.common_errors.items():
            lines.append(f"  [{count}x] {err}")
        lines.append("")

    lines.append("RECOMMENDATIONS:")
    for i, rec in enumerate(report.recommendations, 1):
        lines.append(f"  {i}. {rec}")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    log = sys.argv[1] if len(sys.argv) > 1 else "/tmp/decomp_batch_run5.log"
    report = review_batch_log(log)
    print(format_report(report))
