import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIP_DIRS = {".git", ".pytest_cache", "__pycache__", ".venv", "venv", "node_modules", ".ipynb_checkpoints"}
MAX_FILE_BYTES = 5 * 1024 * 1024
FORBIDDEN_SUFFIXES = (".csv.gz", ".gz", ".pt", ".pth", ".ckpt", ".joblib", ".pkl", ".parquet", ".npy", ".npz", ".ubj")
FORBIDDEN_NAMES = {".DS_Store", "HIGGS.csv", ".env"}
SECRET_PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ghp_[A-Za-z0-9]{30,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{30,}"),
    re.compile(r"sk-[A-Za-z0-9]{32,}"),
    re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"][^'\"\s]{12,}['\"]"),
]
METRIC_CLAIM = re.compile(
    r"(?i)(roc[- ]?auc|pr[- ]?auc|average precision|brier|accuracy)[^.\n|]{0,40}?"
    r"(\b0\.\d{2,}\b|\b\d{2,3}(\.\d+)?\s?%)"
)
ALLOW_MARKER = "<!-- allow-metric -->"


def project_files():
    for path in ROOT.rglob("*"):
        if path.is_file() and not (set(path.relative_to(ROOT).parts) & SKIP_DIRS):
            yield path


def markdown_files(*folders):
    for path in project_files():
        if path.suffix == ".md" and (not folders or path.relative_to(ROOT).parts[0] in folders):
            yield path


def strip_code_fences(text):
    return re.sub(r"```.*?```", "", text, flags=re.S)


def test_every_file_in_repository_file_list_exists():
    listing = (ROOT / "REPOSITORY_FILE_LIST.txt").read_text(encoding="utf-8").splitlines()
    paths = [line.strip() for line in listing if line.strip() and "/" in line or line.strip() in {".gitignore", ".editorconfig"}]
    paths = [p for p in paths if not p.startswith("Higgs")]
    missing = [p for p in paths if not (ROOT / p).exists()]
    assert not missing, f"Listed but missing: {missing}"


def test_no_data_model_or_system_junk_files_are_present():
    offenders = []
    for path in project_files():
        rel = path.relative_to(ROOT)
        if path.name in FORBIDDEN_NAMES or str(path).endswith(FORBIDDEN_SUFFIXES):
            offenders.append(str(rel))
    assert not offenders, f"Remove or git-ignore: {offenders}"


def test_no_large_files():
    big = [str(p.relative_to(ROOT)) for p in project_files() if p.stat().st_size > MAX_FILE_BYTES]
    assert not big, f"Files over 5 MB: {big}"


def test_no_secret_like_strings():
    hits = []
    for path in project_files():
        if path.suffix in {".md", ".py", ".txt", ".cff", ".csv", ".yml", ".yaml", ".json", ".ipynb", ".toml", ".cfg", ".ini"} or path.name.startswith(".git"):
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for pattern in SECRET_PATTERNS:
                if pattern.search(text):
                    hits.append((str(path.relative_to(ROOT)), pattern.pattern[:30]))
    assert not hits, f"Possible secrets: {hits}"


def test_gitignore_protects_data_secrets_and_artifacts():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for required in ("data/raw/*", ".env", "outputs/*", "artifacts/*", ".DS_Store", "HIGGS.csv"):
        assert required in text, f".gitignore missing {required}"


def test_github_templates_exist():
    for rel in (".github/PULL_REQUEST_TEMPLATE.md", ".github/ISSUE_TEMPLATE/bug_report.md", ".github/ISSUE_TEMPLATE/feature_request.md"):
        assert (ROOT / rel).is_file(), rel


def test_relative_markdown_links_resolve():
    link = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
    broken = []
    for path in markdown_files():
        text = strip_code_fences(path.read_text(encoding="utf-8"))
        for target in link.findall(text):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            clean = target.split("#")[0]
            if clean and not (path.parent / clean).exists():
                broken.append((str(path.relative_to(ROOT)), target))
    assert not broken, f"Broken links: {broken}"


def test_no_unsourced_metric_numbers_in_docs_and_reports():
    offenders = []
    for path in markdown_files("docs", "reports", "team"):
        text = strip_code_fences(path.read_text(encoding="utf-8"))
        for number, line in enumerate(text.splitlines(), start=1):
            if ALLOW_MARKER in line:
                continue
            if METRIC_CLAIM.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()[:90]}")
    assert not offenders, "Metric-looking numbers found. Replace with artifact-generated values or mark allowed:\n" + "\n".join(offenders)


def test_task_backlog_is_well_formed():
    with open(ROOT / "team" / "TASK_BACKLOG.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate task ids"
    owners = {"Krishna", "Saumya Mishra", "Aditya Rana", "Rachit Gupta", "Ronit Singh"}
    assert {r["primary_owner"] for r in rows} <= owners
    assert {r["priority"] for r in rows} <= {"P0", "P1", "P2"}
    assert {r["status"] for r in rows} <= {"Backlog", "Ready", "In Progress", "In Review", "Blocked", "Done"}


def test_citation_file_has_real_authors():
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert "Project team" not in text, "CITATION.cff still has the placeholder author"
    for name in ("Saumya", "Aditya", "Rachit", "Ronit", "Krishna"):
        assert name in text


def test_changelog_has_unreleased_section_and_no_results_claims():
    text = (ROOT / "team" / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## [Unreleased]" in text
