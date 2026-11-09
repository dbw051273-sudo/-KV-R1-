"""Build Publication-Grade Academic PDF from Technical Report Markdown.

Workflow:
1. Parse technical report markdown.
2. Embed high-resolution figures (base64 encoded) so the PDF is 100% self-contained.
3. Apply top-tier academic styling (GitHub/IEEE style, clean page-breaks, elegant typography, zebra tables).
4. Render LaTeX formulas using KaTeX / MathJax.
5. Compile to PDF using Microsoft Edge headless engine.
6. Archive desktop .md file to docs/ and Desktop/KV-R1论文资料/ folder, placing the final PDF directly on the Desktop.
"""

import base64
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT_DIR / "docs"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
DESKTOP_DIR = Path(r"C:\Users\felix\Desktop")
ARCHIVE_DIR = DESKTOP_DIR / "KV-R1_技术文档归档"

SOURCE_MD = DOCS_DIR / "长思维链推理模型KV缓存压缩评估与分析_技术报告.md"
TARGET_PDF = DESKTOP_DIR / "长思维链推理模型KV缓存压缩评估与分析_技术报告.pdf"
EDGE_EXE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
if not EDGE_EXE.exists():
    EDGE_EXE = Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe")

def encode_image_base64(path: Path) -> str:
    if path.exists():
        data = path.read_bytes()
        b64 = base64.b64encode(data).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    return ""

def generate_pdf():
    print(f"Reading {SOURCE_MD}...")
    md_content = SOURCE_MD.read_text(encoding="utf-8")

    # Prepare temporary HTML file
    temp_dir = ROOT_DIR / "build_pdf_tmp"
    temp_dir.mkdir(exist_ok=True)
    temp_md = temp_dir / "report_processed.md"
    temp_html = temp_dir / "report.html"

    # Embed images in Section 5
    fig_map = {
        "fig1_coverage_and_horizon_survival.png": FIGURES_DIR / "fig1_coverage_and_horizon_survival.png",
        "fig2_suite_a_q17b_vs_q4b_all3.png": FIGURES_DIR / "fig2_suite_a_q17b_vs_q4b_all3.png",
        "fig3_math500_benchmark.png": FIGURES_DIR / "fig3_math500_benchmark.png",
        "fig4_budget_scaling_pareto.png": FIGURES_DIR / "fig4_budget_scaling_pareto.png",
        "fig5_efficiency_and_memory_scaling.png": FIGURES_DIR / "fig5_efficiency_and_memory_scaling.png",
        "fig6_gsm8k_benchmark.png": FIGURES_DIR / "fig6_gsm8k_benchmark.png",
        "fig7_qknorm_control_qwen25.png": FIGURES_DIR / "fig7_qknorm_control_qwen25.png",
        "fig8_unified_latency_benchmark.png": FIGURES_DIR / "fig8_unified_latency_benchmark.png",
    }

    # Enhance markdown with actual images
    processed_lines = []
    lines = md_content.splitlines()
    for line in lines:
        processed_lines.append(line)
        # Check if line introduces a figure
        for fig_name, fig_path in fig_map.items():
            if f"`results/figures/{fig_name}`" in line and fig_path.exists():
                b64_uri = encode_image_base64(fig_path)
                processed_lines.append(f'\n<div class="figure-container"><img src="{b64_uri}" alt="{fig_name}" /><div class="figure-caption">图：{fig_name}</div></div>\n')

    temp_md.write_text("\n".join(processed_lines), encoding="utf-8")

    # Run pandoc to convert to standalone HTML with MathJax
    pandoc_cmd = [
        "pandoc",
        str(temp_md),
        "-s",
        "--mathjax",
        "-o",
        str(temp_html),
        "--metadata",
        "title=长思维链推理模型KV缓存压缩评估与分析：机理解构、实证基准与系统优化",
    ]
    print("Running pandoc...")
    subprocess.run(pandoc_cmd, check=True)

    # Inject Academic CSS & MathJax Configuration
    html_content = temp_html.read_text(encoding="utf-8")

    custom_style = """
<style>
@page {
    size: A4;
    margin: 18mm 16mm 18mm 16mm;
    @bottom-center {
        content: counter(page);
        font-size: 9pt;
        color: #718096;
    }
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", "Hiragino Sans GB", "WenQuanYi Micro Hei", sans-serif;
    font-size: 10.2pt;
    line-height: 1.62;
    color: #2d3748;
    background: #ffffff;
    padding: 0;
    margin: 0;
}

h1 {
    font-size: 20pt;
    font-weight: 700;
    color: #1a202c;
    border-bottom: 2.5px solid #3182ce;
    padding-bottom: 8px;
    margin-top: 24px;
    margin-bottom: 16px;
    page-break-after: avoid;
}

h2 {
    font-size: 14pt;
    font-weight: 600;
    color: #2b6cb0;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 5px;
    margin-top: 22px;
    margin-bottom: 12px;
    page-break-after: avoid;
}

h3 {
    font-size: 11.5pt;
    font-weight: 600;
    color: #2d3748;
    margin-top: 16px;
    margin-bottom: 8px;
    page-break-after: avoid;
}

h4 {
    font-size: 10.5pt;
    font-weight: 600;
    color: #4a5568;
    margin-top: 14px;
    margin-bottom: 6px;
    page-break-after: avoid;
}

p {
    margin-top: 6px;
    margin-bottom: 8px;
    text-align: justify;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0;
    font-size: 8.8pt;
    page-break-inside: avoid;
}

th, td {
    padding: 6px 8px;
    text-align: left;
    border: 1px solid #cbd5e0;
}

th {
    background-color: #ebf8ff;
    color: #2b6cb0;
    font-weight: 600;
    border-bottom: 2px solid #bee3f8;
}

tr:nth-child(even) {
    background-color: #f7fafc;
}

code {
    font-family: "Cascadia Code", "Fira Code", Consolas, Courier, monospace;
    font-size: 8.8pt;
    background-color: #edf2f7;
    color: #c53030;
    padding: 2px 4px;
    border-radius: 3px;
}

pre {
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #3182ce;
    padding: 10px 12px;
    border-radius: 4px;
    overflow-x: auto;
    font-size: 8.5pt;
    line-height: 1.45;
    page-break-inside: avoid;
}

pre code {
    background-color: transparent;
    color: #2d3748;
    padding: 0;
}

blockquote {
    margin: 10px 0;
    padding: 8px 14px;
    background-color: #ebf8ff;
    border-left: 4px solid #3182ce;
    color: #2c5282;
    font-style: normal;
    border-radius: 0 4px 4px 0;
    page-break-inside: avoid;
}

blockquote p {
    margin: 4px 0;
}

.figure-container {
    text-align: center;
    margin: 16px 0;
    page-break-inside: avoid;
}

.figure-container img {
    max-width: 95%;
    height: auto;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.06);
}

.figure-caption {
    font-size: 8.5pt;
    color: #718096;
    margin-top: 5px;
    font-style: italic;
}

hr {
    border: 0;
    height: 1px;
    background: #e2e8f0;
    margin: 20px 0;
}

ul, ol {
    padding-left: 20px;
    margin-top: 4px;
    margin-bottom: 8px;
}

li {
    margin-bottom: 4px;
}

.math {
    font-size: 9.8pt;
}
</style>
<script>
window.MathJax = {
    tex: {
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
    },
    svg: { fontCache: 'global' }
};
</script>
"""
    # Insert style before </head>
    html_content = html_content.replace("</head>", f"{custom_style}</head>")
    temp_html.write_text(html_content, encoding="utf-8")

    # Compile HTML to PDF via Headless Edge
    print(f"Compiling PDF via {EDGE_EXE}...")
    edge_cmd = [
        str(EDGE_EXE),
        "--headless",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=6000",
        "--print-to-pdf-no-header",
        f"--print-to-pdf={TARGET_PDF}",
        str(temp_html),
    ]
    res = subprocess.run(edge_cmd, capture_output=True, text=True)
    if not TARGET_PDF.exists() or TARGET_PDF.stat().st_size == 0:
        raise RuntimeError(f"Edge PDF generation failed: {res.stderr}")

    print(f"PDF successfully created: {TARGET_PDF} ({TARGET_PDF.stat().st_size / 1024:.1f} KB)")

    # Clean up and manage markdown files
    desktop_md = DESKTOP_DIR / "长思维链推理模型KV缓存压缩评估与分析_技术报告.md"
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    if desktop_md.exists():
        archive_target = ARCHIVE_DIR / desktop_md.name
        shutil.move(desktop_md, archive_target)
        print(f"Moved desktop raw markdown to archive: {archive_target}")

    # Copy PDF to repo docs as well
    repo_pdf = DOCS_DIR / TARGET_PDF.name
    shutil.copy2(TARGET_PDF, repo_pdf)
    print(f"Mirrored PDF to repository docs: {repo_pdf}")

if __name__ == "__main__":
    generate_pdf()
