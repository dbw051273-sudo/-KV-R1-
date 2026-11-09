"""Build Publication-Grade Academic PDF from Technical Report Markdown.

Workflow:
1. Parse technical report markdown.
2. Embed high-resolution figures (base64 encoded) so the PDF is 100% self-contained.
3. Apply top-tier academic styling (GitHub/IEEE style, clean page-breaks, elegant typography, three-line tables).
4. Render LaTeX formulas using MathJax.
5. Compile to PDF using Microsoft Edge headless engine with --no-pdf-header-footer.
6. Ensure no duplicate headers, full A4 academic layout, and clean typography.
7. Archive desktop .md file to docs/ and Desktop/KV-R1_技术文档归档/ folder, placing the final PDF directly on the Desktop.
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

    # Enhance markdown with actual images (replace placeholder with figure block)
    processed_lines = []
    lines = md_content.splitlines()
    for line in lines:
        matched_fig = False
        for fig_name, fig_path in fig_map.items():
            if line.strip() == f"`results/figures/{fig_name}`" and fig_path.exists():
                b64_uri = encode_image_base64(fig_path)
                processed_lines.append(f'\n<div class="figure-container"><img src="{b64_uri}" alt="{fig_name}" /><div class="figure-caption">图：{fig_name}</div></div>\n')
                matched_fig = True
                break
        if not matched_fig:
            processed_lines.append(line)

    temp_md.write_text("\n".join(processed_lines), encoding="utf-8")

    # Run pandoc to convert to standalone HTML with MathJax.
    # Note: We omit --metadata title=... so Pandoc does NOT inject an extra <header id="title-block-header">.
    pandoc_cmd = [
        "pandoc",
        str(temp_md),
        "-s",
        "--mathjax",
        "-o",
        str(temp_html),
    ]
    print("Running pandoc...")
    subprocess.run(pandoc_cmd, check=True)

    # Inject Academic CSS & MathJax Configuration
    html_content = temp_html.read_text(encoding="utf-8")

    custom_style = """
<style>
@page {
    size: A4 portrait;
    margin: 18mm 16mm 18mm 16mm;
}

html, body {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Strictly hide any pandoc auto-injected title block header to prevent duplicate titles */
#title-block-header {
    display: none !important;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "WenQuanYi Micro Hei", sans-serif;
    font-size: 10pt;
    line-height: 1.62;
    color: #1f2937;
    background: #ffffff;
    text-rendering: optimizeLegibility;
    -webkit-font-smoothing: antialiased;
}

/* Paper Title styling */
h1:first-of-type, h1[id*="长思维链推理模型中-kv-缓存压缩的失效机理与实证分析"] {
    font-size: 21pt;
    font-weight: 800;
    color: #0f172a;
    text-align: center;
    margin-top: 8px;
    margin-bottom: 12px;
    padding-bottom: 0;
    border-bottom: none !important;
    line-height: 1.35;
    letter-spacing: -0.01em;
    page-break-after: avoid;
}

/* Metadata paragraph immediately below title */
h1:first-of-type + p {
    text-align: center;
    font-size: 9.3pt;
    line-height: 1.65;
    color: #4b5563;
    margin-top: 0;
    margin-bottom: 16px;
}

h1:first-of-type + p a {
    color: #2563eb;
    text-decoration: none;
}

hr {
    border: 0;
    height: 1px;
    background: #e2e8f0;
    margin: 16px 0 20px 0;
}

/* Abstract header */
h2#摘要-abstract {
    text-align: center;
    font-size: 12.5pt;
    font-weight: 700;
    color: #0f172a;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: none !important;
    padding-bottom: 0;
    margin-top: 10px;
    margin-bottom: 12px;
    page-break-after: avoid;
}

/* Standard Section Headings */
h2 {
    font-size: 13pt;
    font-weight: 700;
    color: #1e3a8a;
    border-bottom: 1.5px solid #cbd5e1;
    padding-bottom: 4px;
    margin-top: 24px;
    margin-bottom: 12px;
    page-break-after: avoid;
}

h3 {
    font-size: 11pt;
    font-weight: 700;
    color: #1f2937;
    margin-top: 18px;
    margin-bottom: 8px;
    page-break-after: avoid;
}

h4 {
    font-size: 10pt;
    font-weight: 600;
    color: #374151;
    margin-top: 14px;
    margin-bottom: 6px;
    page-break-after: avoid;
}

p {
    margin-top: 6px;
    margin-bottom: 8px;
    text-align: justify;
    text-justify: inter-ideograph;
}

/* Professional Academic Three-Line Tables (Booktabs style) */
table {
    width: 100% !important;
    max-width: 100% !important;
    border-collapse: collapse !important;
    margin: 14px 0 16px 0 !important;
    font-size: 8.8pt !important;
    line-height: 1.45 !important;
    border-top: 2px solid #1e293b !important;
    border-bottom: 2px solid #1e293b !important;
    border-left: none !important;
    border-right: none !important;
    page-break-inside: avoid !important;
}

th {
    border-top: none !important;
    border-bottom: 1.5px solid #1e293b !important;
    border-left: none !important;
    border-right: none !important;
    padding: 7px 8px !important;
    background-color: #f8fafc !important;
    color: #0f172a !important;
    font-weight: 700 !important;
    text-align: left !important;
}

td {
    border-top: 1px solid #f1f5f9 !important;
    border-bottom: 1px solid #f1f5f9 !important;
    border-left: none !important;
    border-right: none !important;
    padding: 6px 8px !important;
    color: #334155 !important;
    vertical-align: top !important;
}

tr:nth-child(even) td {
    background-color: #f8fafc !important;
}

code {
    font-family: "Cascadia Code", "Fira Code", Consolas, Courier, monospace;
    font-size: 8.6pt;
    background-color: #f1f5f9;
    color: #b91c1c;
    padding: 1.5px 4px;
    border-radius: 3px;
    border: 1px solid #e2e8f0;
}

pre {
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 3.5px solid #2563eb;
    padding: 8px 12px;
    border-radius: 4px;
    overflow-x: auto;
    font-size: 8.2pt;
    line-height: 1.45;
    page-break-inside: avoid;
    margin: 10px 0;
}

pre code {
    background-color: transparent;
    border: none;
    color: #1e293b;
    padding: 0;
}

blockquote {
    margin: 10px 0;
    padding: 8px 14px;
    background-color: #f0fdf4;
    border-left: 3.5px solid #10b981;
    color: #166534;
    font-style: normal;
    border-radius: 0 4px 4px 0;
    page-break-inside: avoid;
}

blockquote p {
    margin: 4px 0;
}

.figure-container {
    text-align: center;
    margin: 16px 0 14px 0;
    page-break-inside: avoid;
}

.figure-container img {
    max-width: 96%;
    height: auto;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}

.figure-caption {
    font-size: 8.5pt;
    color: #64748b;
    margin-top: 6px;
    font-style: italic;
}

ul, ol {
    padding-left: 22px;
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
        "--no-pdf-header-footer",
        f"--print-to-pdf={TARGET_PDF}",
        str(temp_html),
    ]
    res = subprocess.run(edge_cmd, capture_output=True, encoding="utf-8", errors="replace")
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
