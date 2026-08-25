from __future__ import annotations

import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "final draft"
SOURCE_MD = OUT_DIR / "main-中文优化.md"
TEMPLATE_SRC = ROOT / "MSc_and_BEng_Dissertation_Template_the_University_of_Manchester_EEE"
ZH_TEMPLATE = OUT_DIR / "MSc_and_BEng_Dissertation_Template_the_University_of_Manchester_EEE_Chinese"
MAIN_TEX = ZH_TEMPLATE / "main.tex"
ROOT_TEX_COPY = OUT_DIR / "main-中文Latex.tex"
BIB_FILE = ZH_TEMPLATE / "references.bib"
IMAGE_SRC = ROOT / "image"

TITLE_ZH = "基于强化学习的车辆并道双层决策系统"
STUDENT_ID = "14226506"
SUBMIT_YEAR = "2026"
COURSE = "Robotics"


def has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def latex_escape_text(text: str) -> str:
    placeholders: dict[str, str] = {}

    def hold(value: str) -> str:
        token = f"ZZZHOLD{len(placeholders)}ZZZ"
        placeholders[token] = value
        return token

    def cite_repl(match: re.Match[str]) -> str:
        raw = match.group(1).replace(" ", "")
        keys: list[str] = []
        for item in raw.split(","):
            if "-" in item:
                left, right = item.split("-", 1)
                if left.startswith("R") and right.startswith("R"):
                    keys.extend(f"R{i}" for i in range(int(left[1:]), int(right[1:]) + 1))
            elif item.startswith("R"):
                keys.append(item)
        return hold(r"\cite{" + ",".join(keys) + "}") if keys else match.group(0)

    text = re.sub(r"\[(R[\d,\-\sR]+)\]", cite_repl, text)
    text = re.sub(r"\\\(.*?\\\)", lambda m: hold(m.group(0)), text)
    text = re.sub(r"`([^`]+)`", lambda m: hold(r"\texttt{" + m.group(1) + "}"), text)

    chars = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    escaped = "".join(chars.get(ch, ch) for ch in text)
    escaped = re.sub(r"\*\*(.*?)\*\*", r"\\textbf{\1}", escaped)
    escaped = (
        escaped.replace("±", r"\(\pm\)")
        .replace("α", r"\(\alpha\)")
        .replace("β", r"\(\beta\)")
        .replace("γ", r"\(\gamma\)")
        .replace("≥", r"\(\geq\)")
        .replace("≤", r"\(\leq\)")
        .replace("≈", r"\(\approx\)")
    )
    for token, value in placeholders.items():
        escaped = escaped.replace(token, value)
    return escaped


def split_front_matter(markdown: str) -> tuple[str, str]:
    abstract_match = re.search(r"^# 摘要\s*\n(?P<body>.*?)(?=^# )", markdown, flags=re.S | re.M)
    abstract = abstract_match.group("body").strip() if abstract_match else ""
    start = re.search(r"^# 简介\s*$", markdown, flags=re.M)
    end = re.search(r"^# 参考文献\s*$", markdown, flags=re.M)
    if not start:
        return abstract, markdown
    body = markdown[start.start() : end.start() if end else len(markdown)].strip()
    return abstract, body


def extract_references(markdown: str) -> list[tuple[str, str]]:
    if "# 参考文献" not in markdown:
        return []
    ref_section = markdown.split("# 参考文献", 1)[1]
    refs: list[tuple[str, str]] = []
    for match in re.finditer(r"^R(\d+)\.\s+(.*?)(?=\n\nR\d+\.|\Z)", ref_section, flags=re.S | re.M):
        refs.append((f"R{match.group(1)}", " ".join(match.group(2).split())))
    return refs


def table_to_latex(lines: list[str]) -> str:
    rows: list[list[str]] = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        rows.append(cells)
    if not rows:
        return ""
    cols = len(rows[0])
    width = max(0.16, min(0.28, 0.92 / cols))
    spec = "".join([f"p{{{width:.2f}\\linewidth}}" for _ in range(cols)])
    out = [r"\begin{longtable}{" + spec + "}", r"\toprule"]
    for row_index, row in enumerate(rows):
        row = row + [""] * (cols - len(row))
        out.append(" & ".join(latex_escape_text(cell) for cell in row[:cols]) + r" \\")
        out.append(r"\midrule" if row_index == 0 else "")
    out.append(r"\bottomrule")
    out.append(r"\end{longtable}")
    return "\n".join(line for line in out if line)


def caption_from_following(lines: list[str], index: int, fallback: str) -> tuple[str, int]:
    if index + 1 < len(lines) and lines[index + 1].strip().startswith("Fig"):
        caption = lines[index + 1].strip()
        if "：" in caption:
            caption = caption.split("：", 1)[1]
        if "/" in caption:
            caption = caption.split("/", 1)[0]
        return caption.strip() or fallback, index + 2
    return fallback, index + 1


def markdown_to_latex(markdown: str) -> str:
    _, body = split_front_matter(markdown)
    lines = body.splitlines()
    output: list[str] = []
    in_math = False
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            level = len(heading.group(1))
            title = latex_escape_text(heading.group(2))
            command = {1: "chapter", 2: "section", 3: "subsection", 4: "subsubsection"}.get(level, "paragraph")
            output.append(f"\\{command}{{{title}}}")
            i += 1
            continue

        if stripped == "$$":
            output.append(r"\[")
            in_math = not in_math
            i += 1
            continue

        if in_math:
            output.append(line)
            i += 1
            continue

        image = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
        if image:
            filename = Path(image.group(2)).name
            fallback = image.group(1).strip() or "图"
            caption, next_index = caption_from_following(lines, i, fallback)
            label = re.sub(r"[^a-zA-Z0-9]+", "-", Path(filename).stem).strip("-").lower()
            output.append(
                "\n".join(
                    [
                        r"\begin{figure}[H]",
                        r"\centering",
                        rf"\includegraphics[width=0.92\linewidth]{{{filename}}}",
                        rf"\caption{{{latex_escape_text(caption)}}}",
                        rf"\label{{fig:{label}}}",
                        r"\end{figure}",
                    ]
                )
            )
            i = next_index
            continue

        if stripped.startswith("|"):
            table_lines: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            output.append(table_to_latex(table_lines))
            continue

        if stripped.startswith("Fig"):
            i += 1
            continue

        output.append(latex_escape_text(line) if stripped else "")
        i += 1

    return "\n\n".join(output)


def references_to_bib(refs: list[tuple[str, str]]) -> str:
    entries = []
    for key, text in refs:
        doi_match = re.search(r"https://doi\.org/([^\s]+)", text)
        fields = [
            f"  title = {{{text.replace('{', '').replace('}', '')}}}",
            "  author = {{See reference list}}",
            "  year = {{n.d.}}",
        ]
        if doi_match:
            fields.append(f"  doi = {{{doi_match.group(1)}}}")
        entries.append("@misc{" + key + ",\n" + ",\n".join(fields) + "\n}")
    return "\n\n".join(entries) + "\n"


def make_chinese_latex(markdown: str) -> str:
    abstract, _ = split_front_matter(markdown)
    body = markdown_to_latex(markdown)
    return rf"""%%%%%%%%%%%%%%%%%% GENERATED CHINESE DISSERTATION SOURCE %%%%%%%%%%%%%%%%%%
% 中文版论文源文件。请优先使用 XeLaTeX + biber 编译。
% 原始 Manchester 模板已复制到 final draft 中，原模板未被修改。

\begin{{filecontents*}}{{\jobname.xmpdata}}
  \Title{{{TITLE_ZH}}}
  \Author{{{STUDENT_ID}}}
  \Language{{zh-CN}}
  \Copyrighted{{True}}
\end{{filecontents*}}

\documentclass[11pt,msc]{{uom_eee_dissertation_casson}}

\usepackage{{iftex}}
\ifXeTeX
  \usepackage{{xeCJK}}
  \setCJKmainfont{{SimSun}}
  \setCJKsansfont{{Microsoft YaHei}}
  \setCJKmonofont{{SimSun}}
\else
  \ifLuaTeX
    \usepackage{{luatexja-fontspec}}
    \setmainjfont{{SimSun}}
  \else
    \PackageError{{main}}{{Chinese version requires XeLaTeX or LuaLaTeX}}{{Use xelatex or lualatex instead of pdflatex.}}
  \fi
\fi

\usepackage{{graphicx}}
  \graphicspath{{ {{./images/}} }}
\usepackage{{amsmath}}
  \allowdisplaybreaks[1]
\usepackage{{amssymb}}
\usepackage{{url}}
\usepackage{{float}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{array}}
\usepackage{{tabularx}}
\usepackage{{multirow}}
\usepackage{{subcaption}}
\usepackage{{siunitx}}
\usepackage[base]{{babel}}

\newcommand{{\degree}}{{\ensuremath{{^\circ}}}}
\newcommand{{\sus}}[1]{{$^{{\mbox{{\scriptsize #1}}}}$}}
\newcommand{{\sub}}[1]{{$_{{\mbox{{\scriptsize #1}}}}$}}
\newcommand{{\otoprule}}{{\midrule[\heavyrulewidth]}}
\newcolumntype{{Z}}{{>{{\centering\arraybackslash}}X}}

\usepackage[style=ieee,backend=biber,backref=true,hyperref=auto,maxbibnames=3,minbibnames=1]{{biblatex}}
\DefineBibliographyStrings{{english}}{{backrefpage = {{cited on p\adddot}}, backrefpages = {{cited on pp\adddot}}}}
\addbibresource{{references.bib}}

\quickwordcount{{\currfilebase}}

\begin{{document}}
\makeatletter
\title{{\xmp@Title}}
\studentid{{\xmp@Author}}
\makeatother

\course{{{COURSE}}}
\faculty{{Science and Engineering}}
\school{{School of Engineering}}
\submitdate{{{SUBMIT_YEAR}}}
\wordcount{{\mywordcount}}
\maketitle

\uomtoc

\begin{{abstract}}
{latex_escape_text(abstract)}
\end{{abstract}}
\clearpage

\begin{{uomoriginality}}
本人特此确认，本文中所述的任何部分内容均未被提交用于申请本校或其他任何大学或学习机构的其他学位或资格。
\end{{uomoriginality}}
\uomcopyrightstatement

\begin{{uomacknowledgements}}
致谢内容可在此处补充。
\end{{uomacknowledgements}}

\uomstartmainbody

{body}

\printbibliography

\end{{document}}
"""


def prepare_template() -> None:
    if ZH_TEMPLATE.exists():
        return
    shutil.copytree(TEMPLATE_SRC, ZH_TEMPLATE)


def copy_images() -> None:
    target = ZH_TEMPLATE / "images"
    target.mkdir(parents=True, exist_ok=True)
    for file in IMAGE_SRC.glob("*"):
        if file.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf"}:
            shutil.copy2(file, target / file.name)


def main() -> None:
    prepare_template()
    copy_images()
    markdown = SOURCE_MD.read_text(encoding="utf-8-sig")
    refs = extract_references(markdown)
    tex = make_chinese_latex(markdown)
    MAIN_TEX.write_text(tex, encoding="utf-8")
    ROOT_TEX_COPY.write_text(tex, encoding="utf-8")
    BIB_FILE.write_text(references_to_bib(refs), encoding="utf-8")
    print(f"wrote {MAIN_TEX}")
    print(f"wrote {ROOT_TEX_COPY}")
    print(f"wrote {BIB_FILE}")


if __name__ == "__main__":
    main()
