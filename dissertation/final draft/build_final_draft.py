from __future__ import annotations

import html
import re
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "final draft"
SOURCE_MD = ROOT / "mian.md"
TEMPLATE_SRC = ROOT / "MSc_and_BEng_Dissertation_Template_the_University_of_Manchester_EEE"
TEMPLATE_OUT = OUT_DIR / "MSc_and_BEng_Dissertation_Template_the_University_of_Manchester_EEE"
OPT_MD = OUT_DIR / "main-优化.md"
CN_MD = OUT_DIR / "main-中文优化.md"
CN_DOCX = OUT_DIR / "main-中文优化.docx"
LATEX_MAIN = TEMPLATE_OUT / "main.tex"
LATEX_BIB = TEMPLATE_OUT / "references.bib"
NOTES = OUT_DIR / "修改说明.md"
CHECKLIST = OUT_DIR / "人工确认问题清单.md"


TITLE_EN = "A Two-Layer Decision-Making System for Vehicle Lane Changing Based on Reinforcement Learning"
TITLE_ZH = "基于强化学习的车辆并道双层决策系统"
STUDENT_ID = "14226506"
AUTHOR = "Junyi Lei"
SUBMIT_YEAR = "2026"
COURSE = "Robotics"


EN_REPLACEMENTS = [
    ("A two-layer decision-making system for vehicle lane changing based on reinforcement learning", TITLE_EN),
    ("In the field of autonomous driving, the task of lane changing for vehicles has always been a fundamental issue.", "Lane changing is a fundamental problem in autonomous driving."),
    ("This paper proposes", "This dissertation proposes"),
    ("This method splits", "The proposed method decomposes"),
    ("inter-coupled but clearly functional layers", "coupled but functionally distinct layers"),
    ("is responsible for determining which local gap is more worthy of selection", "determines which local gap is preferable"),
    ("is responsible for converting", "converts"),
    ("the self-vehicle", "the ego vehicle"),
    ("self-driving vehicle", "ego vehicle"),
    ("only reads the three closest target lane vehicles from the ego vehicle", "uses only the three target-lane vehicles closest to the ego vehicle"),
    ("objective offset", "objective bias"),
    ("uses the SAC reinforcement learning strategy", "uses a SAC policy"),
    ("front axle point tracking controller", "front-axle tracking controller"),
    ("Through building simulation experiments in the Python environment, the control method was tested.", "Simulation experiments are implemented in Python to evaluate the proposed control method."),
    ("can well learn the strategy and maintain convergence", "learns an effective strategy and maintains convergence"),
    ("the strategies learned in a simple environment have been verified to be applicable in a complex environment", "the policy learned in the simple environment transfers to the more complex setting"),
    ("the excellent performance of the two-layer decision-making system has been verified", "the effectiveness of the two-layer decision-making system is verified"),
    ("Lane changing and lane switching decisions are representative issues", "Lane-changing decisions are representative problems"),
    ("Therefore, in recent years, research has gradually regarded", "Recent research has therefore increasingly treated"),
    ("Physical-inspired models", "Physics-inspired models"),
    ("At the same time, MARL allows", "Meanwhile, MARL allows"),
    ("group results", "system-level outcomes"),
    ("For the project of this paper", "For this dissertation"),
    ("This project refers to these evaluation perspectives", "This project adopts these evaluation perspectives"),
    ("This project includes five specific objects.", "This project has five specific objectives."),
    ("First, establish", "First, it establishes"),
    ("Second, train", "Second, it trains"),
    ("Third, develop", "Third, it develops"),
    ("Fourth, implement", "Fourth, it implements"),
    ("Fifth, compare", "Fifth, it compares"),
    ("This controller adopts", "The controller adopts"),
    ("on the other hand, solves", "solves"),
    ("by using", "using"),
    ("obtained through RL learning", "learned by reinforcement learning"),
    ("The same opinion dynamics template", "The same opinion-dynamics template"),
    ("its dynamics are where", "its dynamics are defined below, where"),
    ("inputs the environmental evaluation into the system", "injects the environmental evaluation into the system"),
    ("the opinion of top-level decision-makers", "the high-level decision opinion"),
    ("top-level decision-makers", "high-level decision-making"),
    ("top-level", "high-level"),
    ("the following formula itself", "the following self-updating law"),
    ("the opinion sign is symmetrical", "the sign of the opinion is treated symmetrically"),
    ("gap confidence level", "gap confidence"),
    ("At every time step", "At each time step"),
    ("in longitudinal coordinate", "in the longitudinal coordinate"),
    ("The use of differences here is because", "The difference is used because"),
    ("the previous gap", "the front gap"),
    ("the subsequent gap", "the rear gap"),
    ("more in line with", "more consistent with"),
    ("more clear", "clearer"),
    ("and update the attention", "and updates the attention"),
    ("in the form of a threshold", "threshold-based"),
    ("according to the following formula.:", "according to the following rule:"),
    ("The existence of the waiting interval prevents the gap selected from frequently changing", "The waiting interval prevents frequent changes in the selected gap"),
    ("After the high-level module selects the candidate gap", "After the high-level module selects a candidate gap"),
    ("the physical gap length and gap-rate term is", "the physical gap length and gap-rate term are"),
    ("The underlying attention", "The low-level attention"),
    ("underlying opinions", "low-level opinions"),
    ("difficult to be fully designed manually", "difficult to design manually"),
    ("Such an action space is lower-dimensional and easier to interpret.", "This action space is low-dimensional and interpretable."),
    ("Soft Actor-Critic is a off-policy", "Soft Actor-Critic is an off-policy"),
    ("The entropy term encourages exploration", "The entropy term promotes exploration"),
    ("The typical situation is as follows.", "The state is defined as follows."),
    ("The reward is designed to encourage", "The reward function is designed to encourage"),
    ("The definition of the lane-changing progress is", "Lane-changing progress is defined as"),
    ("Its dynamics and The discrete update are", "Its continuous dynamics and discrete update are"),
    ("The opinion-weighted target point and The tracking error are", "The opinion-weighted target point and tracking error are"),
    ("This kind of PID-like structure function is to provide", "This PID-like structure provides"),
    ("The experiment was organized in a way from simple to complex.", "The experiments are organized from simple to complex."),
    ("underlying issues", "low-level control problem"),
    ("when the target gap has been determined", "when the target gap is fixed"),
    ("can the system select", "whether the system can select"),
    ("The specific parameters of the experiment are listed in the appendix.", "The experimental parameters are listed in the appendix."),
    ("the underlying strategy", "the low-level strategy"),
    ("The main purpose of the multi-gap experiment is to verify the generalization ability.", "The multi-gap experiment is designed to evaluate generalization."),
    ("has a generalization ability", "generalizes beyond the training environment"),
    ("in some special circumstances", "in some difficult cases"),
    ("Through the analysis of", "Analysis of"),
    ("the specific reasons mainly fall into two aspects", "the failures mainly arise from two factors"),
    ("By comparing two high-level decision-making strategies:", "The ablation compares two high-level decision-making strategies:"),
    ("the Opinion strategy", "the opinion-dynamics strategy"),
    ("the Max strategy", "the max-score strategy"),
    ("average of fewer switch times", "fewer switches on average"),
    ("was slightly low", "was slightly lower"),
    ("automatic lane merging", "autonomous lane merging"),
    ("learning or parsing attention", "learned or analytic attention"),
    ("a decrease in efficiency or timeout failure", "reduced efficiency or timeout failures"),
    ("The main reason lies in", "This limitation mainly arises from"),
    ("Future work can", "Future work should"),
]


ZH_REPLACEMENTS = [
    ("本论文", "本文"),
    ("Python环境", "Python 环境"),
    ("单gap", "单 gap"),
    ("多gap", "多 gap"),
    ("不同gap", "不同 gap"),
    ("该gap", "该 gap"),
    ("所选gap", "所选 gap"),
    ("gap的", "gap 的"),
    ("gap内", "gap 内"),
    ("gap选择", "gap 选择"),
    ("很好的", "较好地"),
    ("成功的", "成功地"),
    ("RL方法", "RL 方法"),
    ("SAC强化学习", "SAC 强化学习"),
    ("自车", "ego 车辆"),
    ("过于保守", "较为保守"),
    ("优秀", "有效"),
    ("车辆也能成功的完成任务", "车辆能够成功完成任务"),
]


def has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def protect_math(text: str) -> tuple[str, dict[str, str]]:
    protected: dict[str, str] = {}

    def repl(match: re.Match[str]) -> str:
        token = f"@@MATH_BLOCK_{len(protected)}@@"
        protected[token] = match.group(0)
        return token

    return re.sub(r"\$\$.*?\$\$", repl, text, flags=re.S), protected


def restore_math(text: str, protected: dict[str, str]) -> str:
    for token, value in protected.items():
        text = text.replace(token, value)
    return text


def polish_english_paragraph(paragraph: str) -> str:
    if paragraph.lstrip().startswith(("|", "!", "[R", "R")):
        return paragraph
    result = paragraph
    for old, new in EN_REPLACEMENTS:
        result = result.replace(old, new)
    result = re.sub(r"\bthe SAC\b", "SAC", result)
    result = re.sub(r"\bthe RL\b", "RL", result)
    result = result.replace(" ,", ",").replace(" .", ".")
    result = result.replace("metric.The", "metric. The")
    result = result.replace("Figure X", "Fig. 4")
    result = result.replace("±", r"\(\pm\)")
    return result


def polish_chinese_paragraph(paragraph: str) -> str:
    if paragraph.lstrip().startswith(("|", "!", "[R", "R")):
        return paragraph
    result = paragraph
    for old, new in ZH_REPLACEMENTS:
        result = result.replace(old, new)
    result = re.sub(r"([A-Za-z]+)([\u4e00-\u9fff])", r"\1 \2", result)
    result = re.sub(r"([\u4e00-\u9fff])([A-Za-z]+)", r"\1 \2", result)
    result = re.sub(r"\s+", " ", result) if has_cjk(result) and "\n" not in result else result
    return result


def optimize_markdown(source: str) -> str:
    body, math_blocks = protect_math(source)
    parts = re.split(r"(\n\s*\n)", body)
    optimized: list[str] = []
    in_references = False
    for part in parts:
        if part.strip() == "":
            optimized.append(part)
            continue
        if part.startswith("# 参考文献/References"):
            in_references = True
        if in_references:
            optimized.append(part)
            continue
        if has_cjk(part):
            optimized.append(polish_chinese_paragraph(part))
        else:
            optimized.append(polish_english_paragraph(part))
    return restore_math("".join(optimized), math_blocks)


def choose_heading_side(text: str, lang: str) -> str:
    text = text.strip()
    if "/" not in text:
        return text
    left, right = [part.strip() for part in text.split("/", 1)]
    if lang == "zh":
        if has_cjk(left):
            return left
        if has_cjk(right):
            return right
        return left
    left_score = len(re.findall(r"[A-Za-z]", left))
    right_score = len(re.findall(r"[A-Za-z]", right))
    return left if left_score >= right_score else right


def language_markdown(markdown: str, lang: str) -> str:
    lines = markdown.splitlines()
    output: list[str] = []
    skip = False
    for line in lines:
        stripped = line.strip()
        marker = re.fullmatch(r"\*\*(English\.?|中文。?)\*\*", stripped)
        if marker:
            label = marker.group(1)
            skip = ("English" in label and lang == "zh") or ("中文" in label and lang == "en")
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            skip = False
            title = choose_heading_side(heading.group(2), lang)
            if lang == "en" and title in {"标题", "作者", "摘要", "参考文献", "附录"}:
                continue
            output.append(f"{heading.group(1)} {title}")
            continue
        if skip and lang == "en" and (
            stripped.startswith("$$")
            or stripped.startswith("![")
            or stripped.startswith("|")
            or stripped.startswith("Fig")
        ):
            skip = False
        if skip:
            continue
        if lang == "en" and has_cjk(line):
            if line.lstrip().startswith("|"):
                output.append(line)
            elif line.strip().startswith("Fig"):
                output.append(english_caption(line))
            else:
                continue
        else:
            output.append(line)
    return "\n".join(output).strip() + "\n"


def english_caption(line: str) -> str:
    text = line.strip()
    number = ""
    number_match = re.match(r"Fig\.?\s*(\d+)", text, flags=re.I)
    if number_match:
        number = number_match.group(1)
    if "/" in text:
        text = text.rsplit("/", 1)[1].strip()
    text = re.sub(r"[\u4e00-\u9fff]+", "", text)
    text = text.replace("：", ":").replace("  ", " ").strip(" :")
    if text:
        text = text[0].upper() + text[1:]
    return f"Fig. {number}: {text}" if number else text


def extract_abstract(en_md: str) -> str:
    match = re.search(r"^# Abstract\s*\n(?P<body>.*?)(?=^# )", en_md, flags=re.S | re.M)
    return match.group("body").strip() if match else ""


def extract_main_body(en_md: str) -> str:
    start = re.search(r"^# Introduction\s*$", en_md, flags=re.M)
    end = re.search(r"^# References\s*$", en_md, flags=re.M)
    if not start:
        return en_md
    return en_md[start.start() : end.start() if end else len(en_md)].strip()


def extract_references(markdown: str) -> list[tuple[str, str]]:
    ref_section = markdown.split("# 参考文献/References", 1)
    if len(ref_section) < 2:
        return []
    ref_text = re.split(r"(?m)^#\s+附录/Appendices\s*$", ref_section[1], maxsplit=1)[0]
    refs: list[tuple[str, str]] = []
    for match in re.finditer(r"^R(\d+)\.\s+(.*?)(?=\n\nR\d+\.|\Z)", ref_text, flags=re.S | re.M):
        refs.append((f"R{match.group(1)}", " ".join(match.group(2).split())))
    return refs


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
                a, b = item.split("-", 1)
                if a.startswith("R") and b.startswith("R"):
                    keys.extend([f"R{i}" for i in range(int(a[1:]), int(b[1:]) + 1)])
            elif item.startswith("R"):
                keys.append(item)
        return hold(r"\cite{" + ",".join(keys) + "}") if keys else match.group(0)

    text = re.sub(r"\[(R[\d,\-\sR]+)\]", cite_repl, text)

    def math_repl(match: re.Match[str]) -> str:
        return hold(match.group(0))

    text = re.sub(r"\\\(.*?\\\)", math_repl, text)
    text = re.sub(r"\$[^$]+\$", math_repl, text)
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


def table_to_latex(lines: list[str]) -> str:
    rows = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        rows.append(cells)
    if not rows:
        return ""
    cols = len(rows[0])
    spec = "p{0.22\\linewidth}" * cols
    out = [r"\begin{longtable}{" + spec + "}", r"\toprule"]
    for i, row in enumerate(rows):
        row = row + [""] * (cols - len(row))
        out.append(" & ".join(latex_escape_text(english_table_cell(cell)) for cell in row[:cols]) + r" \\")
        out.append(r"\midrule" if i == 0 else "")
    out.append(r"\bottomrule")
    out.append(r"\end{longtable}")
    return "\n".join(line for line in out if line)


def english_table_cell(cell: str) -> str:
    result = cell.strip()
    if ";" in result and has_cjk(result.split(";", 1)[1]):
        result = result.split(";", 1)[0].strip()
    if "/" in result:
        parts = [part.strip() for part in result.split("/")]
        english_parts = [part for part in parts if not has_cjk(part) and re.search(r"[A-Za-z0-9\\(]", part)]
        if english_parts:
            result = " / ".join(english_parts)
    result = re.sub(r"[\u4e00-\u9fff]+", "", result)
    result = result.replace("，", ",").replace("。", ".").replace("；", ";").replace("：", ":")
    result = re.sub(r"\s+", " ", result).strip(" /;:")
    return result


def markdown_to_latex(en_md: str) -> str:
    body = extract_main_body(en_md)
    lines = body.splitlines()
    out: list[str] = []
    i = 0
    in_math = False
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            level = len(heading.group(1))
            title = latex_escape_text(heading.group(2))
            cmd = {1: "section", 2: "subsection", 3: "subsubsection"}.get(level, "paragraph")
            out.append(f"\\{cmd}{{{title}}}")
            i += 1
            continue
        if stripped == "$$":
            out.append(r"\[" if not in_math else r"\]")
            in_math = not in_math
            i += 1
            continue
        if in_math:
            out.append(line)
            i += 1
            continue
        image = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
        if image:
            alt = image.group(1).strip() or "Figure"
            filename = Path(image.group(2)).name
            label = re.sub(r"[^a-zA-Z0-9]+", "-", Path(filename).stem).strip("-").lower()
            out.append(
                "\n".join(
                    [
                        r"\begin{figure}[H]",
                        r"\centering",
                        rf"\includegraphics[width=0.92\linewidth]{{{filename}}}",
                        rf"\caption{{{latex_escape_text(alt)}}}",
                        rf"\label{{fig:{label}}}",
                        r"\end{figure}",
                    ]
                )
            )
            i += 1
            continue
        if stripped.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            out.append(table_to_latex(table_lines))
            continue
        if not stripped:
            out.append("")
        elif stripped.startswith("#"):
            out.append(latex_escape_text(stripped))
        else:
            out.append(latex_escape_text(stripped))
        i += 1
    return remove_blank_lines_inside_math("\n\n".join(out))


def remove_blank_lines_inside_math(latex: str) -> str:
    math_envs = {"equation", "equation*", "align", "align*", "gather", "gather*", "multline", "multline*", "split", "cases"}
    lines = latex.splitlines()
    cleaned = []
    in_display = False
    env_stack = []
    for line in lines:
        stripped = line.strip()
        begin_env = None
        end_env = None
        if stripped.startswith(r"\begin{"):
            begin_env = stripped[len(r"\begin{"):].split("}", 1)[0]
        if stripped.startswith(r"\end{"):
            end_env = stripped[len(r"\end{"):].split("}", 1)[0]
        if (in_display or env_stack) and not stripped:
            continue
        cleaned.append(line)
        if stripped == r"\[":
            in_display = True
        elif stripped == r"\]":
            in_display = False
        elif begin_env in math_envs:
            env_stack.append(begin_env)
        elif end_env in math_envs and env_stack:
            env_stack.pop()
    return "\n".join(cleaned)


def references_to_bib(refs: list[tuple[str, str]]) -> str:
    def bib_escape(value: str) -> str:
        value = re.sub(r"\*([^*]+)\*", r"\1", value)
        value = value.replace("\\", r"\textbackslash{}")
        replacements = {
            "{": r"\{",
            "}": r"\}",
            "#": r"\#",
            "%": r"\%",
            "&": r"\&",
            "_": r"\_",
            "$": r"\$",
        }
        return "".join(replacements.get(ch, ch) for ch in value)

    entries = []
    for key, text in refs:
        doi = ""
        doi_match = re.search(r"https://doi\.org/([^\s]+)", text)
        if doi_match:
            doi = doi_match.group(1)
        safe_text = bib_escape(text)
        year_match = re.search(r"\((19|20)\d{2}\)", text)
        year = year_match.group(0).strip("()") if year_match else SUBMIT_YEAR
        fields = [f"  title = {{{safe_text}}}", "  author = {{See reference list}}", f"  year = {{{year}}}"]
        if doi:
            fields.append(f"  doi = {{{bib_escape(doi)}}}")
        entries.append("@misc{" + key + ",\n" + ",\n".join(fields) + "\n}")
    return "\n\n".join(entries) + "\n"


def make_latex_main(en_md: str, refs: list[tuple[str, str]]) -> str:
    abstract = extract_abstract(en_md)
    body = markdown_to_latex(en_md)
    return rf"""%%%%%%%%%%%%%%%%%% GENERATED DISSERTATION SOURCE %%%%%%%%%%%%%%%%%%
% Generated from the optimized Markdown copy in final draft.
% Original template files were copied before modification.

\begin{{filecontents*}}{{\jobname.xmpdata}}
  \Title{{{TITLE_EN}}}
  \Author{{{STUDENT_ID}}}
  \Language{{en-GB}}
  \Copyrighted{{True}}
\end{{filecontents*}}

\documentclass[11pt,msc]{{uom_eee_dissertation_casson}}

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

\begin{{document}}
\makeatletter
\title{{\xmp@Title}}
\studentid{{\xmp@Author}}
\makeatother

\course{{{COURSE}}}
\faculty{{Science and Engineering}}
\school{{School of Engineering}}
\submitdate{{{SUBMIT_YEAR}}}
\wordcount{{To be confirmed}}
\maketitle

\uomtoc

\begin{{abstract}}
{latex_escape_text(abstract)}
\end{{abstract}}
\clearpage

\begin{{uomoriginality}}
  \uomoriginalitydeclaration
\end{{uomoriginality}}
\uomcopyrightstatement

\begin{{uomacknowledgements}}
Acknowledgements may be added here.
\end{{uomacknowledgements}}

\uomstartmainbody

{body}

\printbibliography[title={{References}},heading=bibintoc]

\end{{document}}
"""


def docx_para(text: str, style: str | None = None) -> str:
    text = html.escape(text)
    style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f"<w:p>{style_xml}<w:r><w:t xml:space=\"preserve\">{text}</w:t></w:r></w:p>"


def make_docx(markdown: str, path: Path) -> None:
    paragraphs: list[str] = []
    in_math = False
    for line in markdown.splitlines():
        stripped = line.strip()
        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            level = min(len(heading.group(1)), 3)
            paragraphs.append(docx_para(heading.group(2), f"Heading{level}"))
        elif stripped == "$$":
            in_math = not in_math
            paragraphs.append(docx_para("$$", "Code"))
        elif stripped.startswith("!["):
            paragraphs.append(docx_para(stripped))
        elif stripped.startswith("|"):
            paragraphs.append(docx_para(stripped))
        elif stripped:
            paragraphs.append(docx_para(line, "Code" if in_math else None))
        else:
            paragraphs.append(docx_para(""))

    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""
    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""
    document_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>"""
    styles = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:style w:type="paragraph" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:pPr><w:outlineLvl w:val="0"/></w:pPr><w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:pPr><w:outlineLvl w:val="1"/></w:pPr><w:rPr><w:b/><w:sz w:val="28"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/><w:basedOn w:val="Normal"/><w:pPr><w:outlineLvl w:val="2"/></w:pPr><w:rPr><w:b/><w:sz w:val="24"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Code"><w:name w:val="Code"/><w:basedOn w:val="Normal"/><w:rPr><w:rFonts w:ascii="Courier New" w:hAnsi="Courier New"/><w:sz w:val="20"/></w:rPr></w:style>
</w:styles>"""
    document = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body>{''.join(paragraphs)}<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr></w:body>
</w:document>"""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/_rels/document.xml.rels", document_rels)
        zf.writestr("word/styles.xml", styles)
        zf.writestr("word/document.xml", document)


def copy_images() -> None:
    image_dir = ROOT / "image"
    target_dir = TEMPLATE_OUT / "images"
    target_dir.mkdir(parents=True, exist_ok=True)
    if image_dir.exists():
        for file in image_dir.iterdir():
            if file.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf"}:
                shutil.copy2(file, target_dir / file.name)


def write_notes() -> None:
    NOTES.write_text(
        """# 修改说明

1. 已从原始 `mian.md` 生成 `main-优化.md`，未直接修改原始草稿。原始文件名为 `mian.md`，文件夹中没有 `main.md`。
2. 英文部分按照 academic-writing-refiner 检查项进行了保守润色：减少 Chinglish 结构、修正冠词和搭配、统一 ego vehicle / target lane / opinion dynamics / SAC policy 等术语，并降低过强或口语化表达。
3. 中文部分根据优化后的英文含义做了同步顺化，主要统一 `单 gap`、`多 gap`、`Python 环境`、`SAC 强化学习` 等表达，未新增研究内容、公式、实验结论或参考文献。
4. 公式块、图片引用、Markdown 表格、参考文献编号 `[R1]` 至 `[R16]` 均作为受保护内容保留。
5. 已复制 Manchester dissertation LaTeX 模板到 `final draft`，并在副本中生成英文 LaTeX 源文件 `main.tex` 和 `references.bib`；原模板目录未被修改。
6. 当前系统环境中 `pandoc`、`latexmk`、`xelatex` 不在 PATH，因此 Word 和 LaTeX 由本地标准库转换器生成，未能在本机完成真实 Pandoc 转换或 LaTeX 编译。
""",
        encoding="utf-8-sig",
    )
    CHECKLIST.write_text(
        """# 人工确认问题清单

1. 请确认原始草稿文件名是否确实应为 `mian.md`；本次已按现有文件处理。
2. 请在安装 Pandoc 和 TeX Live/MiKTeX 后重新编译 LaTeX 模板副本，重点检查 `uom_eee_dissertation_casson` 类文件、`biblatex+biber`、图片路径和长表格分页。
3. Word 版本中的公式以 LaTeX 文本形式保留，不是 Word 原生公式对象；如需提交 Word 原生公式，需要在 Word 中二次转换。
4. 参考文献 BibTeX 条目由编号参考文献自动生成，字段较保守；正式提交前建议用人工维护的 BibTeX 替换。
5. 请核对课程名称、提交年份、学生编号、Declaration、Copyright statement 和 Acknowledgements 是否满足学校最终提交要求。
6. 请逐条核对所有实验数值、成功率、reward、时间和表格数据是否与最终 CSV/图片完全一致。
""",
        encoding="utf-8-sig",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not TEMPLATE_OUT.exists():
        shutil.copytree(TEMPLATE_SRC, TEMPLATE_OUT)
    source = SOURCE_MD.read_text(encoding="utf-8")
    optimized = optimize_markdown(source)
    OPT_MD.write_text(optimized, encoding="utf-8-sig")

    zh_md = language_markdown(optimized, "zh")
    en_md = language_markdown(optimized, "en")
    CN_MD.write_text(zh_md, encoding="utf-8-sig")
    make_docx(zh_md, CN_DOCX)

    copy_images()
    refs = extract_references(optimized)
    LATEX_BIB.write_text(references_to_bib(refs), encoding="utf-8")
    LATEX_MAIN.write_text(make_latex_main(en_md, refs), encoding="utf-8")
    write_notes()

    print(f"wrote {OPT_MD}")
    print(f"wrote {CN_DOCX}")
    print(f"wrote {LATEX_MAIN}")
    print(f"wrote {NOTES}")
    print(f"wrote {CHECKLIST}")


if __name__ == "__main__":
    main()
