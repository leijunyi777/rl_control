from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "final draft"
EN_TEMPLATE = FINAL_DIR / "MSc_and_BEng_Dissertation_Template_the_University_of_Manchester_EEE"
MAIN_TEX = EN_TEMPLATE / "main.tex"
PROJECT_OUTLINE = ROOT / "project outline.docx"
RISK_ASSESSMENT = ROOT / "Risk Assessment.docx"
OPTIMIZED_MD = FINAL_DIR / "main-优化.md"

START_MARKER = "% BEGIN GENERATED DOCX APPENDICES"
END_MARKER = "% END GENERATED DOCX APPENDICES"
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def latex_escape(text: str) -> str:
    placeholders: dict[str, str] = {}

    def hold(value: str) -> str:
        token = f"ZZZPH{len(placeholders)}ZZZ"
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
    text = re.sub(r"\$[^$]+\$", lambda m: hold(m.group(0)), text)
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
    escaped = (
        escaped.replace("±", r"\(\pm\)")
        .replace("α", r"\(\alpha\)")
        .replace("β", r"\(\beta\)")
        .replace("γ", r"\(\gamma\)")
        .replace("≥", r"\(\geq\)")
        .replace("≤", r"\(\leq\)")
        .replace("≈", r"\(\approx\)")
        .replace("–", "--")
        .replace("—", "---")
    )
    for token, value in placeholders.items():
        escaped = escaped.replace(token, value)
    return escaped


def paragraph_text(paragraph: ET.Element) -> str:
    parts: list[str] = []
    for node in paragraph.iter():
        if node.tag == f"{{{NS['w']}}}t" and node.text:
            parts.append(node.text)
        elif node.tag == f"{{{NS['w']}}}tab":
            parts.append("\t")
        elif node.tag == f"{{{NS['w']}}}br":
            parts.append("\n")
    return "".join(parts).strip()


def paragraph_style(paragraph: ET.Element) -> str:
    style = paragraph.find("./w:pPr/w:pStyle", NS)
    if style is None:
        return ""
    return style.attrib.get(f"{{{NS['w']}}}val", "")


def table_rows(table: ET.Element) -> list[list[str]]:
    rows: list[list[str]] = []
    for tr in table.findall("./w:tr", NS):
        row: list[str] = []
        for tc in tr.findall("./w:tc", NS):
            cell_parts = [paragraph_text(p) for p in tc.findall(".//w:p", NS)]
            row.append(" ".join(part for part in cell_parts if part).strip())
        if any(row):
            rows.append(row)
    return rows


def table_to_latex(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    cols = max(len(row) for row in rows)
    width = max(0.12, min(0.28, 0.92 / cols))
    spec = "".join(f"p{{{width:.2f}\\linewidth}}" for _ in range(cols))
    out = [r"\begin{small}", r"\begin{longtable}{" + spec + "}", r"\toprule"]
    for index, row in enumerate(rows):
        padded = row + [""] * (cols - len(row))
        out.append(" & ".join(latex_escape(cell) for cell in padded[:cols]) + r" \\")
        if index == 0:
            out.append(r"\midrule")
    out.extend([r"\bottomrule", r"\end{longtable}", r"\end{small}"])
    return "\n".join(out)


def docx_to_latex(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    body = root.find("w:body", NS)
    if body is None:
        return ""

    output: list[str] = []
    for child in body:
        if child.tag == f"{{{NS['w']}}}p":
            text = paragraph_text(child)
            if not text:
                continue
            style = paragraph_style(child).lower()
            if "heading1" in style:
                output.append(r"\section*{" + latex_escape(text) + "}")
            elif "heading2" in style:
                output.append(r"\subsection*{" + latex_escape(text) + "}")
            elif "heading3" in style:
                output.append(r"\subsubsection*{" + latex_escape(text) + "}")
            else:
                output.append(latex_escape(text))
        elif child.tag == f"{{{NS['w']}}}tbl":
            output.append(table_to_latex(table_rows(child)))
    return "\n\n".join(output)


def english_table_cell(cell: str) -> str:
    result = cell.strip()
    if ";" in result and re.search(r"[\u4e00-\u9fff]", result.split(";", 1)[1]):
        result = result.split(";", 1)[0].strip()
    if "/" in result:
        parts = [part.strip() for part in result.split("/")]
        english_parts = [part for part in parts if not re.search(r"[\u4e00-\u9fff]", part)]
        if english_parts:
            result = " / ".join(english_parts)
    result = re.sub(r"[\u4e00-\u9fff]+", "", result)
    result = result.replace("，", ",").replace("。", ".").replace("；", ";").replace("：", ":")
    return re.sub(r"\s+", " ", result).strip(" /;:")


def markdown_table_to_latex(lines: list[str]) -> str:
    rows: list[list[str]] = []
    for line in lines:
        cells = [english_table_cell(cell) for cell in line.strip().strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        rows.append(cells)
    cols = max(len(row) for row in rows)
    width = max(0.12, min(0.24, 0.92 / cols))
    spec = "".join(f"p{{{width:.2f}\\linewidth}}" for _ in range(cols))
    out = [r"\begin{small}", r"\begin{longtable}{" + spec + "}", r"\toprule"]
    for index, row in enumerate(rows):
        padded = row + [""] * (cols - len(row))
        out.append(" & ".join(latex_escape(cell) for cell in padded[:cols]) + r" \\")
        if index == 0:
            out.append(r"\midrule")
    out.extend([r"\bottomrule", r"\end{longtable}", r"\end{small}"])
    return "\n".join(out)


def parameter_summary_latex() -> str:
    markdown = OPTIMIZED_MD.read_text(encoding="utf-8-sig")
    match = re.search(r"^## Experimental Parameter Summary.*?\n(?P<body>.*)", markdown, flags=re.S | re.M)
    if not match:
        return ""
    lines = [line for line in match.group("body").splitlines() if line.strip().startswith("|")]
    return markdown_table_to_latex(lines)


def appendices_latex() -> str:
    return "\n\n".join(
        [
            START_MARKER,
            r"\begin{uomappendix}",
            r"\section{Project Outline}",
            docx_to_latex(PROJECT_OUTLINE),
            r"\section{Risk Assessment}",
            docx_to_latex(RISK_ASSESSMENT),
            r"\section{Experimental Parameter Summary}",
            parameter_summary_latex(),
            r"\end{uomappendix}",
            END_MARKER,
        ]
    )


def update_main_tex() -> None:
    tex = MAIN_TEX.read_text(encoding="utf-8")
    block = appendices_latex()
    pattern = re.compile(r"\n?" + re.escape(START_MARKER) + r".*?" + re.escape(END_MARKER) + r"\n?", re.S)
    tex = pattern.sub("\n", tex)
    end_document = r"\end{document}"
    if end_document not in tex:
        raise RuntimeError("Could not find \\end{document} in English main.tex")
    tex = tex.replace(end_document, block + "\n\n" + end_document, 1)
    MAIN_TEX.write_text(tex, encoding="utf-8")


def main() -> None:
    update_main_tex()
    print(f"updated {MAIN_TEX}")


if __name__ == "__main__":
    main()
