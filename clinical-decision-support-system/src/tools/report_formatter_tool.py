"""
Report Formatter Tool — takes a structured report dict and renders it
as both a human-readable text string and a PDF file.

PDF generation uses ReportLab (pure Python, no external dependencies).
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional


class ReportFormatterTool:
    """
    Fills the standard clinical report template and optionally exports to PDF.
    """

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    #  Table helpers                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _row(label: str, value: str, w: int = 70) -> str:
        """Single two-column row: | label | value |"""
        col1 = 20
        col2 = w - col1 - 5          # 5 = len("| " + " | " + " |")
        lbl  = label[:col1].ljust(col1)
        val  = str(value)
        # wrap value if too long
        lines_out = []
        while val:
            chunk, val = val[:col2], val[col2:]
            lines_out.append(f"| {lbl} | {chunk.ljust(col2)} |")
            lbl = " " * col1          # indent continuation lines
        return "\n".join(lines_out)

    @staticmethod
    def _section(title: str, w: int = 70) -> str:
        inner = w - 2                 # 2 = len("| " + " |") borders
        return f"| {'  ' + title:<{inner}} |"

    @staticmethod
    def _divider(char: str = "-", w: int = 70) -> str:
        return "+" + char * (w - 2) + "+"

    @staticmethod
    def _wrap_text(text: str, width: int = 66) -> list:
        """Break text into lines that fit within width."""
        import textwrap
        return textwrap.wrap(text, width=width) or [""]

    def format_text_report(self, report: dict) -> str:
        """Return a clean table-based plain-text version of the report."""
        W = 72          # total table width
        c = report.get("classification", {})
        tumor_type = c.get("tumor_type", "N/A")
        confidence = c.get("confidence", 0)

        def divider(ch="-"):   return "+" + ch * (W - 2) + "+"
        def section(title):
            inner = W - 4
            return "| " + f"  {title}".ljust(inner) + " |"
        def row(label, value):
            col1, col2 = 22, W - 22 - 5
            lbl = label[:col1].ljust(col1)
            val = str(value)
            out = []
            while True:
                chunk, val = val[:col2], val[col2:]
                out.append(f"| {lbl} | {chunk.ljust(col2)} |")
                if not val:
                    break
                lbl = " " * col1
            return "\n".join(out)

        lines = []

        # Header
        lines.append(divider("="))
        title = "CLINICAL DECISION SUPPORT — MRI BRAIN TUMOR REPORT"
        lines.append("| " + title.center(W - 4) + " |")
        lines.append(divider("="))

        # Patient info
        pm = report.get("patient_metadata", {})
        lines.append(section("PATIENT INFO"))
        lines.append(divider())
        lines.append(row("Patient ID",   report.get("patient_id", "ANONYMOUS")))
        lines.append(row("Name",         pm.get("name", "Anonymous")))
        lines.append(row("Age",          f"{pm.get('age', '—')} yrs" if pm.get("age") else "—"))
        lines.append(row("Sex",          pm.get("sex", "—")))
        lines.append(row("Comorbidities", ", ".join(pm.get("comorbidities", [])) or "None reported"))
        lines.append(row("Symptoms",     ", ".join(pm.get("presenting_symptoms", [])) or "None reported"))
        ts = report.get("timestamp", "")[:19].replace("T", "  ")
        lines.append(row("Date / Time",  ts))
        lines.append(row("Run ID",       report.get("run_id", "N/A")))
        lines.append(divider("="))

        # Classification
        lines.append(section("CLASSIFICATION RESULT"))
        lines.append(divider())
        lines.append(row("Tumor Type",  tumor_type.upper()))
        bar = "█" * int(confidence * 30)
        lines.append(row("Confidence",  f"{confidence:.1%}  {bar}"))
        lines.append(divider("="))

        # Class probabilities
        lines.append(section("CLASS PROBABILITIES"))
        lines.append(divider())
        col1, col2 = 22, 10
        col3 = W - col1 - col2 - 7   # 7 = borders + separators
        hdr = f"| {'Class'.ljust(col1)} | {'Score'.ljust(col2)} | {'Probability Bar'.ljust(col3)} |"
        lines.append(hdr)
        lines.append(divider())
        for cls, score in sorted(c.get("all_predictions", {}).items(),
                                 key=lambda x: x[1], reverse=True):
            bar = "█" * int(score * col3)
            marker = " <--" if cls == tumor_type else ""
            lines.append(
                f"| {cls.ljust(col1)} | {f'{score:.3f}'.ljust(col2)} | {(bar + marker).ljust(col3)} |"
            )
        lines.append(divider("="))

        # Clinical summary
        lines.append(section("CLINICAL SUMMARY"))
        lines.append(divider())
        import textwrap
        for wrapped_line in textwrap.wrap(report.get("clinical_summary", "N/A"), W - 6):
            inner = W - 4
            lines.append("| " + f"  {wrapped_line}".ljust(inner) + " |")
        lines.append(divider("="))

        # Patient-specific notes
        patient_notes = report.get("patient_specific_notes", "")
        if patient_notes:
            lines.append(section("PATIENT-SPECIFIC CLINICAL CONSIDERATIONS"))
            lines.append(divider())
            for note_line in patient_notes.split("\n"):
                if note_line.strip():
                    for wrapped_line in textwrap.wrap(note_line, W - 6):
                        inner = W - 4
                        lines.append("| " + f"  {wrapped_line}".ljust(inner) + " |")
            lines.append(divider("="))

        # Recommended next steps
        lines.append(section("RECOMMENDED NEXT STEPS"))
        lines.append(divider())
        steps = report.get("recommended_next_steps", [])
        if steps:
            col_n, col_s = 4, W - 4 - 4 - 4   # num col, step col
            for i, step in enumerate(steps, 1):
                for j, chunk in enumerate(textwrap.wrap(step, col_s) or [step]):
                    num = str(i) if j == 0 else " "
                    lines.append(f"| {num.center(col_n)} | {chunk.ljust(col_s)} |")
        else:
            inner = W - 4
            lines.append("| " + "  (none)".ljust(inner) + " |")
        lines.append(divider("="))

        # Radiologist review
        lines.append(section("RADIOLOGIST REVIEW"))
        lines.append(divider())
        lines.append(row("Status",  report.get("radiologist_approval", "Pending")))
        notes = report.get("radiologist_notes", "") or "—"
        lines.append(row("Notes",   notes))
        lines.append(divider("="))

        # Disclaimer
        disc = report.get("disclaimer", "")
        lines.append(section("DISCLAIMER"))
        lines.append(divider())
        for wrapped_line in textwrap.wrap(disc, W - 6):
            inner = W - 4
            lines.append("| " + f"  {wrapped_line}".ljust(inner) + " |")
        lines.append(divider("="))

        return "\n".join(lines)

    def save_pdf_report(self, report: dict, output_path: str) -> dict:
        """
        Export the report to a PDF file.

        Returns {"status": "success", "path": "..."} or {"status": "error", ...}
        """
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.units import cm
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
        except ImportError:
            return {"status": "error", "message": "reportlab not installed. Run: pip install reportlab"}

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        doc    = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2*cm, rightMargin=2*cm,
                                   topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        story  = []

        # Title
        story.append(Paragraph("Clinical Decision Support System", styles["Title"]))
        story.append(Paragraph("MRI Brain Tumor Analysis Report", styles["Heading2"]))
        story.append(Spacer(1, 0.5*cm))

        # Patient info table
        c  = report.get("classification", {})
        pm = report.get("patient_metadata", {})
        info_data = [
            ["Patient ID",   report.get("patient_id", "ANONYMOUS")],
            ["Name",         pm.get("name", "Anonymous")],
            ["Age",          f"{pm.get('age', '—')} years" if pm.get("age") else "—"],
            ["Sex",          pm.get("sex", "—")],
            ["Comorbidities", ", ".join(pm.get("comorbidities", [])) or "None reported"],
            ["Symptoms",     ", ".join(pm.get("presenting_symptoms", [])) or "None reported"],
            ["Date/Time",    report.get("timestamp", "")],
            ["Run ID",       report.get("run_id", "N/A")],
            ["Tumor Type",   c.get("tumor_type", "N/A").upper()],
            ["Confidence",   f"{c.get('confidence', 0):.1%}"],
            ["Radiologist",  report.get("radiologist_approval", "Pending")],
        ]
        t = Table(info_data, colWidths=[5*cm, 12*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.lightblue),
            ("FONTNAME",   (0, 0), (0, -1), "Helvetica-Bold"),
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING",    (0, 0), (-1, -1), 6),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.5*cm))

        # Clinical summary
        story.append(Paragraph("Clinical Summary", styles["Heading3"]))
        story.append(Paragraph(report.get("clinical_summary", ""), styles["Normal"]))
        story.append(Spacer(1, 0.3*cm))

        # Patient-specific clinical notes
        patient_notes = report.get("patient_specific_notes", "")
        if patient_notes:
            story.append(Paragraph("Patient-Specific Clinical Considerations", styles["Heading3"]))
            for line in patient_notes.split("\n"):
                if line.strip():
                    story.append(Paragraph(line, styles["Normal"]))
            story.append(Spacer(1, 0.3*cm))

        # Recommendations
        story.append(Paragraph("Recommended Next Steps", styles["Heading3"]))
        for i, step in enumerate(report.get("recommended_next_steps", []), 1):
            story.append(Paragraph(f"{i}. {step}", styles["Normal"]))
        story.append(Spacer(1, 0.3*cm))

        # Guidelines
        guidelines = report.get("medical_guidelines", "")
        if guidelines:
            story.append(Paragraph("Clinical Guidelines", styles["Heading3"]))
            excerpt = guidelines[:2000] + ("..." if len(guidelines) > 2000 else "")
            for para in excerpt.split("\n"):
                if para.strip():
                    story.append(Paragraph(para, styles["Normal"]))
            story.append(Spacer(1, 0.3*cm))

        # Disclaimer
        story.append(Spacer(1, 0.5*cm))
        disc_style = styles["Normal"]
        story.append(Paragraph(f"<i>{report.get('disclaimer', '')}</i>", disc_style))

        doc.build(story)
        return {"status": "success", "path": str(out)}

    def save_json_report(self, report: dict, output_path: str) -> dict:
        """Save the raw report dict as a JSON file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        return {"status": "success", "path": str(out)}
