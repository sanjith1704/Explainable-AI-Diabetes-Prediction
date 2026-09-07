"""
pdf_report.py: Professional PDF clinical assessment report generator using ReportLab.
Produces a structured, research-grade document with patient inputs, model attributions,
personalized SHAP-based educational guidance, AI narrative synthesis, and mandatory disclaimers.

PDF Structure:
TITLE: Explainable AI Diabetes Risk Assessment
Subtitle: Academic Research Prototype
1. Patient Information
2. Clinical Measurements
3. Risk Assessment
4. Top Influential Factors (Local SHAP Table + Chart)
5. Personalized Model-Based Summary
6. Personalized Educational Guidance & Wellness Recommendations
7. General Preventive Wellness Guidance
8. AI Natural Language Synthesis
9. Important Limitations & Disclaimer
"""

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from services.ai_report import MANDATORY_DISCLAIMER
from services.recommendations import generate_personalized_recommendations


def generate_pdf_report(
    patient_data: Dict[str, Any],
    prediction: int,
    probability: float,
    top_shap_features: List[Dict[str, Any]],
    model_info: Optional[Dict[str, Any]] = None,
    interpretation_text: str = "",
    recommendations: Optional[List[str]] = None,
    chart_image_path: Optional[str] = None,
    output_path: Optional[str] = None,
    personalized_guidance: Optional[Dict[str, Any]] = None,
) -> bytes:
    """
    Generates a structured, professional 9-section PDF Diabetes Risk Assessment Report.
    Returns the generated PDF as bytes, and optionally saves to output_path.
    """
    if model_info is None:
        model_info = {"model_name": "XGBoost (Calibrated)", "roc_auc": 0.8198}

    if personalized_guidance is None:
        personalized_guidance = generate_personalized_recommendations(
            patient_data=patient_data,
            shap_features=top_shap_features,
            prediction_probability=probability,
        )

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=2,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#475569"),
        spaceAfter=6,
    )
    section_title = ParagraphStyle(
        "SectionTitle",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#0f766e"),
        spaceBefore=8,
        spaceAfter=4,
    )
    sub_section_title = ParagraphStyle(
        "SubSectionTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=4,
        spaceAfter=2,
    )
    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#334155"),
    )
    body_bold = ParagraphStyle(
        "DocBodyBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1e293b"),
    )
    bullet_style = ParagraphStyle(
        "DocBullet",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11.5,
        textColor=colors.HexColor("#334155"),
        leftIndent=10,
        spaceAfter=2,
    )
    caption_style = ParagraphStyle(
        "DocCaption",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#64748b"),
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#1e293b"),
    )
    table_cell_center = ParagraphStyle(
        "TableCellCenter",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10.5,
        alignment=1,
        textColor=colors.HexColor("#1e293b"),
    )
    table_header = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10.5,
        alignment=1,
        textColor=colors.white,
    )
    disclaimer_style = ParagraphStyle(
        "DisclaimerText",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=10.5,
        textColor=colors.HexColor("#991b1b"),
        alignment=1,
    )

    story = []

    # Title & Subtitle
    story.append(Paragraph("Explainable AI Diabetes Risk Assessment", title_style))
    report_time = datetime.now().strftime("%B %d, %Y - %H:%M:%S")
    story.append(
        Paragraph(
            f"Academic Research Prototype | Assessment Date: {report_time}",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0f766e"), spaceAfter=8))

    # 1. Patient Information
    p_name = str(patient_data.get("Name") or "Not Provided").strip()
    if not p_name:
        p_name = "Not Provided"
    p_sex = str(patient_data.get("Sex") or patient_data.get("Gender") or "Not specified / Prefer not to say").strip()
    p_age = str(patient_data.get("Age", "N/A"))

    story.append(Paragraph("1. Patient Information", section_title))
    patient_info_data = [
        [
            Paragraph(f"<b>Patient Name:</b> {p_name}", body_style),
            Paragraph(f"<b>Biological Sex:</b> {p_sex}", body_style),
        ],
        [
            Paragraph(f"<b>Age:</b> {p_age} years", body_style),
            Paragraph(f"<b>Assessment Date:</b> {report_time}", body_style),
        ],
    ]
    patient_info_table = Table(patient_info_data, colWidths=[270, 270])
    patient_info_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(patient_info_table)
    story.append(Spacer(1, 6))

    # 2. Clinical Measurements
    story.append(Paragraph("2. Clinical Measurements", section_title))
    measurements_data = [
        [
            Paragraph(f"<b>Pregnancies:</b> {patient_data.get('Pregnancies', 0)}", body_style),
            Paragraph(f"<b>Glucose:</b> {patient_data.get('Glucose', 'N/A')} mg/dL", body_style),
        ],
        [
            Paragraph(f"<b>Blood Pressure:</b> {patient_data.get('BloodPressure', 'N/A')} mmHg", body_style),
            Paragraph(f"<b>Skin Thickness:</b> {patient_data.get('SkinThickness', 'N/A')} mm", body_style),
        ],
        [
            Paragraph(f"<b>Insulin:</b> {patient_data.get('Insulin', 'N/A')} muU/mL", body_style),
            Paragraph(f"<b>BMI:</b> {patient_data.get('BMI', 'N/A')} kg/m²", body_style),
        ],
        [
            Paragraph(f"<b>Diabetes Pedigree Function:</b> {patient_data.get('DiabetesPedigreeFunction', 'N/A')}", body_style),
            Paragraph("<i>(Patient Age recorded in Patient Information above)</i>", caption_style),
        ],
    ]
    measurements_table = Table(measurements_data, colWidths=[270, 270])
    measurements_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(measurements_table)
    story.append(Spacer(1, 6))

    # 3. Risk Assessment Box
    story.append(Paragraph("3. Risk Assessment", section_title))
    risk_text = "ELEVATED / HIGHER RISK (Class 1)" if prediction == 1 else "STANDARD / LOWER RISK (Class 0)"
    risk_bg = colors.HexColor("#fef2f2") if prediction == 1 else colors.HexColor("#f0fdf4")
    risk_border = colors.HexColor("#ef4444") if prediction == 1 else colors.HexColor("#22c55e")
    risk_color = "#b91c1c" if prediction == 1 else "#15803d"

    prob_pct = f"{probability * 100:.2f}%"
    model_name = model_info.get("model_name", "XGBoost")
    roc_auc = model_info.get("roc_auc", 0.8198)

    risk_html = f"""
    <b>Risk Classification:</b> <font color="{risk_color}"><b>{risk_text}</b></font><br/>
    <b>Estimated Probability:</b> {prob_pct} &nbsp;|&nbsp; <b>Decision Threshold:</b> 50.0%<br/>
    <b>Evaluated Architecture:</b> {model_name} (Holdout ROC-AUC: {roc_auc:.4f})
    """
    risk_table = Table([[Paragraph(risk_html, body_style)]], colWidths=[540])
    risk_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), risk_bg),
                ("BOX", (0, 0), (-1, -1), 1.25, risk_border),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(risk_table)
    story.append(Spacer(1, 6))

    # 4. Top Influential Factors (SHAP Local Attribution)
    story.append(Paragraph("4. Top Influential Factors", section_title))
    story.append(
        Paragraph(
            "Local SHAP (SHapley Additive exPlanations) features ranked by contribution to this prediction:",
            body_style,
        )
    )
    story.append(Spacer(1, 3))

    shap_rows = [
        [
            Paragraph("Rank", table_header),
            Paragraph("Clinical Parameter", table_header),
            Paragraph("Patient Value", table_header),
            Paragraph("SHAP Attribution", table_header),
            Paragraph("Directional Impact on Model", table_header),
        ]
    ]
    for idx, item in enumerate(top_shap_features[:5], start=1):
        feat = item.get("Feature", "N/A")
        val = str(item.get("Patient_Value", patient_data.get(feat, "N/A")))
        shap_val = float(item.get("SHAP_Value", 0.0))

        if shap_val > 0:
            direction_str = "Pushes model prediction toward diabetes-positive class"
            direction_color = "#b91c1c"
        elif shap_val < 0:
            direction_str = "Pushes model prediction toward diabetes-negative class"
            direction_color = "#15803d"
        else:
            direction_str = "Neutral impact on prediction"
            direction_color = "#475569"

        direction_html = f'<font color="{direction_color}">{direction_str}</font>'

        shap_rows.append(
            [
                Paragraph(str(idx), table_cell_center),
                Paragraph(f"<b>{feat}</b>", table_cell),
                Paragraph(val, table_cell_center),
                Paragraph(f"{shap_val:+.4f}", table_cell_center),
                Paragraph(direction_html, table_cell),
            ]
        )

    shap_table = Table(shap_rows, colWidths=[35, 120, 75, 90, 220])
    shap_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("PADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(shap_table)
    story.append(Spacer(1, 3))
    story.append(
        Paragraph(
            "<b>Interpretability Note:</b> SHAP values describe how individual features influenced this model's "
            "prediction for this patient. They do not establish biological causation or clinical importance.",
            caption_style,
        )
    )
    story.append(Spacer(1, 6))

    # Embedded graphic (if provided)
    if chart_image_path and Path(chart_image_path).exists():
        try:
            story.append(Image(chart_image_path, width=480, height=160))
            story.append(Spacer(1, 6))
        except Exception:
            pass

    # 5. Personalized Model-Based Summary
    story.append(Paragraph("5. Personalized Model-Based Summary", section_title))
    story.append(Paragraph(personalized_guidance["summary_text"], body_style))
    story.append(Spacer(1, 4))

    if personalized_guidance["priority_areas"]:
        story.append(Paragraph("<b>Highest-Priority Areas for this Model Prediction:</b>", sub_section_title))
        for idx, area in enumerate(personalized_guidance["priority_areas"], start=1):
            mod_tag = "Modifiable factor" if area["is_modifiable"] else "Non-modifiable factor"
            story.append(
                Paragraph(
                    f"<b>{idx}. {area['display_name']} ({area['feature']})</b> — Patient Value: {area['patient_value']} "
                    f"| Positive SHAP Contribution: <b>{area['shap_value']:+.4f}</b> [<i>{mod_tag}</i>]",
                    body_bold,
                )
            )
            story.append(
                Paragraph(f"<i>Why it matters to the model:</i> {area['why_it_matters']}", bullet_style)
            )
        story.append(Spacer(1, 4))

    if personalized_guidance.get("negative_contributors_summary"):
        story.append(Paragraph("<b>Factors That Lowered Model-Estimated Risk:</b>", sub_section_title))
        for item in personalized_guidance["negative_contributors_summary"][:3]:
            story.append(Paragraph(f"• {item}", bullet_style))
        story.append(Spacer(1, 6))

    # 6. Personalized Educational Guidance & Wellness Recommendations
    story.append(
        Paragraph(
            "6. Personalized Educational Guidance & Wellness Recommendations",
            section_title,
        )
    )
    if personalized_guidance["priority_areas"]:
        for area in personalized_guidance["priority_areas"]:
            story.append(
                Paragraph(
                    f"<b>Educational Guidance for {area['display_name']}:</b>",
                    sub_section_title,
                )
            )
            for bullet in area["guidance"]:
                story.append(Paragraph(f"• {bullet}", bullet_style))
    else:
        story.append(
            Paragraph(
                "No features exerted a positive local push toward higher risk. Standard preventive recommendations apply.",
                bullet_style,
            )
        )
    story.append(Spacer(1, 6))

    # 7. General Preventive Wellness Guidance
    story.append(Paragraph("7. General Preventive Wellness Guidance", section_title))
    for rec in personalized_guidance.get("general_wellness", []):
        story.append(Paragraph(f"• {rec}", bullet_style))
    story.append(Spacer(1, 6))

    # 8. AI Natural Language Synthesis
    story.append(Paragraph("8. AI Natural Language Synthesis", section_title))
    story.append(Paragraph(interpretation_text or "Computational prediction synthesized from model feature weights.", body_style))
    story.append(Spacer(1, 8))

    # 9. Important Limitations & Disclaimer (Kept together)
    disclaimer_box = []
    disclaimer_box.append(Paragraph("9. Important Limitations & Disclaimer", section_title))
    disclaimer_box.append(
        Paragraph(
            "<b>Clinical Limitations:</b> This assessment is produced by an academic machine learning research "
            "prototype trained on the historical PIMA Indians Diabetes Dataset. It is not an FDA-cleared or CE-marked "
            "diagnostic medical device. Prediction scores describe statistical pattern associations and should never "
            "be interpreted as clinical certainty. Patients must not initiate, alter, or discontinue any medications "
            "or treatments without direct consultation with a qualified medical professional.",
            body_style,
        )
    )
    disclaimer_box.append(Spacer(1, 4))
    disclaimer_box.append(
        Table(
            [[Paragraph(f"<b>MANDATORY MEDICAL NOTICE:</b> {MANDATORY_DISCLAIMER}", disclaimer_style)]],
            colWidths=[540],
            style=[
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff1f2")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#f43f5e")),
                ("PADDING", (0, 0), (-1, -1), 6),
            ],
        )
    )
    story.append(KeepTogether(disclaimer_box))

    # Build Document
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "wb") as f:
            f.write(pdf_bytes)

    return pdf_bytes
