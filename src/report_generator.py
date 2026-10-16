"""
GlucoGuard: Multi-Factor Diabetes Risk & Early Prevention Intelligence Engine
Module: Clean Clinical Diagnostic PDF Report Generator (src/report_generator.py)
Uses lightweight, ASCII-safe fpdf2 with explicit horizontal margins to prevent
'Not enough horizontal space to render a single character' exceptions.
"""

import io
from datetime import datetime
from typing import Dict, Any, List, Optional

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False


class CleanClinicalPDF(FPDF):
    def header(self):
        self.set_fill_color(13, 148, 136)  # Medical Teal
        self.rect(0, 0, 210, 18, 'F')
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(255, 255, 255)
        self.set_xy(10, 4)
        self.cell(190, 10, 'GLUCOGUARD CLINICAL INTELLIGENCE REPORT', align='C')
        self.set_xy(10, 22)

    def footer(self):
        self.set_y(-12)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(148, 163, 184)
        self.set_x(10)
        self.cell(190, 6, f'Confidential Medical Decision Support Document | Generated {datetime.now().strftime("%Y-%m-%d %H:%M")} | Page {self.page_no()}', align='C')


def generate_patient_pdf(
    patient_data: dict, 
    risk_prob: float, 
    risk_tier: str, 
    key_factors: list = None,
    patient_meta: dict = None
) -> bytes:
    """
    Generates an executive-grade single-page clinical PDF report
    with pure ASCII characters, patient demographic intake records,
    and explicit cell widths to prevent horizontal space layout errors.
    Returns bytes suitable for Streamlit st.download_button.
    """
    if not FPDF_AVAILABLE:
        raise RuntimeError("fpdf2 library is not installed. Please run: pip install fpdf2")

    if patient_meta is None:
        patient_meta = {}

    pdf = CleanClinicalPDF(orientation='P', unit='mm', format='A4')
    pdf.set_margins(left=10, top=10, right=10)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    epw = 190.0  # Exact printable width: 210mm - 20mm margins
    
    # ---------------------------------------------------------
    # 1. Report Title & Meta
    # ---------------------------------------------------------
    pdf.set_xy(10, 22)
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(epw, 6, 'Patient Metabolic Evaluation & Risk Stratification', ln=1)
    
    pdf.set_x(10)
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(100, 116, 139)
    report_id = patient_meta.get('mrn') or f"GG-{datetime.now().strftime('%y%m%d')}-{abs(hash(str(patient_data))) % 10000:04d}"
    pdf.cell(epw, 4.5, f'Evaluation ID: {report_id}  |  Assessment Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}', ln=1)
    pdf.ln(2)

    # ---------------------------------------------------------
    # 2. Patient Demographics & Identification Block
    # ---------------------------------------------------------
    demo_y = pdf.get_y()
    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(10, demo_y, epw, 17, 'DF')
    
    # Demographics Row 1
    pdf.set_xy(14, demo_y + 2)
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(24, 4.5, 'Patient Name:')
    pdf.set_font('Helvetica', 'B', 8.5)
    pdf.set_text_color(15, 23, 42)
    p_name = str(patient_meta.get('name', 'Jane Doe'))[:26]
    pdf.cell(46, 4.5, p_name)
    
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(22, 4.5, 'Patient ID:')
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(35, 4.5, str(report_id)[:20])
    
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(24, 4.5, 'Blood Group:')
    pdf.set_font('Helvetica', 'B', 8.5)
    bgroup = patient_meta.get('blood_group', 'Not Specified')
    if bgroup and bgroup != "Not Specified":
        pdf.set_text_color(225, 29, 72)  # Highlight blood type in clinical crimson
    else:
        pdf.set_text_color(100, 116, 139)
    pdf.cell(25, 4.5, str(bgroup), ln=1)
    
    # Demographics Row 2
    pdf.set_xy(14, demo_y + 8.5)
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(24, 4.5, 'Age / Sex:')
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(15, 23, 42)
    age_val = int(patient_data.get('Age', patient_meta.get('age', 38)))
    gender_val = str(patient_meta.get('gender', 'Female'))
    pdf.cell(46, 4.5, f"{age_val} Years  /  {gender_val}")
    
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(22, 4.5, 'Physician:')
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(15, 23, 42)
    phys_val = str(patient_meta.get('physician', 'Clinical Decision Support'))[:20]
    pdf.cell(35, 4.5, phys_val)
    
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(24, 4.5, 'Intake Date:')
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(25, 4.5, datetime.now().strftime("%Y-%m-%d"), ln=1)
    
    # Reset cursor below demographic box
    pdf.set_xy(10, demo_y + 20)
    
    # ---------------------------------------------------------
    # 3. Executive Risk Summary Box
    # ---------------------------------------------------------
    box_y = pdf.get_y()
    pdf.set_fill_color(241, 245, 249)
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(10, box_y, epw, 16, 'DF')
    
    pdf.set_xy(14, box_y + 2.5)
    pdf.set_font('Helvetica', 'B', 9.5)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(42, 5.5, 'DIAGNOSTIC STATUS:')
    
    # Color-coded tier text
    if 'High' in risk_tier or 'ALERT' in risk_tier.upper():
        pdf.set_text_color(225, 29, 72)  # Red
    elif 'Borderline' in risk_tier or 'Moderate' in risk_tier:
        pdf.set_text_color(217, 119, 6)   # Amber
    else:
        pdf.set_text_color(13, 148, 136)  # Teal
    pdf.cell(50, 5.5, risk_tier.upper())
    
    pdf.set_text_color(15, 23, 42)
    pdf.cell(40, 5.5, 'RISK PROBABILITY:')
    pdf.set_font('Helvetica', 'B', 10.5)
    prob_val = risk_prob if risk_prob <= 1.0 else (risk_prob / 100.0)
    pdf.cell(35, 5.5, f'{prob_val * 100:.2f}%')
    
    # Reset cursor cleanly below summary box
    pdf.set_xy(10, box_y + 19)
    
    # ---------------------------------------------------------
    # 4. Patient Biomarkers Table
    # ---------------------------------------------------------
    pdf.set_font('Helvetica', 'B', 9.5)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(epw, 5.5, 'Recorded Clinical Biomarkers', ln=1)
    
    # Table Header
    pdf.set_x(10)
    pdf.set_fill_color(15, 23, 42)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 8)
    pdf.cell(65, 6, ' Parameter / Biomarker', 1, 0, 'L', True)
    pdf.cell(38, 6, ' Observed Value', 1, 0, 'C', True)
    pdf.cell(87, 6, ' Clinical Reference Range', 1, 1, 'L', True)
    
    # Table Rows
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(30, 41, 59)
    ref_ranges = {
        'Glucose': ('Fasting Plasma Glucose', f"{patient_data.get('Glucose', 0):.1f} mg/dL", 'Normal: 70 - 99 mg/dL | Pre-DM: 100 - 125'),
        'BloodPressure': ('Diastolic Blood Pressure', f"{patient_data.get('BloodPressure', 0):.1f} mmHg", 'Normal: <80 mmHg | Elevated: 80 - 89'),
        'BMI': ('Body Mass Index (BMI)', f"{patient_data.get('BMI', 0):.1f} kg/m2", 'Normal: 18.5 - 24.9 | Overweight: 25 - 29.9'),
        'Insulin': ('2-Hour Serum Insulin', f"{patient_data.get('Insulin', 0):.1f} uIU/mL", 'Normal: 16 - 166 uIU/mL'),
        'SkinThickness': ('Triceps Skinfold Thickness', f"{patient_data.get('SkinThickness', 0):.1f} mm", 'Median: 20 - 35 mm (Subcutaneous fat)'),
        'Age': ('Patient Age', f"{int(patient_data.get('Age', 0))} Years", 'Metabolic risk increases above 45 yrs'),
        'DiabetesPedigreeFunction': ('Genetic Pedigree (DPF)', f"{patient_data.get('DiabetesPedigreeFunction', 0):.2f}", 'Cohort Baseline Median: 0.37'),
        'Pregnancies': ('Gravidity / Pregnancies', f"{int(patient_data.get('Pregnancies', 0))}", 'Gestational stress indicator')
    }
    
    fill = False
    for k, (name, val, ref) in ref_ranges.items():
        pdf.set_x(10)
        pdf.set_fill_color(248, 250, 252) if fill else pdf.set_fill_color(255, 255, 255)
        pdf.cell(65, 5, f'  {name}', 1, 0, 'L', fill)
        pdf.cell(38, 5, val, 1, 0, 'C', fill)
        pdf.cell(87, 5, f'  {ref}', 1, 1, 'L', fill)
        fill = not fill
        
    pdf.ln(2.5)
    
    # ---------------------------------------------------------
    # 5. Clinical Recommendations
    # ---------------------------------------------------------
    pdf.set_x(10)
    pdf.set_font('Helvetica', 'B', 9.5)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(epw, 5.5, 'Clinical Decision Support Interventions', ln=1)
    
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(51, 65, 85)
    
    rec_points = [
        "- Fasting glycemic evaluation: Schedule laboratory HbA1c screening within 30 to 60 days.",
        "- Nutritional intervention: Restrict simple sugars; target Mediterranean low-glycemic dietary protocols.",
        "- Physical conditioning: 150 min/week moderate aerobic exertion + 2 resistance training sessions.",
        "- Biometric monitoring: Re-assess blood pressure and fasting blood glucose bimonthly."
    ]
    for pt in rec_points:
        pdf.set_x(10)
        pdf.multi_cell(epw, 4.2, pt)
        
    pdf.ln(2.5)
    
    # ---------------------------------------------------------
    # 6. Regulatory Disclaimer
    # ---------------------------------------------------------
    pdf.set_x(10)
    pdf.set_font('Helvetica', 'I', 7)
    pdf.set_text_color(148, 163, 184)
    pdf.multi_cell(epw, 3.5, 'Disclaimer: GlucoGuard AI is a Clinical Decision Support System intended for screening and research risk assessment. This report does not constitute a definitive medical diagnosis. Clinical validation by a licensed medical practitioner is required.')
    
    return bytes(pdf.output())


# Backwards compatibility alias
def generate_patient_pdf_report(
    patient_data: dict, 
    risk_score_pct: float, 
    risk_tier: str, 
    contributions: list = None, 
    interventions: list = None,
    patient_meta: dict = None
) -> bytes:
    prob = risk_score_pct / 100.0 if risk_score_pct > 1.0 else risk_score_pct
    return generate_patient_pdf(
        patient_data=patient_data, 
        risk_prob=prob, 
        risk_tier=risk_tier, 
        key_factors=contributions,
        patient_meta=patient_meta
    )
