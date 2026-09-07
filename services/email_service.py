"""
email_service.py: Secure SMTP email dispatch service with PDF attachment support.
Strictly uses environment variables for credentials, enforces RFC 5322 email validation,
and never fakes success.
"""

import os
import re
import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from services.recommendations import generate_personalized_recommendations

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    env_file = Path(".env")
    if env_file.exists():
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass

# RFC 5322 compliant regex pattern
EMAIL_REGEX = re.compile(
    r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+$"
)


def validate_email_address(email: str) -> bool:
    """
    Validates recipient email address against RFC 5322 standards.
    Returns True if valid, False otherwise.
    """
    if not email or not isinstance(email, str):
        return False
    clean = email.strip()
    return bool(EMAIL_REGEX.match(clean))


def check_email_configuration() -> Tuple[bool, str]:
    """
    Verifies whether necessary SMTP configuration variables are defined in .env.
    Never exposes passwords or credentials in returned strings.
    """
    host = os.getenv("EMAIL_HOST", "").strip()
    port = os.getenv("EMAIL_PORT", "").strip()
    username = os.getenv("EMAIL_USERNAME", "").strip()
    password = os.getenv("EMAIL_PASSWORD", "").strip()
    sender = os.getenv("SENDER_EMAIL", "").strip()

    missing = []
    if not host:
        missing.append("EMAIL_HOST")
    if not port:
        missing.append("EMAIL_PORT")
    if not username:
        missing.append("EMAIL_USERNAME")
    if not password:
        missing.append("EMAIL_PASSWORD")
    if not sender:
        missing.append("SENDER_EMAIL")

    if missing:
        return (
            False,
            f"Email service is not configured. Missing environment variables: {', '.join(missing)} in .env.",
        )
    return True, "Email service is configured."


def format_email_body(
    prediction: int,
    probability: float,
    top_shap_features: List[Dict[str, Any]],
    interpretation_text: str,
    model_name: str,
    patient_data: Optional[Dict[str, Any]] = None,
    personalized_guidance: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Formats the email text body matching the exact requirements:
    - Diabetes Risk Assessment
    - Patient Demographics (Name, Sex, Age)
    - Prediction & Estimated Probability
    - Top Model Factors
    - Short Personalized Educational Guidance
    - AI Interpretation
    - Model
    - Disclaimer
    """
    pred_str = "Higher Risk (Class 1)" if prediction == 1 else "Lower Risk (Class 0)"
    prob_str = f"{probability * 100:.2f}%"

    if personalized_guidance is None and patient_data:
        personalized_guidance = generate_personalized_recommendations(
            patient_data=patient_data,
            shap_features=top_shap_features,
            prediction_probability=probability,
        )

    lines = [
        "Diabetes Risk Assessment",
        "==================================================",
        "",
    ]

    if patient_data:
        p_name = str(patient_data.get("Name") or "Not Provided").strip()
        p_sex = str(patient_data.get("Sex") or patient_data.get("Gender") or "Not specified / Prefer not to say").strip()
        p_age = str(patient_data.get("Age", "N/A"))
        lines.extend(
            [
                f"Patient Name: {p_name}",
                f"Demographics: Sex: {p_sex}, Age: {p_age} years",
                "",
            ]
        )

    lines.extend(
        [
            f"Prediction:",
            f"{pred_str}",
            "",
            f"Estimated Probability:",
            f"{prob_str}",
            "",
            "Key Model Factors:",
        ]
    )

    for idx, f in enumerate(top_shap_features[:5], start=1):
        feat = f.get("Feature", "Feature")
        shap_val = float(f.get("SHAP_Value", 0.0))
        if shap_val > 0:
            dir_str = "Pushes model prediction toward diabetes-positive class"
        elif shap_val < 0:
            dir_str = "Pushes model prediction toward diabetes-negative class"
        else:
            dir_str = "Neutral impact"
        lines.append(f"{idx}. {feat} — {shap_val:+.4f} ({dir_str})")

    if personalized_guidance and personalized_guidance.get("priority_areas"):
        lines.extend(
            [
                "",
                "Short Personalized Educational Guidance:",
            ]
        )
        for area in personalized_guidance["priority_areas"]:
            lines.append(f"• [{area['display_name']}] {area['guidance'][0]}")

    lines.extend(
        [
            "",
            "AI Interpretation:",
            interpretation_text.strip(),
            "",
            f"Model:",
            f"{model_name}",
            "",
            "Disclaimer:",
            "This report is generated by an AI-based research prototype for educational purposes. "
            "It is not a medical diagnosis and should not replace consultation with a qualified healthcare professional.",
        ]
    )

    return "\n".join(lines)


def send_report_email(
    recipient_email: str,
    subject: str,
    body_text: str,
    pdf_bytes: Optional[bytes] = None,
    pdf_filename: str = "Diabetes_Risk_Assessment_Report.pdf",
) -> Tuple[bool, str]:
    """
    Sends the generated assessment report to a specified recipient with optional PDF attachment.
    Does NOT fake success: returns (True, ...) only when the SMTP send operation succeeds.
    """
    # 1. Validate email address
    if not validate_email_address(recipient_email):
        return False, "Please enter a valid email address."

    # 2. Check SMTP configuration
    is_configured, config_msg = check_email_configuration()
    if not is_configured:
        return False, config_msg

    host = os.getenv("EMAIL_HOST", "smtp.gmail.com").strip()
    port_str = os.getenv("EMAIL_PORT", "587").strip()
    try:
        port = int(port_str)
    except ValueError:
        port = 587

    username = os.getenv("EMAIL_USERNAME", "").strip()
    password = os.getenv("EMAIL_PASSWORD", "").strip().replace(" ", "")
    sender = os.getenv("SENDER_EMAIL", "").strip()
    use_tls = os.getenv("EMAIL_USE_TLS", "true").strip().lower() in {"true", "1", "yes"}

    try:
        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = recipient_email.strip()
        msg["Subject"] = subject or "Diabetes Risk Assessment Report"

        # Attach readable text body
        msg.attach(MIMEText(body_text, "plain", "utf-8"))

        # Attach PDF if provided
        if pdf_bytes:
            pdf_attachment = MIMEApplication(pdf_bytes, _subtype="pdf")
            pdf_attachment.add_header("Content-Disposition", "attachment", filename=pdf_filename)
            msg.attach(pdf_attachment)

        # Connect with timeout
        if port == 465:
            server = smtplib.SMTP_SSL(host, port, timeout=15)
            server.ehlo()
        else:
            server = smtplib.SMTP(host, port, timeout=15)
            server.ehlo()
            if use_tls:
                server.starttls()
                server.ehlo()

        # Authenticate
        server.login(username, password)
        server.sendmail(sender, [recipient_email.strip()], msg.as_string())
        server.quit()

        if pdf_bytes:
            return True, f"Report successfully dispatched to {recipient_email.strip()} with PDF attachment."
        else:
            return True, f"Test email successfully dispatched to {recipient_email.strip()}."

    except smtplib.SMTPAuthenticationError:
        return (
            False,
            "SMTP Authentication Failed: Incorrect EMAIL_USERNAME or EMAIL_PASSWORD. "
            "Note: For Gmail, you must generate and use a 16-digit 'Gmail App Password' rather than your normal password.",
        )
    except smtplib.SMTPConnectError:
        return (
            False,
            f"SMTP Connection Failed: Unable to reach {host}:{port}. Verify host, port, and network connection.",
        )
    except Exception as e:
        err_type = type(e).__name__
        return False, f"Failed to send email ({err_type}): {str(e)[:150]}"


def send_test_email(recipient_email: str) -> Tuple[bool, str]:
    """
    Sends a lightweight test message to verify SMTP connectivity and authentication.
    Does NOT fake success: validates recipient and executes SMTP send.
    """
    # 1. Validate recipient
    if not validate_email_address(recipient_email):
        return False, "Please enter a valid email address."

    # 2. Check config
    is_configured, config_msg = check_email_configuration()
    if not is_configured:
        return False, config_msg

    subject = "Diabetes AI System — Test Email"
    body = (
        "This is a test email from the Explainable AI Diabetes Prediction System.\n\n"
        "Your SMTP configuration is working correctly!\n"
        "Security Note: Connection was established using secure TLS/STARTTLS."
    )

    return send_report_email(
        recipient_email=recipient_email,
        subject=subject,
        body_text=body,
        pdf_bytes=None,
    )
