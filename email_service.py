import os
import smtplib
from dotenv import load_dotenv
from email.message import EmailMessage


load_dotenv()

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")

def send_email_smtp(
    to: str,
    subject: str,
    body: str
) -> str:

    if not EMAIL_ADDRESS:
        return "EMAIL_ADDRESS is not configured."

    if not EMAIL_APP_PASSWORD:
        return "EMAIL_APP_PASSWORD is not configured."

    try:
        msg = EmailMessage()

        msg["From"] = EMAIL_ADDRESS
        msg["To"] = to
        msg["Subject"] = subject

        msg.set_content(body)

        with smtplib.SMTP_SSL(
            "smtp.gmail.com",
            465
        ) as smtp:

            smtp.login(
                EMAIL_ADDRESS,
                EMAIL_APP_PASSWORD
            )

            smtp.send_message(msg)

        print(
            f"========== EMAIL SENT ==========\n"
            f"To: {to}\n"
            f"Subject: {subject}\n"
            f"================================"
        )

        return f"Email successfully sent to {to}."

    except Exception as e:

        print("EMAIL ERROR:", e)

        return f"Failed to send email: {str(e)}"