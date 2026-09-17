from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from email_service import send_email_smtp


scheduler = BackgroundScheduler(
    timezone="Asia/Tehran"
)

scheduler.start()


def schedule_email_job(
    to: str,
    subject: str,
    body: str,
    send_at: str
) -> str:

    try:
        run_date = datetime.fromisoformat(send_at)

    except ValueError:
        return (
            "زمان نامعتبر است. "
            "زمان باید ISO-8601 باشد."
        )

    job = scheduler.add_job(
        func=send_email_smtp,
        trigger="date",
        run_date=run_date,
        kwargs={
            "to": to,
            "subject": subject,
            "body": body,
        },
        misfire_grace_time=3600,
    )

    return (
        "ایمیل با موفقیت زمان‌بندی شد.\n"
        f"Job ID: {job.id}\n"
        f"زمان ارسال: {run_date}"
    )