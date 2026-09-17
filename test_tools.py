from time_parser import parse_persian_datetime
from scheduler import schedule_email_job


run_date = parse_persian_datetime(
    "2 دقیقه بعد"
)

result = schedule_email_job(
    to="hemmatzahraaa@gmail.com",
    subject="تست زمان‌بندی",
    body="این یک تست Scheduler است.",
    run_date=run_date
)

print(result)

input(
    "Scheduler در حال اجراست. "
    "برای خروج Enter بزنید..."
)