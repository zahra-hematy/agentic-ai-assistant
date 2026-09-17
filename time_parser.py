import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import jdatetime


TEHRAN_TZ = ZoneInfo("Asia/Tehran")


PERSIAN_MONTHS = {
    "فروردین": 1,
    "اردیبهشت": 2,
    "خرداد": 3,
    "تیر": 4,
    "مرداد": 5,
    "شهریور": 6,
    "مهر": 7,
    "آبان": 8,
    "آذر": 9,
    "دی": 10,
    "بهمن": 11,
    "اسفند": 12,
}


def normalize_digits(text: str) -> str:
    """
    Convert Persian and Arabic digits to English digits.
    """

    translation_table = str.maketrans(
        "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩",
        "01234567890123456789"
    )

    return text.translate(translation_table)


def now_tehran() -> datetime:
    """
    Current time in Tehran timezone.
    """

    return datetime.now(TEHRAN_TZ)


def parse_time(text: str):
    """
    Extract time from Persian text.

    Supported examples:

    ساعت 10
    ساعت 10:30
    ساعت 10 شب
    ساعت 8 صبح
    ساعت 14:30
    """

    text = normalize_digits(text)

    pattern = (
        r"(?:ساعت\s*)?"
        r"(\d{1,2})"
        r"(?:[:٫](\d{1,2}))?"
        r"\s*"
        r"(صبح|ظهر|عصر|شب)?"
    )

    matches = list(re.finditer(pattern, text))

    if not matches:
        return None

    # Prefer a match that appears after "ساعت"
    match = None

    for item in matches:
        start = item.start()

        if "ساعت" in text[max(0, start - 6):start]:
            match = item
            break

    if match is None:
        match = matches[0]

    hour = int(match.group(1))

    minute = (
        int(match.group(2))
        if match.group(2)
        else 0
    )

    period = match.group(3)

    if period == "صبح":

        if hour == 12:
            hour = 0

    elif period in ["ظهر", "عصر", "شب"]:

        if hour < 12:
            hour += 12

    if hour > 23:
        raise ValueError(f"ساعت نامعتبر است: {hour}")

    if minute > 59:
        raise ValueError(f"دقیقه نامعتبر است: {minute}")

    return hour, minute


def parse_relative_time(text: str):
    """
    Parse relative expressions.

    Examples:

    دو ساعت بعد
    2 ساعت بعد
    30 دقیقه بعد
    1 روز بعد
    """

    text = normalize_digits(text)

    pattern = (
        r"(\d+(?:\.\d+)?)\s*"
        r"(ثانیه|دقیقه|ساعت|روز)"
        r"\s*بعد"
    )

    match = re.search(pattern, text)

    if not match:
        return None

    amount = float(match.group(1))
    unit = match.group(2)

    now = now_tehran()

    if unit == "ثانیه":
        result = now + timedelta(seconds=amount)

    elif unit == "دقیقه":
        result = now + timedelta(minutes=amount)

    elif unit == "ساعت":
        result = now + timedelta(hours=amount)

    elif unit == "روز":
        result = now + timedelta(days=amount)

    else:
        return None

    return result


def parse_jalali_date(text: str):
    """
    Parse Jalali dates.

    Examples:

    1405/06/25
    1405-06-25
    25 شهریور 1405
    25 شهریور
    """

    text = normalize_digits(text)

    # -----------------------------------------
    # Format: 1405/06/25
    # -----------------------------------------

    numeric_pattern = (
        r"(?<!\d)"
        r"(13\d{2}|14\d{2})"
        r"[/\-]"
        r"(\d{1,2})"
        r"[/\-]"
        r"(\d{1,2})"
        r"(?!\d)"
    )

    match = re.search(
        numeric_pattern,
        text
    )

    if match:

        year = int(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3))

        return jdatetime.date(
            year,
            month,
            day
        ).togregorian()

    # -----------------------------------------
    # Format: 25 شهریور 1405
    # -----------------------------------------

    month_names = "|".join(
        PERSIAN_MONTHS.keys()
    )

    named_pattern = (
        rf"(\d{{1,2}})\s*"
        rf"({month_names})"
        rf"(?:\s*(13\d{{2}}|14\d{{2}}))?"
    )

    match = re.search(
        named_pattern,
        text
    )

    if not match:
        return None

    day = int(match.group(1))
    month_name = match.group(2)

    year_text = match.group(3)

    month = PERSIAN_MONTHS[month_name]

    if year_text:
        year = int(year_text)

    else:
        # If year is not specified,
        # assume current Jalali year.
        now = now_tehran()

        jalali_now = jdatetime.datetime.fromgregorian(
            datetime=now.replace(tzinfo=None)
        )

        year = jalali_now.year

    return jdatetime.date(
        year,
        month,
        day
    ).togregorian()


def parse_persian_datetime(text: str) -> datetime:
    """
    Convert a Persian natural-language time expression
    into a timezone-aware datetime in Tehran timezone.

    Supported examples:

    همین الان
    الان
    دو ساعت بعد
    30 دقیقه بعد
    فردا ساعت 10 شب
    پس فردا ساعت 8 صبح
    امروز ساعت 22
    25 شهریور 1405 ساعت 9
    1405/06/25 ساعت 09:30
    """

    if not text:
        raise ValueError("زمان ارسال مشخص نشده است.")

    text = text.strip()
    normalized = normalize_digits(text)

    now = now_tehran()

    # =========================================
    # 1. NOW
    # =========================================

    immediate_words = [
        "همین الان",
        "الان",
        "فورا",
        "فوری",
        "همین لحظه",
    ]

    if any(word in normalized for word in immediate_words):

        return now

    # =========================================
    # 2. RELATIVE TIME
    # =========================================

    relative_result = parse_relative_time(
        normalized
    )

    if relative_result:

        return relative_result

    # =========================================
    # 3. EXPLICIT JALALI DATE
    # =========================================

    jalali_date = parse_jalali_date(
        normalized
    )

    if jalali_date:

        time_result = parse_time(normalized)

        if time_result:
            hour, minute = time_result

        else:
            hour = 0
            minute = 0

        result = datetime(
            year=jalali_date.year,
            month=jalali_date.month,
            day=jalali_date.day,
            hour=hour,
            minute=minute,
            tzinfo=TEHRAN_TZ
        )

        if result <= now:

            raise ValueError(
                "زمان مشخص‌شده در گذشته است."
            )

        return result

    # =========================================
    # 4. TODAY / TOMORROW / DAY AFTER TOMORROW
    # =========================================

    if "پس فردا" in normalized or "پس‌فردا" in normalized:

        target_date = (
            now + timedelta(days=2)
        ).date()

    elif "فردا" in normalized:

        target_date = (
            now + timedelta(days=1)
        ).date()

    elif "امروز" in normalized:

        target_date = now.date()

    else:

        raise ValueError(
            "نتوانستم زمان ارسال را تشخیص دهم. "
            "مثلاً بگویید «دو ساعت بعد»، "
            "«فردا ساعت ۱۰ شب» یا "
            "«۲۵ شهریور ۱۴۰۵ ساعت ۹ صبح»."
        )

    # =========================================
    # 5. TIME
    # =========================================

    time_result = parse_time(normalized)

    if not time_result:

        raise ValueError(
            "تاریخ مشخص شد اما ساعت مشخص نیست."
        )

    hour, minute = time_result

    result = datetime(
        year=target_date.year,
        month=target_date.month,
        day=target_date.day,
        hour=hour,
        minute=minute,
        tzinfo=TEHRAN_TZ
    )

    if result <= now:

        raise ValueError(
            "زمان مشخص‌شده در گذشته است."
        )

    return result