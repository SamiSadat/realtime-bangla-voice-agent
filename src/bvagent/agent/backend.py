"""A stand-in for the systems a real deployment would call.

Deliberately in-memory and deterministic: the point of this project is to
measure the pipeline, and a tool that sometimes takes 300ms and sometimes 30ms
would show up as pipeline variance. ``latency_secs`` simulates a backend call
explicitly, so the cost of tool use is visible in the report rather than
accidental.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

# Bengali digits, as a caller would say and hear them.
_BN_DIGITS = "০১২৩৪৫৬৭৮৯"


def to_bengali_digits(value: Any) -> str:
    """Render a number in Bengali numerals.

    Tool results are read aloud, so they are returned in the script the caller
    is listening in. Leaving them as ASCII digits makes the TTS model switch
    into a different phoneme set mid-sentence, which is audible.
    """
    return "".join(_BN_DIGITS[int(c)] if c.isdigit() else c for c in str(value))


def from_bengali_digits(text: str) -> str:
    """Normalize Bengali numerals to ASCII, for lookups.

    Callers and the LLM both produce either form, so every identifier is
    normalized on the way in.
    """
    table = {ord(d): str(i) for i, d in enumerate(_BN_DIGITS)}
    return str(text).translate(table)


@dataclass
class LoanAccount:
    account_number: str
    customer_name: str
    outstanding_bdt: int
    installment_bdt: int
    due_date: date
    overdue_days: int = 0


@dataclass
class Appointment:
    reference: str
    customer_name: str
    branch: str
    slot: datetime


@dataclass
class Backend:
    """Fake CRM/loan book with a configurable, deterministic call latency."""

    latency_secs: float = 0.05
    loans: dict[str, LoanAccount] = field(default_factory=dict)
    appointments: dict[str, Appointment] = field(default_factory=dict)
    branches: tuple[str, ...] = ("গুলশান", "ধানমন্ডি", "উত্তরা", "চট্টগ্রাম")

    def __post_init__(self) -> None:
        if not self.loans:
            today = date.today()
            self.loans = {
                "4421": LoanAccount("4421", "রহিম উদ্দিন", 45000, 5000, today + timedelta(days=4)),
                "7813": LoanAccount("7813", "ফাতেমা বেগম", 128000, 9500, today - timedelta(days=6), 6),
                "9004": LoanAccount("9004", "করিম শেখ", 15000, 2500, today + timedelta(days=12)),
            }

    async def _call(self) -> None:
        if self.latency_secs > 0:
            await asyncio.sleep(self.latency_secs)

    # -- loan collection -------------------------------------------------

    async def get_loan_status(self, account_number: str) -> dict[str, Any]:
        await self._call()
        key = from_bengali_digits(account_number).strip()
        loan = self.loans.get(key)
        if loan is None:
            return {
                "found": False,
                "message": f"অ্যাকাউন্ট {to_bengali_digits(key)} পাওয়া যায়নি।",
            }
        return {
            "found": True,
            "customer_name": loan.customer_name,
            "account_number": to_bengali_digits(loan.account_number),
            "installment_bdt": to_bengali_digits(loan.installment_bdt),
            "outstanding_bdt": to_bengali_digits(loan.outstanding_bdt),
            "due_date": _bangla_date(loan.due_date),
            "is_overdue": loan.overdue_days > 0,
            "overdue_days": to_bengali_digits(loan.overdue_days),
        }

    async def record_payment_promise(
        self, account_number: str, promised_date: str, amount_bdt: int | None = None
    ) -> dict[str, Any]:
        await self._call()
        key = from_bengali_digits(account_number).strip()
        loan = self.loans.get(key)
        if loan is None:
            return {"recorded": False, "message": "অ্যাকাউন্ট পাওয়া যায়নি।"}
        amount = amount_bdt or loan.installment_bdt
        return {
            "recorded": True,
            "account_number": to_bengali_digits(loan.account_number),
            "promised_date": promised_date,
            "amount_bdt": to_bengali_digits(amount),
            "message": "প্রতিশ্রুতি রেকর্ড করা হয়েছে।",
        }

    # -- appointment booking ---------------------------------------------

    async def find_available_slots(self, branch: str, preferred_day: str | None = None) -> dict[str, Any]:
        await self._call()
        # Deterministic per branch, so a replayed benchmark gets the same
        # answer and therefore the same reply length and TTS cost.
        seed = random.Random(f"{branch}:{preferred_day}")
        base = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(days=1)
        slots = [base + timedelta(days=seed.randint(0, 3), hours=seed.randint(1, 7)) for _ in range(3)]
        slots.sort()
        return {
            "branch": branch,
            "slots": [_bangla_datetime(s) for s in slots],
            "slot_ids": [s.isoformat(timespec="minutes") for s in slots],
        }

    async def book_appointment(
        self, customer_name: str, branch: str, slot_id: str
    ) -> dict[str, Any]:
        await self._call()
        try:
            slot = datetime.fromisoformat(slot_id)
        except ValueError:
            return {"booked": False, "message": "সময়টি বোঝা যায়নি, আবার বলুন।"}
        reference = f"BR{abs(hash((customer_name, slot_id))) % 10000:04d}"
        self.appointments[reference] = Appointment(reference, customer_name, branch, slot)
        return {
            "booked": True,
            "reference": to_bengali_digits(reference[2:]),
            "branch": branch,
            "slot": _bangla_datetime(slot),
            "message": "অ্যাপয়েন্টমেন্ট নিশ্চিত করা হয়েছে।",
        }


_BN_MONTHS = [
    "জানুয়ারি", "ফেব্রুয়ারি", "মার্চ", "এপ্রিল", "মে", "জুন",
    "জুলাই", "আগস্ট", "সেপ্টেম্বর", "অক্টোবর", "নভেম্বর", "ডিসেম্বর",
]


def _bangla_date(d: date) -> str:
    return f"{to_bengali_digits(d.day)} {_BN_MONTHS[d.month - 1]}"


def _bangla_datetime(dt: datetime) -> str:
    period = "সকাল" if dt.hour < 12 else ("দুপুর" if dt.hour < 16 else "বিকেল")
    hour = dt.hour if dt.hour <= 12 else dt.hour - 12
    return f"{_bangla_date(dt.date())}, {period} {to_bengali_digits(hour)}টা"
