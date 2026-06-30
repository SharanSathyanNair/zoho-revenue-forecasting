import random
from datetime import timedelta

import numpy as np
import pandas as pd

from faker import Faker

fake = Faker()

# ==========================================================
# Configuration
# ==========================================================

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

START_DATE = pd.Timestamp("2022-01-03")
SIMULATION_WEEKS = 208

INITIAL_CUSTOMERS = 220
TARGET_ACTIVE_CUSTOMERS = 240

AVG_NEW_CUSTOMERS = 1.5

CUSTOMER_PROFILE_WEIGHTS = {"reliable": 0.55, "standard": 0.30, "risky": 0.15}

PROFILE_CONFIG = {
    "reliable": {
        "invoice_probability": 0.60,
        "avg_invoice": 45000,
        "invoice_sigma": 0.20,
        "payment_delay_mean": 4,
        "payment_delay_std": 2,
        "split_probability": 0.10,
        "partial_probability": 0.02,
        "default_probability": 0.003,
        "weekly_churn_probability": 0.0015,
    },
    "standard": {
        "invoice_probability": 0.46,
        "avg_invoice": 30000,
        "invoice_sigma": 0.28,
        "payment_delay_mean": 10,
        "payment_delay_std": 4,
        "split_probability": 0.18,
        "partial_probability": 0.05,
        "default_probability": 0.012,
        "weekly_churn_probability": 0.0040,
    },
    "risky": {
        "invoice_probability": 0.30,
        "avg_invoice": 18000,
        "invoice_sigma": 0.35,
        "payment_delay_mean": 22,
        "payment_delay_std": 7,
        "split_probability": 0.30,
        "partial_probability": 0.10,
        "default_probability": 0.05,
        "weekly_churn_probability": 0.0100,
    },
}


MONTHLY_SEASONALITY = {
    1: 0.90,
    2: 0.95,
    3: 1.00,
    4: 1.03,
    5: 1.05,
    6: 1.00,
    7: 0.98,
    8: 1.00,
    9: 1.05,
    10: 1.10,
    11: 1.15,
    12: 1.25,
}


# ==========================================================
# Global Storage
# ==========================================================

customers = []
invoices = []
payments = []

customer_counter = 1
invoice_counter = 1
payment_counter = 1


# ==========================================================
# ID Generators
# ==========================================================


def next_customer_id():
    global customer_counter
    customer_id = f"CUST{customer_counter:04d}"
    customer_counter += 1
    return customer_id


def next_invoice_id():
    global invoice_counter
    invoice_id = f"INV{invoice_counter:06d}"
    invoice_counter += 1
    return invoice_id


def next_payment_id():
    global payment_counter
    payment_id = f"PAY{payment_counter:06d}"
    payment_counter += 1
    return payment_id


# ==========================================================
# Helper Functions
# ==========================================================


def choose_profile():
    return random.choices(
        population=list(CUSTOMER_PROFILE_WEIGHTS.keys()),
        weights=list(CUSTOMER_PROFILE_WEIGHTS.values()),
        k=1,
    )[0]


def seasonal_multiplier(date):
    return MONTHLY_SEASONALITY[date.month]


def random_invoice_date(week_start):
    return week_start + timedelta(days=random.randint(0, 6))


def generate_base_invoice(profile):
    config = PROFILE_CONFIG[profile]
    amount = np.random.lognormal(
        mean=np.log(config["avg_invoice"]), sigma=config["invoice_sigma"]
    )
    return round(amount, 2)


def generate_invoice_amount(customer, invoice_date):
    weekly_variation = np.clip(np.random.normal(loc=1.0, scale=0.08), 0.75, 1.25)

    amount = (
        customer["base_invoice"]
        * customer["customer_value_factor"]
        * seasonal_multiplier(invoice_date)
        * weekly_variation
    )

    return round(max(amount, 500), 2)


def payment_delay(customer):
    delay = np.random.normal(
        customer["payment_delay_mean"], customer["payment_delay_std"]
    )
    return max(0, int(round(delay)))


def active_customer_count():
    return sum(customer["status"] == "active" for customer in customers)


# ==========================================================
# Customer Functions
# ==========================================================


def create_customer(join_date):
    profile = choose_profile()
    config = PROFILE_CONFIG[profile]

    customer = {
        # Zoho Books fields
        "customer_id": next_customer_id(),
        "customer_name": fake.company(),
        "created_time": join_date,
        "status": "active",
        "payment_terms": random.choice([15, 30, 45, 60]),

        # Internal simulation fields
        "profile": profile,
        "customer_value_factor": round(np.random.normal(1.0, 0.12), 2),
        "purchase_frequency_factor": round(np.random.normal(1.0, 0.10), 2),
        "base_invoice": generate_base_invoice(profile),
        "invoice_probability": config["invoice_probability"],
        "payment_delay_mean": config["payment_delay_mean"],
        "payment_delay_std": config["payment_delay_std"],
        "split_probability": config["split_probability"],
        "partial_probability": config["partial_probability"],
        "default_probability": config["default_probability"],
        "weekly_churn_probability": config["weekly_churn_probability"],
    }

    customers.append(customer)
    return customer


def initialize_business():
    for _ in range(INITIAL_CUSTOMERS):
        create_customer(START_DATE)


# ==========================================================
# Customer Churn
# ==========================================================


def churn_customers(current_date):
    for customer in customers:
        if customer["status"] != "active":
            continue

        if random.random() < customer["weekly_churn_probability"]:
            customer["status"] = "inactive"


# ==========================================================
# Weekly Customer Behaviour
# ==========================================================


def weekly_invoice_probability(customer):
    base_probability = (
        customer["invoice_probability"] * customer["purchase_frequency_factor"]
    )
    probability = np.random.normal(base_probability, 0.03)
    probability = np.clip(probability, 0.05, 0.95)
    return probability


# ==========================================================
# Invoice Functions
# ==========================================================


def create_invoice(customer, invoice_date):
    invoice = {
        "invoice_id": next_invoice_id(),
        "customer_id": customer["customer_id"],
        "invoice_date": invoice_date,
        "due_date": invoice_date + timedelta(days=30),
        "invoice_amount": generate_invoice_amount(customer, invoice_date),
        "amount_paid": 0.0,
        "status": "Open",
    }

    invoices.append(invoice)

    return invoice


def generate_weekly_invoices(week_start):
    weekly_invoices = []

    for customer in customers:
        if customer["status"] != "active":
            continue

        probability = weekly_invoice_probability(customer)

        if random.random() > probability:
            continue

        invoice_date = random_invoice_date(week_start)
        invoice = create_invoice(customer, invoice_date)
        weekly_invoices.append((customer, invoice))

    return weekly_invoices


# ==========================================================
# Customer Acquisition
# ==========================================================


def acquire_new_customers(current_date):
    active = active_customer_count()
    gap = TARGET_ACTIVE_CUSTOMERS - active
    expected = AVG_NEW_CUSTOMERS

    if gap >= 20:
        expected += 1.0
    elif gap >= 10:
        expected += 0.5
    elif gap <= -20:
        expected -= 1.0
    elif gap <= -10:
        expected -= 0.5

    expected = max(0.5, expected)

    new_customers = np.random.poisson(expected)

    for _ in range(new_customers):
        create_customer(current_date)


# ==========================================================
# Payment Functions
# ==========================================================


def create_payment(invoice, payment_date, amount):
    payment = {
        "payment_id": next_payment_id(),
        "invoice_id": invoice["invoice_id"],
        "customer_id": invoice["customer_id"],
        "payment_date": payment_date,
        "payment_amount": round(amount, 2),
    }

    payments.append(payment)


def process_invoice_payment(customer, invoice):
    invoice_amount = invoice["invoice_amount"]

    # Customer defaults
    if random.random() < customer["default_probability"]:
        invoice["status"] = "Overdue"
        return

    payment_date = invoice["invoice_date"] + timedelta(days=payment_delay(customer))

    # Partial payment
    if random.random() < customer["partial_probability"]:
        paid_amount = round(invoice_amount * random.uniform(0.40, 0.90), 2)
        create_payment(invoice, payment_date, paid_amount)
        invoice["amount_paid"] = paid_amount
        invoice["status"] = "Partially Paid"
        return

    # Split payment
    if random.random() < customer["split_probability"]:
        first_payment = round(invoice_amount * random.uniform(0.30, 0.70), 2)
        second_payment = round(invoice_amount - first_payment, 2)
        create_payment(invoice, payment_date, first_payment)
        create_payment(invoice, payment_date + timedelta(days=7), second_payment)
        invoice["amount_paid"] = invoice_amount
        invoice["status"] = "Paid"
        return

    # Full payment
    create_payment(invoice, payment_date, invoice_amount)
    invoice["amount_paid"] = invoice_amount
    invoice["status"] = "Paid"


# ==========================================================
# Weekly Payment Simulation
# ==========================================================


def process_weekly_payments(weekly_invoices):
    for customer, invoice in weekly_invoices:
        process_invoice_payment(customer, invoice)


# ==========================================================
# Weekly Business Simulation
# ==========================================================


def simulate_week(week_start):
    # Existing customers may churn
    churn_customers(week_start)

    # Replace lost customers
    acquire_new_customers(week_start)

    # Generate invoices
    weekly_invoices = generate_weekly_invoices(week_start)

    # Process payments
    process_weekly_payments(weekly_invoices)


# ==========================================================
# Full Business Simulation
# ==========================================================


def simulate_business():
    current_week = START_DATE

    for _ in range(SIMULATION_WEEKS):
        simulate_week(current_week)
        current_week += timedelta(days=7)





# ==========================================================
# Export Functions
# ==========================================================

def export_customers():
    customers_df = pd.DataFrame(customers)

    export_columns = [
        "customer_id",
        "customer_name",
        "created_time",
        "status",
        "payment_terms",
    ]

    customers_df = customers_df[export_columns]
    customers_df.sort_values("customer_id", inplace=True)
    customers_df.to_csv("data/customers.csv", index=False)

    return customers_df


def export_invoices():
    invoices_df = pd.DataFrame(invoices)

    invoices_df = invoices_df.rename(
        columns={
            "invoice_date": "date",
            "invoice_amount": "total",
        }
    )

    invoices_df.sort_values("date", inplace=True)
    invoices_df.to_csv("data/invoices.csv", index=False)

    return invoices_df


def export_payments():
    payments_df = pd.DataFrame(payments)

    payments_df = payments_df.rename(
        columns={
            "payment_date": "date",
            "payment_amount": "amount",
        }
    )

    payments_df.sort_values("date", inplace=True)
    payments_df.to_csv("data/customerpayments.csv", index=False)

    return payments_df


def export_payments():
    payments_df = pd.DataFrame(payments)

    payments_df = payments_df.rename(
        columns={
            "payment_date": "date",
            "payment_amount": "amount",
        }
    )

    payments_df.sort_values("date", inplace=True)

    payments_df.to_csv("data/customerpayments.csv", index=False)

    return payments_df

# ==========================================================
# Validation
# ==========================================================


def validate_data(customers_df, invoices_df, payments_df):
    print("\nRunning validation...\n")

    assert customers_df["customer_id"].is_unique
    assert invoices_df["invoice_id"].is_unique
    assert payments_df["payment_id"].is_unique

    assert (invoices_df["total"] > 0).all()
    assert (invoices_df["amount_paid"] <= invoices_df["total"]).all()

    print("Validation passed.")


# ==========================================================
# Business Summary
# ==========================================================


def business_summary(customers_df, invoices_df, payments_df):
    active = (customers_df["status"] == "active").sum()
    inactive = (customers_df["status"] == "inactive").sum()

    print("\n==============================")
    print("Business Summary")
    print("==============================")
    print(f"Simulation Weeks : {SIMULATION_WEEKS}")
    print(f"Customers        : {len(customers_df)}")
    print(f"Active Customers : {active}")
    print(f"Inactive         : {inactive}")
    print(f"Invoices         : {len(invoices_df)}")
    print(f"Payments         : {len(payments_df)}")
    print(f"Average Invoice  : {invoices_df['total'].mean():,.2f}")
    print(f"Average Payment  : {payments_df['amount'].mean():,.2f}")
    print("==============================")

# ==========================================================
# Main
# ==========================================================


def main():
    print("\nStarting business simulation...\n")

    initialize_business()
    simulate_business()

    customers_df = export_customers()
    invoices_df = export_invoices()
    payments_df = export_payments()

    validate_data(customers_df, invoices_df, payments_df)
    business_summary(customers_df, invoices_df, payments_df)

    print("\nCSV files generated successfully.")


if __name__ == "__main__":
    main()