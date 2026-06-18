#______importing libraries__________
import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import timedelta

#______initialising data__________
fake = Faker()
random.seed(42)
np.random.seed(42)

START_DATE = pd.Timestamp("2022-01-01")
END_DATE = pd.Timestamp("2025-12-31")
NUM_CUSTOMERS = 300

#______customer profiles, purely a synthetic-data generation device__________
# real Zoho Books data has no equivalent field, this exists only to inject
# realistic variation into the fake invoices and payments we generate below
CUSTOMER_PROFILES = ["reliable", "slow", "risky"]
PROFILE_WEIGHTS = [0.6, 0.3, 0.1]

#______creating customers__________
customers = []
for i in range(NUM_CUSTOMERS):
    profile = random.choices(CUSTOMER_PROFILES, weights=PROFILE_WEIGHTS, k=1)[0]
    customers.append({
        "customer_id": f"CUST{i+1:04d}",
        "customer_name": fake.company(),
        "profile": profile
    })

customers_df = pd.DataFrame(customers)

#______creating invoices, mirrors Zoho Books /invoices fields__________
# no payment_date or amount_paid here anymore, those live in customerpayments
invoices = []
invoice_id_counter = 1

for _, customer in customers_df.iterrows():
    num_invoices = np.random.randint(50, 150)

    for _ in range(num_invoices):
        invoice_date = START_DATE + timedelta(days=int(np.random.randint(0, 1460)))
        payment_terms = int(np.random.choice([15, 30, 45, 60]))
        due_date = invoice_date + timedelta(days=payment_terms)
        total = round(np.random.lognormal(mean=10, sigma=1), 2)

        invoices.append({
            "invoice_id": f"INV{invoice_id_counter:06d}",
            "customer_id": customer["customer_id"],
            "date": invoice_date,
            "due_date": due_date,
            "payment_terms": payment_terms,
            "total": total,
            "balance": total,   # starts equal to total, gets reduced as payments come in below
            "status": "sent"    # will be overwritten to paid/partially_paid/overdue after payments are generated
        })
        invoice_id_counter += 1

invoices_df = pd.DataFrame(invoices)

#______generating payment plans per invoice, based on customer profile__________
# this is the new logic replacing the old single payment_date/amount_paid approach
# each invoice gets a plan: full single payment, two-part split, three-part split, or default (never paid)

def get_payment_plan(profile):
    if profile == "reliable":
        # mostly single payment, small chance of a two-part split, no defaults
        plan = random.choices(["single", "two_part"], weights=[0.85, 0.15], k=1)[0]
        defaulted = False
    elif profile == "slow":
        # mostly single payment but late, modest chance of two-part, no defaults
        plan = random.choices(["single", "two_part"], weights=[0.75, 0.25], k=1)[0]
        defaulted = False
    else:  # risky
        # highest chance of splitting, plus the original 20% default chance
        defaulted = np.random.random() < 0.2
        if defaulted:
            plan = "none"
        else:
            plan = random.choices(["single", "two_part", "three_part"], weights=[0.45, 0.35, 0.20], k=1)[0]
    return plan, defaulted

#______delay distributions per profile, in days relative to due_date__________
def get_delay(profile):
    if profile == "reliable":
        return np.random.randint(-5, 10)
    elif profile == "slow":
        return np.random.randint(10, 45)
    else:
        return np.random.randint(20, 90)

#______generating customerpayments, mirrors Zoho Books /customerpayments fields__________
payments = []
payment_id_counter = 1

# track running balance per invoice as payments are applied
invoice_balances = dict(zip(invoices_df["invoice_id"], invoices_df["total"]))
invoice_status = dict(zip(invoices_df["invoice_id"], ["sent"] * len(invoices_df)))

for _, invoice in invoices_df.iterrows():
    customer_id = invoice["customer_id"]
    profile = customers_df.loc[customers_df["customer_id"] == customer_id, "profile"].values[0]
    plan, defaulted = get_payment_plan(profile)

    if defaulted:
        invoice_status[invoice["invoice_id"]] = "unpaid"
        continue  # no payment rows generated, balance stays at full total

    total = invoice["total"]
    due_date = invoice["due_date"]
    invoice_date = invoice["date"]

    if plan == "single":
        delay = get_delay(profile)
        payment_date = due_date + timedelta(days=int(delay))
        amount = total

        payments.append({
            "payment_id": f"PAY{payment_id_counter:06d}",
            "customer_id": customer_id,
            "invoice_id": invoice["invoice_id"],
            "amount_applied": amount,
            "date": payment_date,
            "payment_mode": random.choice(["banktransfer", "check", "creditcard", "cash"]),
            "status": "success"
        })
        payment_id_counter += 1
        invoice_balances[invoice["invoice_id"]] -= amount

    elif plan == "two_part":
        # first installment: 40-70% of total, lands between invoice date and due date
        first_pct = np.random.uniform(0.4, 0.7)
        first_amount = round(total * first_pct, 2)
        days_into_term = int((due_date - invoice_date).days * np.random.uniform(0.3, 0.9))
        first_date = invoice_date + timedelta(days=days_into_term)

        payments.append({
            "payment_id": f"PAY{payment_id_counter:06d}",
            "customer_id": customer_id,
            "invoice_id": invoice["invoice_id"],
            "amount_applied": first_amount,
            "date": first_date,
            "payment_mode": random.choice(["banktransfer", "check", "creditcard", "cash"]),
            "status": "success"
        })
        payment_id_counter += 1
        invoice_balances[invoice["invoice_id"]] -= first_amount

        # second installment: remaining balance, lands after due date with profile delay
        second_amount = round(total - first_amount, 2)
        delay = get_delay(profile)
        second_date = due_date + timedelta(days=int(delay))

        payments.append({
            "payment_id": f"PAY{payment_id_counter:06d}",
            "customer_id": customer_id,
            "invoice_id": invoice["invoice_id"],
            "amount_applied": second_amount,
            "date": second_date,
            "payment_mode": random.choice(["banktransfer", "check", "creditcard", "cash"]),
            "status": "success"
        })
        payment_id_counter += 1
        invoice_balances[invoice["invoice_id"]] -= second_amount

    elif plan == "three_part":
        # three smaller portions with increasing delay between each
        pct_1 = np.random.uniform(0.25, 0.4)
        pct_2 = np.random.uniform(0.25, 0.4)
        pct_3 = 1 - pct_1 - pct_2

        amount_1 = round(total * pct_1, 2)
        amount_2 = round(total * pct_2, 2)
        amount_3 = round(total - amount_1 - amount_2, 2)

        days_into_term_1 = int((due_date - invoice_date).days * np.random.uniform(0.2, 0.5))
        date_1 = invoice_date + timedelta(days=days_into_term_1)

        days_into_term_2 = int((due_date - invoice_date).days * np.random.uniform(0.6, 0.95))
        date_2 = invoice_date + timedelta(days=days_into_term_2)

        delay_3 = get_delay(profile) + np.random.randint(10, 30)  # extra lag for the final installment
        date_3 = due_date + timedelta(days=int(delay_3))

        for amount, date in [(amount_1, date_1), (amount_2, date_2), (amount_3, date_3)]:
            payments.append({
                "payment_id": f"PAY{payment_id_counter:06d}",
                "customer_id": customer_id,
                "invoice_id": invoice["invoice_id"],
                "amount_applied": amount,
                "date": date,
                "payment_mode": random.choice(["banktransfer", "check", "creditcard", "cash"]),
                "status": "success"
            })
            payment_id_counter += 1
            invoice_balances[invoice["invoice_id"]] -= amount

    # determine final status based on remaining balance
    remaining = round(invoice_balances[invoice["invoice_id"]], 2)
    if remaining <= 0:
        invoice_status[invoice["invoice_id"]] = "paid"
    else:
        invoice_status[invoice["invoice_id"]] = "partially_paid"

payments_df = pd.DataFrame(payments)

#______updating invoices_df with final balance and status__________
invoices_df["balance"] = invoices_df["invoice_id"].map(invoice_balances).round(2)
invoices_df["status"] = invoices_df["invoice_id"].map(invoice_status)

#______saving both tables to data folder__________
invoices_df.to_csv("data/invoices.csv", index=False)
payments_df.to_csv("data/customerpayments.csv", index=False)
customers_df.to_csv("data/customers.csv", index=False)

print(f"Customers generated: {len(customers_df)}")
print(f"Invoices generated: {len(invoices_df)}")
print(f"Payments generated: {len(payments_df)}")
print(invoices_df.head())
print(payments_df.head())