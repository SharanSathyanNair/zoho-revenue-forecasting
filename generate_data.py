#______importing from libraries________
import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import timedelta
#______initialising data ________________
fake=Faker()
random.seed(42)
np.random.seed(42)

Start_Date=pd.Timestamp('2022-01-01')
End_Date=pd.Timestamp('2025-12-31')
Num_Customers=300

Customer_profiles=['reliable','slow','risky']
Profile_weights=[0.6,0.3,0.1]

#_______creating customer profiles___________________________________
customers=[]
for i in range(Num_Customers):
  profile=random.choices(Customer_profiles,weights=Profile_weights,k=1)[0]
  customers.append({
    "customer_ID":f"CUST{i+1:04d}",
    "customer_name": fake.company(),
    "profile":profile
  })
customers_df=pd.DataFrame(customers)

#_______creating invoices_____________________________
invoices=[]
invoice_ID_counter=1
for _,customer in customers_df.iterrows():
  num_invoices=np.random.randint(50,150)

  for _ in range(num_invoices):
    invoice_date=Start_Date+timedelta(days=np.random.randint(0,1460))
    due_date=invoice_date+timedelta(days=int(np.random.choice([15,30,45,60])))
    amount=round(np.random.lognormal(mean=10,sigma=1),2)
#_______matching customer profiles and their relation with respective payment delay ________________________________
    if customer["profile"]=="reliable":
      payment_delay=np.random.randint(-5,10)
      default=False
    elif customer["profile"]=="slow":
      default=False
      payment_delay=np.random.randint(10,24)
    else:
      payment_delay=np.random.randint(20,90)
      default=np.random.random()<0.2
#______setting up the payment date for customers______________________________
    if default:
      status="unpaid"
      payment_date=None
      amount_paid=0
    else:
      payment_date=due_date+timedelta(days=int(payment_delay))
      status="paid"
      amount_paid=amount
#______creating Invoice profiles________________________
    invoices.append({
      "invoice_ID":f"INV{invoice_ID_counter:06d}",
      "customer_ID":customer["customer_ID"],
      "invoice_date":invoice_date,
      "due_date":due_date,
      "amount":amount,
      "remaining_balance":amount,
      "status":status,
      "payment_date":payment_date,
      "amount_paid":amount_paid,
    })
    invoice_ID_counter+=1
invoices_df=pd.DataFrame(invoices)
#______saving invoices and customer profiles to data folder and checking whether all profiles have been made____________
invoices_df.to_csv("data/invoices.csv", index=False)
customers_df.to_csv("data/customers.csv", index=False)

print(f"Customers generated: {len(customers_df)}")
print(f"Invoices generated: {len(invoices_df)}")
print(invoices_df.head())      