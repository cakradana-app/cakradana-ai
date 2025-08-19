import random
from faker import Faker
import pandas as pd
from datetime import date

# Setup Faker with Indonesian locale for realistic names
fake = Faker('id_ID')
random.seed(42)
Faker.seed(42)

# Legal thresholds (in rupiah)
MAX_INDIVIDUAL_PARTY = 200_000_000       # UU Parpol: individual → party
MAX_COMPANY_PARTY    = 800_000_000       # UU Parpol: company  → party
MAX_INDIVIDUAL_CAMPAIGN = 2_500_000_000  # UU Pemilu: individual → candidate
MAX_COMPANY_CAMPAIGN    = 25_000_000_000 # UU Pemilu: company  → candidate

# Number of rows
n_total = 10000
n_risk  = n_total // 2   # 5000 risky
n_not_risk = n_total - n_risk  # 5000 non-risk

# Risk category counts (of 5000 risky)
over_limit_count     = int(n_risk * 0.05)  # 250
illegal_source_count = int(n_risk * 0.05)  # 250
smurfing_count       = int(n_risk * 0.30)  # 1500
proxy_count          = int(n_risk * 0.30)  # 1500
self_funded_count    = n_risk - (over_limit_count + illegal_source_count + smurfing_count + proxy_count)
# = 1500 (leftover)

# Example illegal sources (government/BUMN/BUMD/village)
illegal_sources = [
    "Pemerintah Kabupaten Banyumas", "Pemerintah Kota Surabaya", "Pemerintah Provinsi Bali",
    "Kementerian Pendidikan Nasional", "Kementerian Keuangan",
    "PT PLN (Persero)", "PT Pertamina (Persero)", "PT Kereta Api Indonesia (Persero)",
    "PT Telekomunikasi Indonesia (Persero)", "BUMN Mina Permai", "BUMD Energi Nusantara",
    "Pemerintah Desa Sukamulya", "Pemerintah Desa Cibeber", "Pemkab Toba", "Pemkot Bogor"
]

# Example political party names
party_names = [
    "Partai Harapan Bangsa", "Partai Indonesia Maju", "Partai Rakyat Sejahtera", 
    "Partai Nasional Demokrat", "Partai Persatuan", "Partai Kebangkitan Rakyat", 
    "Partai Amanat Rakyat", "Partai Solidaritas Indonesia", "Partai Keadilan Indonesia", 
    "Partai Demokrat"
]

# Helper functions to generate names and dates
def gen_sender_name(sender_type):
    if sender_type == 'individual':
        return fake.name()
    elif sender_type == 'corporation':
        return fake.company()
    elif sender_type == 'organization':
        return fake.company()
    return "Unknown"

def gen_receiver_name(receiver_type):
    if receiver_type == 'political-party':
        return random.choice(party_names)
    elif receiver_type == 'individual':
        return fake.name()
    return "Unknown"

def gen_date():
    # Random date between Jan 1, 2024 and Dec 31, 2025
    return fake.date_between(start_date=date(2024,1,1), end_date=date(2025,12,31))

# Container for all records
data = []

# 1. Generate Over-Limit donations (risk type "over_limit")
for _ in range(over_limit_count):
    sender_type = random.choice(['individual', 'corporation', 'organization'])
    receiver_type = random.choice(['political-party', 'individual'])
    sender_name = gen_sender_name(sender_type)
    receiver_name = gen_receiver_name(receiver_type)
    # Determine the legal threshold based on type
    if receiver_type == 'political-party':
        threshold = MAX_INDIVIDUAL_PARTY if sender_type == 'individual' else MAX_COMPANY_PARTY
    else:
        threshold = MAX_INDIVIDUAL_CAMPAIGN if sender_type == 'individual' else MAX_COMPANY_CAMPAIGN
    # Set amount *above* the legal limit
    amount = int(threshold * random.uniform(1.1, 2.0))
    data.append({
        "sender": sender_name,
        "sender_type": sender_type,
        "receiver": receiver_name,
        "receiver_type": receiver_type,
        "date": gen_date(),
        "amount": amount,
        "risk": True,
        "risk_type": "over_limit"
    })

# 2. Generate Illegal-Source donations (risk type "illegal_source")
for _ in range(illegal_source_count):
    sender_name = random.choice(illegal_sources)
    # Determine sender_type from name heuristic
    if 'PT ' in sender_name or 'BUMN' in sender_name:
        sender_type = 'corporation'
    else:
        sender_type = 'organization'
    receiver_type = random.choice(['political-party', 'individual'])
    receiver_name = gen_receiver_name(receiver_type)
    # Amount can be random (possibly large or small)
    if receiver_type == 'political-party':
        limit = MAX_INDIVIDUAL_PARTY if sender_type == 'individual' else MAX_COMPANY_PARTY
    else:
        limit = MAX_INDIVIDUAL_CAMPAIGN if sender_type == 'individual' else MAX_COMPANY_CAMPAIGN
    amount = int(random.uniform(1_000_000, limit * random.uniform(0.5, 1.5)))
    data.append({
        "sender": sender_name,
        "sender_type": sender_type,
        "receiver": receiver_name,
        "receiver_type": receiver_type,
        "date": gen_date(),
        "amount": amount,
        "risk": True,
        "risk_type": "illegal_source"
    })

# 3. Generate Smurfing donations (risk type "smurfing")
for _ in range(smurfing_count):
    sender_type = random.choice(['individual', 'corporation', 'organization'])
    sender_name = gen_sender_name(sender_type)
    receiver_type = random.choice(['political-party', 'individual'])
    receiver_name = gen_receiver_name(receiver_type)
    # Small amounts (e.g. under ~100M)
    amount = int(random.uniform(10_000, 100_000_000))
    data.append({
        "sender": sender_name,
        "sender_type": sender_type,
        "receiver": receiver_name,
        "receiver_type": receiver_type,
        "date": gen_date(),
        "amount": amount,
        "risk": True,
        "risk_type": "smurfing"
    })

# 4. Generate Proxy Account donations (risk type "proxy")
for _ in range(proxy_count):
    sender_type = random.choice(['corporation', 'organization'])
    sender_name = gen_sender_name(sender_type)
    receiver_type = random.choice(['political-party', 'individual'])
    receiver_name = gen_receiver_name(receiver_type)
    # Moderate-to-large amounts (e.g. 10M to 2B)
    amount = int(random.uniform(10_000_000, 2_000_000_000))
    data.append({
        "sender": sender_name,
        "sender_type": sender_type,
        "receiver": receiver_name,
        "receiver_type": receiver_type,
        "date": gen_date(),
        "amount": amount,
        "risk": True,
        "risk_type": "proxy"
    })

# 5. Generate False Self-Funded donations (risk type "self_funded")
for _ in range(self_funded_count):
    sender_type = random.choice(['individual', 'corporation'])
    sender_name = gen_sender_name(sender_type)
    receiver_type = 'individual'  # typically a candidate claiming personal funds
    receiver_name = gen_receiver_name(receiver_type)
    # Could be within legal range or moderate
    if sender_type == 'individual':
        amount = int(random.uniform(1_000_000, MAX_INDIVIDUAL_CAMPAIGN))
    else:
        amount = int(random.uniform(1_000_000, MAX_COMPANY_CAMPAIGN))
    data.append({
        "sender": sender_name,
        "sender_type": sender_type,
        "receiver": receiver_name,
        "receiver_type": receiver_type,
        "date": gen_date(),
        "amount": amount,
        "risk": True,
        "risk_type": "self_funded"
    })

# 6. Generate Non-risk (valid) donations (risk=False)
for _ in range(n_not_risk):
    sender_type = random.choices(['individual', 'corporation', 'organization'], weights=[0.6, 0.3, 0.1])[0]
    sender_name = gen_sender_name(sender_type)
    receiver_type = random.choice(['political-party', 'individual'])
    receiver_name = gen_receiver_name(receiver_type)
    # Amount strictly within legal limits
    if receiver_type == 'political-party':
        if sender_type == 'individual':
            amount = int(random.uniform(1_000_000, MAX_INDIVIDUAL_PARTY * 0.9))
        else:
            amount = int(random.uniform(1_000_000, MAX_COMPANY_PARTY * 0.9))
    else:
        if sender_type == 'individual':
            amount = int(random.uniform(1_000_000, MAX_INDIVIDUAL_CAMPAIGN * 0.9))
        else:
            amount = int(random.uniform(1_000_000, MAX_COMPANY_CAMPAIGN * 0.9))
    data.append({
        "sender": sender_name,
        "sender_type": sender_type,
        "receiver": receiver_name,
        "receiver_type": receiver_type,
        "date": gen_date(),
        "amount": amount,
        "risk": False,
        "risk_type": None
    })

# Create DataFrame, shuffle rows, and save to CSV
df = pd.DataFrame(data)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv('synthetic_donations.csv', index=False)

print("Generated synthetic data with", len(df), "rows.")