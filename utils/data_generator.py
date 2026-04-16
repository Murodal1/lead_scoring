"""
Realistic CRM sample data generator
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


INDUSTRIES = [
    'Technology', 'Healthcare', 'Finance', 'Manufacturing',
    'Retail', 'Education', 'Real Estate', 'Consulting',
    'E-commerce', 'Logistics'
]

DEAL_STAGES = ['New', 'Qualified', 'Demo', 'Proposal', 'Negotiation']

LEAD_SOURCES = [
    'Website', 'LinkedIn', 'Cold Email', 'Referral',
    'Conference', 'Inbound Call', 'Google Ads', 'Partner'
]

FIRST_NAMES = [
    'Alisher', 'Bobur', 'Dildora', 'Farrukh', 'Gulnora',
    'Hamid', 'Iroda', 'Jasur', 'Kamola', 'Lochinbek',
    'Sarah', 'Michael', 'Emma', 'James', 'Olivia',
    'David', 'Sophia', 'Daniel', 'Isabella', 'Matthew'
]

LAST_NAMES = [
    'Karimov', 'Rahimov', 'Yusupov', 'Toshmatov', 'Mirzaev',
    'Abdullayev', 'Hasanov', 'Ergashev', 'Normatov', 'Xolmatov',
    'Smith', 'Johnson', 'Williams', 'Brown', 'Jones',
    'Davis', 'Miller', 'Wilson', 'Moore', 'Taylor'
]

COMPANY_PREFIXES = [
    'Tech', 'Global', 'Smart', 'Digital', 'Prime',
    'Alpha', 'Next', 'Pro', 'Elite', 'Vision'
]

COMPANY_SUFFIXES = [
    'Solutions', 'Systems', 'Group', 'Corp', 'Ltd',
    'Ventures', 'Technologies', 'Partners', 'Associates', 'Hub'
]


def generate_sample_data(n: int = 150) -> pd.DataFrame:
    random.seed(42)
    np.random.seed(42)

    records = []
    for i in range(n):
        industry     = random.choice(INDUSTRIES)
        deal_stage   = random.choice(DEAL_STAGES)
        lead_source  = random.choice(LEAD_SOURCES)

        # Budget realistic range by industry
        budget_ranges = {
            'Technology':    (20000, 500000),
            'Healthcare':    (50000, 800000),
            'Finance':       (30000, 600000),
            'Manufacturing': (40000, 400000),
            'Retail':        (10000, 200000),
            'Education':     (5000,  150000),
            'Real Estate':   (50000, 1000000),
            'Consulting':    (15000, 300000),
            'E-commerce':    (10000, 250000),
            'Logistics':     (25000, 350000),
        }
        b_min, b_max = budget_ranges.get(industry, (10000, 300000))
        budget = round(np.random.lognormal(
            mean=np.log((b_min + b_max) / 2), sigma=0.5
        ))
        budget = int(np.clip(budget, b_min, b_max))

        employees = int(np.clip(np.random.lognormal(4, 1), 5, 5000))
        website_visits = int(np.random.poisson(20) + (5 if deal_stage in ['Proposal','Negotiation'] else 0))
        emails_opened = int(np.random.poisson(8) + (3 if deal_stage in ['Demo','Proposal','Negotiation'] else 0))
        meetings_held = int(np.random.choice([0, 1, 2, 3, 4, 5],
                                               p=[0.3, 0.3, 0.2, 0.1, 0.07, 0.03]))
        days_in_pipeline = int(np.random.uniform(1, 180))

        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        company = f"{random.choice(COMPANY_PREFIXES)} {random.choice(COMPANY_SUFFIXES)}"

        created_at = datetime.now() - timedelta(days=random.randint(0, 180))

        records.append({
            'company_name':     company,
            'contact_name':     f"{first} {last}",
            'email':            f"{first.lower()}.{last.lower()}@{company.replace(' ','').lower()}.com",
            'phone':            f"+998 {random.randint(90,99)} {random.randint(100,999)}-{random.randint(10,99)}-{random.randint(10,99)}",
            'industry':         industry,
            'budget':           budget,
            'deal_stage':       deal_stage,
            'lead_source':      lead_source,
            'employees':        employees,
            'website_visits':   website_visits,
            'emails_opened':    emails_opened,
            'meetings_held':    meetings_held,
            'days_in_pipeline': days_in_pipeline,
            'created_at':       created_at.strftime("%Y-%m-%d"),
        })

    return pd.DataFrame(records)