# 🎯 Lead Scoring & Pipeline Manager

ML-based CRM lead tahlili va conversion prediction tizimi.

## 🏗️ Loyha Tuzilishi

```
lead_scoring/
├── app.py                    # Asosiy Streamlit app
├── requirements.txt          # Dependencies
├── .streamlit/
│   ├── config.toml          # Streamlit sozlamalari
│   └── secrets.toml         # DB credentials (local only)
└── utils/
    ├── __init__.py
    ├── database.py          # PostgreSQL / SQLite manager
    ├── ml_model.py          # GradientBoosting + RF model
    └── data_generator.py    # Sample data yaratuvchi
```

## ⚙️ Local PyCharm Setup

### 1. Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 2. Dependencies o'rnatish

```bash
pip install -r requirements.txt
```

### 3. PostgreSQL (ixtiyoriy — SQLite ham ishlaydi)

Agar PostgreSQL ishlatmoqchi bo'lsangiz:

```bash
# .env fayl yarating
DATABASE_URL=postgresql://username:password@localhost:5432/leads_db
```

Yoki `.streamlit/secrets.toml` ga yozing:
```toml
DATABASE_URL = "postgresql://username:password@localhost:5432/leads_db"
```

### 4. Ishga tushirish

```bash
cd lead_scoring
streamlit run app.py
```

Browser'da: `http://localhost:8501`

---

## 🚀 Streamlit Cloud Deploy

### 1. GitHub'ga yuklash

```bash
git init
git add .
git commit -m "Lead Scoring App"
git remote add origin https://github.com/USERNAME/lead-scoring.git
git push -u origin main
```

### 2. share.streamlit.io

1. https://share.streamlit.io ga kiring
2. "New app" tugmasini bosing
3. GitHub repo tanlang
4. Main file path: `app.py`
5. "Deploy!" bosing

### 3. PostgreSQL (Streamlit Cloud uchun)

Streamlit Cloud → App settings → **Secrets** ga kiriting:

```toml
DATABASE_URL = "postgresql://user:pass@host:5432/dbname"
```

**Bepul PostgreSQL providers:**
- [Supabase](https://supabase.com) — 500MB bepul
- [Neon](https://neon.tech) — serverless, bepul tier
- [Railway](https://railway.app) — $5/month

---

## 📊 CSV Format

Custom CSV yuklash uchun quyidagi ustunlar bo'lishi kerak:

| Ustun | Tip | Misol |
|-------|-----|-------|
| company_name | text | TechCorp Ltd |
| contact_name | text | Alisher Karimov |
| email | text | a.karimov@techcorp.com |
| phone | text | +998 90 123-45-67 |
| industry | text | Technology |
| budget | number | 50000 |
| deal_stage | text | Proposal |
| lead_source | text | LinkedIn |
| employees | number | 150 |
| website_visits | number | 35 |
| emails_opened | number | 12 |
| meetings_held | number | 3 |
| days_in_pipeline | number | 45 |

---

## 🤖 ML Model

- **Algoritm:** Gradient Boosting + Random Forest (Ensemble)
- **Features:** Budget, employees, engagement score, deal stage, industry, lead source
- **Output:** Conversion probability (0–100%), Priority (High/Medium/Low), Recommended price
- **Metric:** AUC-ROC score

---

## 🎯 Features

- ✅ CSV upload va sample data generation
- ✅ ML-based conversion probability prediction
- ✅ KPI metric cards (6 ta asosiy ko'rsatkich)
- ✅ Interactive charts (pie, bar, histogram, scatter, funnel)
- ✅ Sidebar filters (budget, industry, probability, priority)
- ✅ Sortable dataframe with search
- ✅ Lead detail view + Sales strategy tavsiya
- ✅ Feature importance visualization
- ✅ CSV export
- ✅ PostgreSQL + SQLite dual support
- ✅ Session state management