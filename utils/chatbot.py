"""
AI Sales Chatbot — Claude API + RAG (leads ma'lumotlari ustida)
"""
import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime


ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-opus-4-5"
MAX_TOKENS = 1024


def _get_api_key() -> str:
    """API key — Streamlit secrets yoki environment variable."""
    try:
        import streamlit as st
        return st.secrets.get("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY", ""))
    except Exception:
        return os.getenv("ANTHROPIC_API_KEY", "")


# ─── RAG: leads context yaratish ─────────────────────────────────────────────
def build_leads_context(df: pd.DataFrame) -> str:
    """
    Leads dataframe'dan chatbot uchun qisqa kontekst string yaratadi.
    Faqat eng muhim statistika + top leadlar kiritiladi (token limit uchun).
    """
    if df is None or len(df) == 0:
        return "Hozircha hech qanday lead ma'lumoti yuklanmagan."

    lines = ["=== LEADS MA'LUMOTLARI (RAG Context) ===\n"]

    # 1. Umumiy statistika
    total = len(df)
    lines.append(f"Jami leads: {total}")

    if 'priority' in df.columns:
        pc = df['priority'].value_counts()
        lines.append(f"Priority: High={pc.get('High',0)}, Medium={pc.get('Medium',0)}, Low={pc.get('Low',0)}")

    if 'budget' in df.columns:
        lines.append(f"Budget: min=${df['budget'].min():,.0f}, max=${df['budget'].max():,.0f}, avg=${df['budget'].mean():,.0f}")
        lines.append(f"Jami pipeline: ${df['budget'].sum():,.0f}")

    if 'conversion_probability' in df.columns:
        lines.append(f"Avg conversion probability: {df['conversion_probability'].mean()*100:.1f}%")

    if 'predicted_conversion' in df.columns:
        lines.append(f"Predicted conversion rate: {df['predicted_conversion'].mean()*100:.1f}%")

    if 'industry' in df.columns:
        top_ind = df['industry'].value_counts().head(5)
        lines.append(f"Top industries: {', '.join([f'{k}({v})' for k,v in top_ind.items()])}")

    if 'deal_stage' in df.columns:
        stage_cnt = df['deal_stage'].value_counts()
        lines.append(f"Deal stages: {dict(stage_cnt)}")

    if 'lead_source' in df.columns:
        top_src = df['lead_source'].value_counts().head(4)
        lines.append(f"Top sources: {', '.join([f'{k}({v})' for k,v in top_src.items()])}")

    lines.append("")

    # 2. Top 10 High priority leads
    if 'priority' in df.columns and 'conversion_probability' in df.columns:
        top_leads = (df[df['priority'] == 'High']
                     .sort_values('conversion_probability', ascending=False)
                     .head(10))
        if len(top_leads) > 0:
            lines.append("=== TOP HIGH PRIORITY LEADS ===")
            for _, row in top_leads.iterrows():
                prob    = row.get('conversion_probability', 0)
                budget  = row.get('budget', 0)
                company = row.get('company_name', '—')
                contact = row.get('contact_name', '—')
                industry= row.get('industry', '—')
                stage   = row.get('deal_stage', '—')
                rec_pr  = row.get('recommended_price', 0)
                lines.append(
                    f"• {company} | {contact} | {industry} | "
                    f"Budget: ${budget:,.0f} | Prob: {prob*100:.0f}% | "
                    f"Stage: {stage} | Rec.Price: ${rec_pr:,.0f}"
                )

    # 3. Industry breakdown (avg budget + avg prob)
    if all(c in df.columns for c in ['industry', 'budget', 'conversion_probability']):
        lines.append("\n=== INDUSTRY TAHLILI ===")
        ind_grp = df.groupby('industry').agg(
            count=('budget','count'),
            avg_budget=('budget','mean'),
            avg_prob=('conversion_probability','mean')
        ).sort_values('avg_prob', ascending=False)
        for ind, row in ind_grp.iterrows():
            lines.append(
                f"• {ind}: {int(row['count'])} leads | "
                f"Avg budget: ${row['avg_budget']:,.0f} | "
                f"Avg prob: {row['avg_prob']*100:.0f}%"
            )

    return "\n".join(lines)


# ─── SYSTEM PROMPT ────────────────────────────────────────────────────────────
def build_system_prompt(leads_context: str) -> str:
    return f"""Siz Lead Scoring & Pipeline Manager tizimining AI Sales Assistantsiz.

Sizning vazifangiz:
1. Leads ma'lumotlarini tahlil qilish va savollarga javob berish (RAG)
2. Sales strategiyasi bo'yicha maslahat berish
3. Conversion yaxshilash uchun tavsiyalar berish
4. Pipeline boshqarish haqida yordam ko'rsatish

MUHIM QOIDALAR:
- O'zbek, Rus yoki Ingliz tilida so'rashsa, o'sha tilda javob bering
- Aniq raqamlar va faktlarga asoslaning
- Qisqa, aniq va amaliy javob bering
- Leads ma'lumotlari bo'yicha savol bo'lsa, quyidagi kontekstdan foydalaning
- Agar ma'lumot kontekstda bo'lmasa, "Bu ma'lumot mavjud emas" deng

LEADS MA'LUMOTLARI KONTEKSTI:
{leads_context}

Bugungi sana: {datetime.now().strftime('%Y-%m-%d')}
"""


# ─── CLAUDE API CALL ─────────────────────────────────────────────────────────
def ask_claude(
    user_message: str,
    chat_history: list,
    leads_df: pd.DataFrame | None = None
) -> str:
    """
    Claude API ga so'rov yuboradi va javob qaytaradi.

    chat_history: [{"role": "user"|"assistant", "content": "..."}, ...]
    """
    api_key = _get_api_key()
    if not api_key:
        return ("❌ **ANTHROPIC_API_KEY topilmadi!**\n\n"
                "Streamlit Cloud → App Settings → Secrets bo'limiga:\n"
                "```\nANTHROPIC_API_KEY = \"sk-ant-...\"\n```\n"
                "yoki local `.streamlit/secrets.toml` ga qo'shing.")

    leads_context = build_leads_context(leads_df)
    system_prompt = build_system_prompt(leads_context)

    # History + yangi message
    messages = chat_history.copy()
    messages.append({"role": "user", "content": user_message})

    headers = {
        "x-api-key":         api_key,
        "anthropic-version": "2023-06-01",
        "content-type":      "application/json",
    }
    payload = {
        "model":      MODEL,
        "max_tokens": MAX_TOKENS,
        "system":     system_prompt,
        "messages":   messages,
    }

    try:
        resp = requests.post(ANTHROPIC_API_URL, headers=headers,
                             json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"]
    except requests.exceptions.Timeout:
        return "⏱️ So'rov vaqti tugadi. Qaytadan urinib ko'ring."
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response else "—"
        if status == 401:
            return "🔑 API key noto'g'ri yoki muddati tugagan."
        elif status == 429:
            return "⚠️ Rate limit. Bir oz kuting va qaytadan urinib ko'ring."
        return f"❌ API xatosi ({status}): {str(e)}"
    except Exception as e:
        return f"❌ Xato: {str(e)}"


# ─── QUICK SUGGESTIONS ────────────────────────────────────────────────────────
QUICK_QUESTIONS = [
    "📊 Eng yuqori priority leadlar kimlar?",
    "💰 Qaysi industry eng ko'p budget sarflaydi?",
    "📈 Conversion rate qanday yaxshilanadi?",
    "🎯 High priority leadlar uchun strategy nima?",
    "🔍 Negotiation stage'dagi leadlar soni?",
    "💡 Eng yaxshi lead source qaysi?",
]