import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import warnings
warnings.filterwarnings('ignore')

from utils.database import DatabaseManager
from utils.ml_model import LeadScoringModel
from utils.data_generator import generate_sample_data
from utils.chatbot import ask_claude, QUICK_QUESTIONS
from datetime import datetime

st.set_page_config(
    page_title="🎯 Lead Scoring & Pipeline Manager",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 22px 30px; border-radius: 14px;
        margin-bottom: 25px; color: white; text-align: center;
    }
    .main-header h1 { margin:0; font-size:2.1rem; }
    .main-header p  { margin:6px 0 0; opacity:.85; font-size:.95rem; }
    .section-title  {
        font-size:1.05rem; font-weight:700; color:#2d3748;
        border-bottom:2px solid #e2e8f0; padding-bottom:7px; margin:22px 0 14px;
    }
    /* FLOAT CHAT BUTTON */
    #chat-fab {
        position:fixed; bottom:28px; right:28px;
        width:62px; height:62px;
        background:linear-gradient(135deg,#667eea,#764ba2);
        border-radius:50%; display:flex; align-items:center; justify-content:center;
        font-size:28px; color:white; cursor:pointer;
        box-shadow:0 4px 20px rgba(102,126,234,.55);
        z-index:9998; border:none;
        animation:chatpulse 2.8s infinite;
        transition:transform .2s;
    }
    #chat-fab:hover { transform:scale(1.12); }
    @keyframes chatpulse {
        0%,100%{ box-shadow:0 4px 20px rgba(102,126,234,.55); }
        50%     { box-shadow:0 4px 32px rgba(102,126,234,.9); }
    }
    .chat-msg-user {
        background:linear-gradient(135deg,#667eea,#764ba2);
        color:white; padding:10px 14px;
        border-radius:14px 14px 4px 14px;
        margin:5px 0 5px 40px; font-size:.91rem; line-height:1.5;
    }
    .chat-msg-bot {
        background:#f1f3f9; color:#2d3748;
        padding:10px 14px; border-radius:14px 14px 14px 4px;
        margin:5px 40px 5px 0; font-size:.91rem; line-height:1.5;
        border-left:3px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)


def get_strategy(row):
    prob     = float(row.get('conversion_probability', 0))
    budget   = float(row.get('budget', 0))
    stage    = str(row.get('deal_stage', ''))
    industry = str(row.get('industry', ''))
    if prob >= 0.7:
        return (f"🔥 **HIGH PRIORITY** — Ehtimol: {prob*100:.0f}%. "
                f"Darhol qo'ng'iroq qiling va demo taklif qiling. "
                f"Budget ${budget:,.0f} — aggressive pricing taklif qiling. Stage: {stage}.")
    elif prob >= 0.4:
        return (f"⚡ **MEDIUM PRIORITY** — {prob*100:.0f}% ehtimol. "
                f"Nurturing campaign boshlang, case study yuboring. "
                f"30 kun ichida follow-up rejalashtiring.")
    else:
        return (f"❄️ **LOW PRIORITY** — {prob*100:.0f}% ehtimol. "
                f"Email drip campaign'ga qo'shing. "
                f"3 oydan keyin qayta murojaat qiling yoki disqualify qiling.")


# Session state
for key, val in [('db', None), ('model', None), ('leads_df', None),
                 ('model_trained', False), ('export_triggered', False),
                 ('chat_open', False), ('chat_history', []),
                 ('chat_input_key', 0)]:
    if key not in st.session_state:
        st.session_state[key] = val

if st.session_state.db    is None: st.session_state.db    = DatabaseManager()
if st.session_state.model is None: st.session_state.model = LeadScoringModel()

db    = st.session_state.db
model = st.session_state.model

# HEADER
st.markdown("""
<div class="main-header">
    <h1>🎯 Lead Scoring & Pipeline Manager</h1>
    <p>ML-based conversion prediction · CRM tahlili · Sales strategy tavsiyalar</p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Boshqaruv Paneli")
    st.divider()
    st.markdown("### 📁 Ma'lumot Yuklash")
    t1, t2 = st.tabs(["📂 CSV", "🎲 Sample"])

    with t1:
        ufile = st.file_uploader("CSV fayl tanlang", type=['csv'])
        if ufile:
            df_up = pd.read_csv(ufile)
            db.save_leads(df_up)
            st.session_state.leads_df      = None
            st.session_state.model_trained = False
            st.success(f"✅ {len(df_up)} lead yuklandi!")

    with t2:
        n_s = st.slider("Lead soni", 50, 500, 150, 50)
        if st.button("🎲 Yaratish", use_container_width=True):
            sdf = generate_sample_data(n_s)
            db.save_leads(sdf)
            st.session_state.leads_df      = None
            st.session_state.model_trained = False
            st.success(f"✅ {n_s} ta lead yaratildi!")

    st.divider()

    raw = db.load_leads()
    if raw is not None and len(raw) > 0:
        if not st.session_state.model_trained:
            with st.spinner("🤖 ML model o'qitilmoqda..."):
                model.train(raw)
                preds = model.predict(raw)
                raw   = raw.copy()
                raw['conversion_probability'] = preds['probability']
                raw['priority']               = preds['priority']
                raw['recommended_price']      = preds['recommended_price']
                raw['predicted_conversion']   = preds['predicted']
                db.save_predictions(raw)
                st.session_state.leads_df    = raw
                st.session_state.model_trained = True
                if model.auc_score > 0:
                    st.success(f"✅ Model AUC: {model.auc_score:.3f}")
        elif st.session_state.leads_df is None:
            st.session_state.leads_df = raw

    df_main = st.session_state.leads_df

    # Filters
    if df_main is not None and len(df_main) > 0:
        st.markdown("### 🔍 Filterlar")
        b_min = int(df_main['budget'].min()) if 'budget' in df_main.columns else 0
        b_max = int(df_main['budget'].max()) if 'budget' in df_main.columns else 1000000
        budget_range = st.slider("💰 Budget ($)", b_min, b_max, (b_min, b_max), step=5000)

        all_ind = sorted(df_main['industry'].dropna().unique().tolist()) \
                  if 'industry' in df_main.columns else []
        sel_ind = st.multiselect("🏭 Industry", all_ind, default=all_ind)

        prob_range  = st.slider("📊 Conv. Probability (%)", 0, 100, (0, 100))
        sel_prior   = st.multiselect("🎯 Priority", ['High','Medium','Low'],
                                     default=['High','Medium','Low'])

        st.divider()
        if st.button("📥 CSV Export", use_container_width=True):
            st.session_state.export_triggered = True
        if st.button("♻️ Qayta O'qitish", use_container_width=True):
            st.session_state.model_trained = False
            st.rerun()

# ── MAIN ─────────────────────────────────────
df_main = st.session_state.leads_df
if df_main is None or len(df_main) == 0:
    st.info("👆 Sidebar'dan **Sample Data** yarating yoki **CSV** yuklang")
    st.stop()

# Apply filters
filtered = df_main.copy()
if 'budget' in filtered.columns:
    filtered = filtered[filtered['budget'].between(*budget_range)]
if 'industry' in filtered.columns and sel_ind:
    filtered = filtered[filtered['industry'].isin(sel_ind)]
if 'conversion_probability' in filtered.columns:
    filtered = filtered[
        filtered['conversion_probability'].between(prob_range[0]/100, prob_range[1]/100)
    ]
if 'priority' in filtered.columns and sel_prior:
    filtered = filtered[filtered['priority'].isin(sel_prior)]

# ── KPI ──────────────────────────────────────
st.markdown('<div class="section-title">📊 Asosiy KPI Ko\'rsatkichlar</div>',
            unsafe_allow_html=True)

total_l   = len(filtered)
high_p    = len(filtered[filtered['priority'] == 'High']) if 'priority' in filtered.columns else 0
conv_r    = filtered['predicted_conversion'].mean()*100 \
            if 'predicted_conversion' in filtered.columns else 0
total_v   = filtered['budget'].sum() if 'budget' in filtered.columns else 0
avg_pr    = filtered['conversion_probability'].mean()*100 \
            if 'conversion_probability' in filtered.columns else 0
exp_rev   = (filtered['budget'] * filtered['conversion_probability']).sum() \
            if all(c in filtered.columns for c in ['budget','conversion_probability']) else 0

k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("👥 Jami Leads",      f"{total_l:,}")
k2.metric("🔴 High Priority",   f"{high_p:,}",
          f"{high_p/total_l*100:.0f}%" if total_l else "0%")
k3.metric("📈 Conv. Rate",      f"{conv_r:.1f}%")
k4.metric("💰 Pipeline",        f"${total_v/1e6:.2f}M" if total_v>1e6 else f"${total_v:,.0f}")
k5.metric("🎯 Avg Probability", f"{avg_pr:.1f}%")
k6.metric("💡 Expected Rev.",   f"${exp_rev/1e6:.2f}M" if exp_rev>1e6 else f"${exp_rev:,.0f}")

st.divider()

# ── CHARTS ───────────────────────────────────
st.markdown('<div class="section-title">📈 Tahlil & Vizualizatsiya</div>',
            unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**Priority Distribution**")
    if 'priority' in filtered.columns:
        pc = filtered['priority'].value_counts().reset_index()
        pc.columns = ['Priority','Count']
        fig1 = px.pie(pc, names='Priority', values='Count', hole=0.42,
                      color='Priority',
                      color_discrete_map={'High':'#e63946','Medium':'#f77f00','Low':'#06d6a0'})
        fig1.update_layout(margin=dict(t=5,b=30,l=5,r=5), height=270,
                           legend=dict(orientation='h', y=-0.2))
        st.plotly_chart(fig1, use_container_width=True)

with c2:
    st.markdown("**Industry — Avg Budget**")
    if 'industry' in filtered.columns and 'budget' in filtered.columns:
        ib = filtered.groupby('industry')['budget'].mean().sort_values(ascending=False).head(7)
        fig2 = px.bar(x=ib.values, y=ib.index, orientation='h',
                      color=ib.values, color_continuous_scale='Viridis',
                      labels={'x':'Avg Budget ($)','y':''})
        fig2.update_layout(margin=dict(t=5,b=5,l=5,r=5), height=270,
                           coloraxis_showscale=False,
                           yaxis=dict(tickfont=dict(size=10)))
        st.plotly_chart(fig2, use_container_width=True)

with c3:
    st.markdown("**Conversion Probability**")
    if 'conversion_probability' in filtered.columns:
        fig3 = px.histogram(filtered, x='conversion_probability', nbins=25,
                            color_discrete_sequence=['#4361ee'],
                            labels={'conversion_probability':'Probability'})
        fig3.update_layout(margin=dict(t=5,b=5,l=5,r=5), height=270,
                           bargap=0.05, showlegend=False,
                           xaxis=dict(tickformat='.0%'))
        st.plotly_chart(fig3, use_container_width=True)

# Scatter
if all(c in filtered.columns for c in ['budget','conversion_probability','priority']):
    st.markdown("**Budget vs Conversion Probability**")
    hover_d = ['company_name','contact_name','industry'] \
              if 'company_name' in filtered.columns else None
    fig4 = px.scatter(filtered, x='budget', y='conversion_probability',
                      color='priority', size='budget', hover_data=hover_d, size_max=28,
                      color_discrete_map={'High':'#e63946','Medium':'#f77f00','Low':'#06d6a0'},
                      labels={'budget':'Budget ($)','conversion_probability':'Conv. Prob'})
    fig4.update_layout(height=350, margin=dict(t=10,b=10,l=10,r=10),
                       yaxis=dict(tickformat='.0%'))
    st.plotly_chart(fig4, use_container_width=True)

# Funnel
if 'deal_stage' in filtered.columns:
    stage_order = ['New','Qualified','Demo','Proposal','Negotiation']
    s_cnt = filtered['deal_stage'].value_counts().reindex(stage_order, fill_value=0)
    fig5  = px.funnel(x=s_cnt.values, y=s_cnt.index,
                      color_discrete_sequence=['#667eea'],
                      labels={'x':'Leads','y':'Stage'})
    fig5.update_layout(height=260, margin=dict(t=30,b=10,l=10,r=10),
                       title="Sales Pipeline Funnel")
    st.plotly_chart(fig5, use_container_width=True)

# ── DATAFRAME ────────────────────────────────
st.divider()
st.markdown('<div class="section-title">📋 Leads Jadvali</div>', unsafe_allow_html=True)

search_q = st.text_input("🔎 Qidirish (kompaniya, kontakt, industry)...", key="search_q")
disp_df  = filtered.copy()
if search_q:
    mask = pd.Series(False, index=disp_df.index)
    for col in ['company_name','contact_name','industry']:
        if col in disp_df.columns:
            mask |= disp_df[col].astype(str).str.contains(search_q, case=False, na=False)
    disp_df = disp_df[mask]

show_cols = [c for c in [
    'company_name','contact_name','industry','budget',
    'conversion_probability','priority','recommended_price',
    'lead_source','deal_stage','created_at'
] if c in disp_df.columns]

disp = disp_df[show_cols].copy()
if 'budget'                in disp.columns: disp['budget']                = disp['budget'].apply(lambda x: f"${x:,.0f}")
if 'recommended_price'     in disp.columns: disp['recommended_price']     = disp['recommended_price'].apply(lambda x: f"${x:,.0f}")
if 'conversion_probability' in disp.columns: disp['conversion_probability'] = disp['conversion_probability'].apply(lambda x: f"{x*100:.1f}%")

st.dataframe(disp, use_container_width=True, height=420,
             column_config={
                 'company_name':            st.column_config.TextColumn("🏢 Kompaniya"),
                 'contact_name':            st.column_config.TextColumn("👤 Kontakt"),
                 'industry':                st.column_config.TextColumn("🏭 Soha"),
                 'budget':                  st.column_config.TextColumn("💰 Budget"),
                 'conversion_probability':  st.column_config.TextColumn("📊 Conv. Prob"),
                 'priority':                st.column_config.TextColumn("🎯 Priority"),
                 'recommended_price':       st.column_config.TextColumn("💡 Rec. Price"),
                 'lead_source':             st.column_config.TextColumn("📌 Manba"),
                 'deal_stage':              st.column_config.TextColumn("📍 Stage"),
                 'created_at':              st.column_config.TextColumn("📅 Sana"),
             })
st.caption(f"🔢 {len(disp_df)} ta lead ko'rsatilmoqda")

# ── LEAD DETAIL ──────────────────────────────
st.divider()
st.markdown('<div class="section-title">🔍 Lead Tafsilotlari & Sales Strategy</div>',
            unsafe_allow_html=True)

if 'company_name' in filtered.columns:
    comp_list  = ['-- Tanlang --'] + sorted(filtered['company_name'].dropna().unique().tolist())
    sel_comp   = st.selectbox("Kompaniya tanlang:", comp_list)

    if sel_comp != '-- Tanlang --':
        row = filtered[filtered['company_name'] == sel_comp].iloc[0]

        with st.expander(f"📋 {sel_comp} — To'liq Ma'lumot", expanded=True):
            d1, d2, d3 = st.columns(3)
            with d1:
                st.markdown("**👤 Kontakt**")
                st.write(f"Ism: **{row.get('contact_name','—')}**")
                st.write(f"Email: {row.get('email','—')}")
                st.write(f"Telefon: {row.get('phone','—')}")
            with d2:
                st.markdown("**💼 Biznes**")
                st.write(f"Industry: **{row.get('industry','—')}**")
                st.write(f"Budget: **${row.get('budget',0):,.0f}**")
                st.write(f"Xodimlar: {row.get('employees','—')}")
                st.write(f"Stage: {row.get('deal_stage','—')}")
            with d3:
                st.markdown("**🤖 ML Prediction**")
                prob     = row.get('conversion_probability', 0)
                priority = row.get('priority', '—')
                rec_pr   = row.get('recommended_price', 0)
                st.metric("Conv. Probability", f"{prob*100:.1f}%")
                st.metric("Priority",          str(priority))
                st.metric("Rec. Price",        f"${rec_pr:,.0f}")

            st.markdown("---")
            st.markdown("**💡 Sales Strategy Tavsiya:**")
            st.info(get_strategy(row))

            st.markdown("**📊 Engagement:**")
            e1, e2, e3 = st.columns(3)
            e1.metric("🌐 Website Visits", int(row.get('website_visits', 0)))
            e2.metric("📧 Emails Opened",  int(row.get('emails_opened',  0)))
            e3.metric("🤝 Meetings Held",  int(row.get('meetings_held',  0)))

# ── MODEL INFO ───────────────────────────────
with st.expander("🤖 ML Model Ma'lumoti"):
    m1, m2, m3 = st.columns(3)
    m1.metric("Algoritm",  "GradientBoosting + RF Ensemble")
    m2.metric("AUC Score", f"{model.auc_score:.3f}" if model.auc_score else "—")
    m3.metric("Features",  str(len(model.feature_cols)) if model.feature_cols else "—")

    if model.is_trained:
        fi = model.feature_importance().head(8)
        if len(fi) > 0:
            fig_fi = px.bar(fi, x='importance', y='feature', orientation='h',
                            color='importance', color_continuous_scale='Blues',
                            title='Feature Importance (Top 8)')
            fig_fi.update_layout(height=280, margin=dict(t=40,b=10,l=10,r=10),
                                 coloraxis_showscale=False)
            st.plotly_chart(fig_fi, use_container_width=True)

# ── EXPORT ───────────────────────────────────
if st.session_state.export_triggered:
    buf = io.StringIO()
    filtered.to_csv(buf, index=False)
    st.download_button(
        "⬇️ CSV Yuklab Olish", buf.getvalue(),
        file_name=f"leads_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime='text/csv'
    )
    st.session_state.export_triggered = False

# ════════════════════════════════════════════════
# 💬 AI CHATBOT — FLOAT BUTTON
# ════════════════════════════════════════════════

# Toggle button (sidebar tugmasi orqali)
with st.sidebar:
    st.divider()
    chat_label = "💬 AI Chat — Yopish" if st.session_state.chat_open else "💬 AI Sales Assistant"
    if st.button(chat_label, use_container_width=True, type="primary"):
        st.session_state.chat_open = not st.session_state.chat_open
        st.rerun()

# Chatbot panel
if st.session_state.chat_open:
    st.divider()
    st.markdown("""
    <div style="
        background:linear-gradient(135deg,#667eea,#764ba2);
        color:white; padding:14px 20px; border-radius:14px 14px 0 0;
        font-weight:700; font-size:1.05rem; display:flex; align-items:center; gap:10px;
    ">
        🤖 AI Sales Assistant &nbsp;<span style="font-weight:400;font-size:.85rem;opacity:.85;">
        — Leads ma'lumotlari + Umumiy sales maslahat</span>
    </div>
    """, unsafe_allow_html=True)

    chat_container = st.container()

    # Chat history ko'rsatish
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""
            <div class="chat-msg-bot">
            👋 Salom! Men AI Sales Assistantman.<br><br>
            Menga leads ma'lumotlari haqida savol bering yoki
            sales strategiyasi bo'yicha maslahat so'rang.
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history:
                role = msg["role"]
                content = msg["content"]
                if role == "user":
                    st.markdown(f'<div class="chat-msg-user">👤 {content}</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-msg-bot">🤖 {content}</div>',
                                unsafe_allow_html=True)

    # Quick questions
    st.markdown("**⚡ Tezkor savollar:**")
    q_cols = st.columns(3)
    for i, q in enumerate(QUICK_QUESTIONS):
        with q_cols[i % 3]:
            if st.button(q, key=f"quick_{i}", use_container_width=True):
                # Quick question yuborish
                with st.spinner("🤖 Javob tayyorlanmoqda..."):
                    # Faqat content list (system prompt alohida)
                    history_for_api = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.chat_history
                    ]
                    answer = ask_claude(q, history_for_api,
                                       leads_df=st.session_state.leads_df)
                st.session_state.chat_history.append({"role": "user",    "content": q})
                st.session_state.chat_history.append({"role": "assistant","content": answer})
                st.rerun()

    # Input
    st.markdown("---")
    input_col, btn_col = st.columns([5, 1])
    with input_col:
        user_input = st.text_input(
            "Savolingizni yozing...",
            key=f"chat_input_{st.session_state.chat_input_key}",
            placeholder="Masalan: High priority leadlar uchun qanday strategy qo'llash kerak?",
            label_visibility="collapsed"
        )
    with btn_col:
        send_btn = st.button("📨", use_container_width=True,
                             help="Yuborish (Enter)")

    # Clear button
    cl1, cl2 = st.columns([4, 1])
    with cl2:
        if st.button("🗑️ Tozalash", use_container_width=True):
            st.session_state.chat_history   = []
            st.session_state.chat_input_key += 1
            st.rerun()

    # Send logic
    if (send_btn or user_input) and user_input.strip():
        with st.spinner("🤖 Javob tayyorlanmoqda..."):
            history_for_api = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.chat_history
            ]
            answer = ask_claude(
                user_input.strip(),
                history_for_api,
                leads_df=st.session_state.leads_df
            )
        st.session_state.chat_history.append({"role": "user",    "content": user_input.strip()})
        st.session_state.chat_history.append({"role": "assistant","content": answer})
        st.session_state.chat_input_key += 1
        st.rerun()

    # API key yo'q bo'lsa ogohlantirish
    import os
    try:
        import streamlit as _st
        api_key_exists = bool(_st.secrets.get("ANTHROPIC_API_KEY", ""))
    except Exception:
        api_key_exists = bool(os.getenv("ANTHROPIC_API_KEY", ""))

    if not api_key_exists:
        st.warning(
            "⚠️ **ANTHROPIC_API_KEY** topilmadi!\n\n"
            "`.streamlit/secrets.toml` ga qo'shing:\n"
            "```\nANTHROPIC_API_KEY = \"sk-ant-...\"\n```"
        )