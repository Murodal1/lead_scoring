"""
Database Manager — PostgreSQL (production) yoki SQLite (local) ishlatadi.
"""
import os
import pandas as pd
import sqlite3
from datetime import datetime

try:
    import psycopg2
    from psycopg2.extras import execute_values
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False


class DatabaseManager:
    def __init__(self):
        self.db_url  = os.getenv("DATABASE_URL", "")
        self.use_pg  = bool(self.db_url and HAS_PSYCOPG2)
        self.sqlite_path = "leads.db"
        self._init_db()

    # ─── INIT ────────────────────────────────
    def _init_db(self):
        if self.use_pg:
            self._pg_init()
        else:
            self._sqlite_init()

    def _sqlite_init(self):
        conn = sqlite3.connect(self.sqlite_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS leads (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                company_name     TEXT,
                contact_name     TEXT,
                email            TEXT,
                phone            TEXT,
                industry         TEXT,
                budget           REAL,
                deal_stage       TEXT,
                lead_source      TEXT,
                employees        INTEGER,
                website_visits   INTEGER,
                emails_opened    INTEGER,
                meetings_held    INTEGER,
                days_in_pipeline INTEGER,
                conversion_probability REAL DEFAULT 0,
                priority         TEXT DEFAULT 'Low',
                recommended_price REAL DEFAULT 0,
                predicted_conversion INTEGER DEFAULT 0,
                created_at       TEXT,
                updated_at       TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _pg_init(self):
        conn = self._pg_conn()
        cur  = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS leads (
                id               SERIAL PRIMARY KEY,
                company_name     VARCHAR(255),
                contact_name     VARCHAR(255),
                email            VARCHAR(255),
                phone            VARCHAR(50),
                industry         VARCHAR(100),
                budget           NUMERIC,
                deal_stage       VARCHAR(100),
                lead_source      VARCHAR(100),
                employees        INTEGER,
                website_visits   INTEGER,
                emails_opened    INTEGER,
                meetings_held    INTEGER,
                days_in_pipeline INTEGER,
                conversion_probability NUMERIC DEFAULT 0,
                priority         VARCHAR(20) DEFAULT 'Low',
                recommended_price NUMERIC DEFAULT 0,
                predicted_conversion INTEGER DEFAULT 0,
                created_at       TIMESTAMP,
                updated_at       TIMESTAMP
            )
        """)
        conn.commit()
        cur.close(); conn.close()

    def _pg_conn(self):
        return psycopg2.connect(self.db_url)

    # ─── SAVE LEADS ──────────────────────────
    def save_leads(self, df: pd.DataFrame):
        """Leads'ni DB'ga saqlaydi (existing data o'chirilib qayta yoziladi)."""
        df = self._normalize_columns(df)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if 'created_at' not in df.columns:
            df['created_at'] = now
        df['updated_at'] = now

        if self.use_pg:
            self._pg_save(df)
        else:
            self._sqlite_save(df)

    def _sqlite_save(self, df: pd.DataFrame):
        conn = sqlite3.connect(self.sqlite_path)
        conn.execute("DELETE FROM leads")
        # Only columns that exist in table
        table_cols = [
            'company_name','contact_name','email','phone','industry','budget',
            'deal_stage','lead_source','employees','website_visits','emails_opened',
            'meetings_held','days_in_pipeline','created_at','updated_at'
        ]
        cols = [c for c in table_cols if c in df.columns]
        df[cols].to_sql('leads', conn, if_exists='append', index=False)
        conn.commit()
        conn.close()

    def _pg_save(self, df: pd.DataFrame):
        conn = self._pg_conn()
        cur  = conn.cursor()
        cur.execute("DELETE FROM leads")
        table_cols = [
            'company_name','contact_name','email','phone','industry','budget',
            'deal_stage','lead_source','employees','website_visits','emails_opened',
            'meetings_held','days_in_pipeline','created_at','updated_at'
        ]
        cols = [c for c in table_cols if c in df.columns]
        rows = [tuple(r) for r in df[cols].itertuples(index=False)]
        execute_values(cur,
            f"INSERT INTO leads ({','.join(cols)}) VALUES %s", rows)
        conn.commit()
        cur.close(); conn.close()

    # ─── SAVE PREDICTIONS ────────────────────
    def save_predictions(self, df: pd.DataFrame):
        """ML predictions'ni DB'ga yozadi."""
        if self.use_pg:
            conn = self._pg_conn()
            cur  = conn.cursor()
            for _, row in df.iterrows():
                cur.execute("""
                    UPDATE leads SET
                        conversion_probability = %s,
                        priority               = %s,
                        recommended_price      = %s,
                        predicted_conversion   = %s,
                        updated_at             = %s
                    WHERE company_name = %s
                """, (
                    float(row.get('conversion_probability', 0)),
                    str(row.get('priority', 'Low')),
                    float(row.get('recommended_price', 0)),
                    int(row.get('predicted_conversion', 0)),
                    datetime.now(),
                    row.get('company_name', '')
                ))
            conn.commit(); cur.close(); conn.close()
        else:
            conn = sqlite3.connect(self.sqlite_path)
            cur  = conn.cursor()
            for _, row in df.iterrows():
                cur.execute("""
                    UPDATE leads SET
                        conversion_probability = ?,
                        priority               = ?,
                        recommended_price      = ?,
                        predicted_conversion   = ?,
                        updated_at             = ?
                    WHERE company_name = ?
                """, (
                    float(row.get('conversion_probability', 0)),
                    str(row.get('priority', 'Low')),
                    float(row.get('recommended_price', 0)),
                    int(row.get('predicted_conversion', 0)),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    row.get('company_name', '')
                ))
            conn.commit(); conn.close()

    # ─── LOAD LEADS ──────────────────────────
    def load_leads(self) -> pd.DataFrame | None:
        try:
            if self.use_pg:
                conn = self._pg_conn()
                df   = pd.read_sql("SELECT * FROM leads ORDER BY id", conn)
                conn.close()
            else:
                conn = sqlite3.connect(self.sqlite_path)
                df   = pd.read_sql("SELECT * FROM leads ORDER BY id", conn)
                conn.close()
            return df if len(df) > 0 else None
        except Exception:
            return None

    # ─── HELPERS ─────────────────────────────
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Column nomlarini standartlashtiradi."""
        rename_map = {
            'Company':        'company_name',
            'company':        'company_name',
            'Contact':        'contact_name',
            'contact':        'contact_name',
            'Email':          'email',
            'Phone':          'phone',
            'Industry':       'industry',
            'Budget':         'budget',
            'Stage':          'deal_stage',
            'Deal Stage':     'deal_stage',
            'Source':         'lead_source',
            'Lead Source':    'lead_source',
            'Employees':      'employees',
            'Website Visits': 'website_visits',
            'Emails Opened':  'emails_opened',
            'Meetings':       'meetings_held',
            'Days':           'days_in_pipeline',
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        if 'budget' in df.columns:
            df['budget'] = pd.to_numeric(df['budget'], errors='coerce').fillna(0)
        return df