import streamlit as st
from supabase import create_client

@st.cache_resource
def get_sb():
    # Don't do st.stop() here. Just raise.
    return create_client(
        st.secrets["supabase"]["url"],
        st.secrets["supabase"]["anon_key"],
    )

def sb_fetch_all(table: str, cols="*", page_size: int = 1000, max_retries: int = 5):
    sb = get_sb()
    out = []
    offset = 0

    while True:
        resp = sb.table(table).select(cols).range(offset, offset + page_size - 1).execute()
        rows = resp.data or []
        out.extend(rows)
        if len(rows) < page_size:
            break
        offset += page_size

    return out
