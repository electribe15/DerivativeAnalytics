#!/usr/bin/env python3
"""Lightweight Streamlit wrapper that password-protects and embeds the local Dash dashboard.

Usage (local/dev):
  1) Start the Dash server (in another terminal):
       python run_dex_gex_dashboard.py --port 8051 --ticker SPY

  2) Run this Streamlit wrapper:
       STREAMLIT_PASSWORD=mysecret streamlit run streamlit_dashboard.py

Notes:
 - For production, put a reverse proxy (nginx/Caddy) in front, enable HTTPS, and use proper auth.
 - This wrapper is convenient for local / internal use. Store secrets in environment variables or Streamlit secrets.
"""

import os
import time
import subprocess
import socket
import hashlib
import requests
import streamlit as st
import streamlit.components.v1 as components


def is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    """
    Lightweight Streamlit wrapper that password-protects and embeds the local Dash dashboard.

    This file includes a convenient local fallback password for quick local/internal use.
    Use of a hardcoded fallback is insecure for production — prefer `STREAMLIT_PASSWORD`
    or `STREAMLIT_PASSWORD_HASH` environment variables or `st.secrets`.

    Usage (local/dev):
      1) Start the Dash server (in another terminal):
          python run_dex_gex_dashboard.py --port 8051 --ticker SPY

      2) Run this Streamlit wrapper (preferred, using env var):
          STREAMLIT_PASSWORD=your_password streamlit run streamlit_dashboard.py

      3) Or run with the local fallback password (not recommended for shared systems):
          streamlit run streamlit_dashboard.py

    Notes:
     - For production, put a reverse proxy (nginx/Caddy) in front, enable HTTPS, and use proper auth.
     - This wrapper is convenient for local / internal use. Store secrets in environment variables or Streamlit secrets.
    """

def start_dash_server(port: int = 8051, host: str = '127.0.0.1') -> subprocess.Popen:
    """Optionally spawn the local Dash server in background using the repo runner.

    This is convenient for local testing but not recommended for production.
    """
    # Try to find python executable
    python = os.environ.get('PYTHON', 'python3')
    cmd = [python, 'run_dex_gex_dashboard.py', '--host', host, '--port', str(port)]
    # Start detached process
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return p


def password_hash(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def get_stored_pw_hash() -> str | None:
    # Priority: STREAMLIT_PASSWORD_HASH env -> STREAMLIT_PASSWORD env (hashed) -> st.secrets -> FALLBACK
    hp = os.environ.get('STREAMLIT_PASSWORD_HASH')
    if hp:
        return hp
    pw = os.environ.get('STREAMLIT_PASSWORD')
    if pw:
        return password_hash(pw)
    try:
        sec = st.secrets.get('streamlit', {})
        pw = sec.get('password')
        if pw:
            return password_hash(pw)
    except Exception:
        pass

    # LOCAL FALLBACK: use only for quick local testing. Replace or remove for any shared/production use.
    FALLBACK_PASSWORD = os.environ.get('STREAMLIT_FALLBACK_PASSWORD') or 'Mstazione2026!'
    # Warn in Streamlit when we rely on the fallback (displayed in UI later)
    return password_hash(FALLBACK_PASSWORD)


def main():
    st.set_page_config(page_title='DEX/GEX (Streamlit Proxy)', layout='wide')

    st.title('DEX / GEX Dashboard (Streamlit proxy)')
    st.markdown('This page embeds the local Dash dashboard behind a simple password gate.')

    host = st.sidebar.text_input('Dash host', value='127.0.0.1')
    port = st.sidebar.number_input('Dash port', value=8051, min_value=1024, max_value=65535)
    dash_url = f'http://{host}:{port}/'

    pw_hash = get_stored_pw_hash()
    if pw_hash is None:
        st.warning('No password configured. Set `STREAMLIT_PASSWORD` env or `STREAMLIT_PASSWORD_HASH` to enable gating.')

    # Optionally offer to start Dash for convenience
    if not is_port_open(host, port):
        if st.button('Start Dash server locally'):
            p = start_dash_server(port=port, host=host)
            st.info('Starting Dash server — give it a few seconds and then press Refresh')

    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False

    if not st.session_state['authenticated']:
        entered = st.text_input('Password', type='password')
        if st.button('Unlock'):
            if pw_hash is None:
                st.error('Password not configured on server. Please set STREAMLIT_PASSWORD environment variable.')
            elif password_hash(entered) == pw_hash:
                st.session_state['authenticated'] = True
                st.success('Authenticated')
            else:
                st.error('Incorrect password')
        st.stop()

    # Authenticated — check Dash availability
    st.sidebar.markdown('Dash URL: ' + dash_url)
    if check_dash_available(dash_url):
        st.success('Dash server available — embedding below')
    else:
        st.warning('Dash server not reachable. Ensure you started it (see sidebar).')

    # Embed via iframe
    components.iframe(dash_url, height=900)


if __name__ == '__main__':
    main()
