import json
import pandas as pd
import streamlit as st
import plotly.express as px
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
import tempfile
from datetime import datetime
import pytz
from config_utils import load_timezone, save_timezone
from bs4 import BeautifulSoup
import base64
import re
import html
import ipaddress
from collections import Counter

BOX_CIDRS = [
    "74.112.184.0/22", "2620:117:b800::/37"
    # ... add more
]
BOX_NETWORKS = [ipaddress.ip_network(cidr) for cidr in BOX_CIDRS]

def is_box_ip(ip, box_networks=BOX_NETWORKS):
    try:
        ip_obj = ipaddress.ip_address(ip)
        return any(ip_obj in net for net in box_networks)
    except ValueError:
        return False

def check_box_com_proxy_usage(har_content, box_networks=BOX_NETWORKS):
    flagged = []
    entries = har_content['log']['entries']
    for entry in entries:
        url = entry['request']['url']
        ip = entry.get('serverIPAddress', '')
        ip = ip.strip('[]')  # Remove any brackets from IPv6 addresses
        m = re.search(r'https?://([a-zA-Z0-9.-]+\.(box\.com|boxcloud\.com))', url)
        if m and ip:
            domain = m.group(1)
            if not is_box_ip(ip, box_networks):
                flagged.append({
                    "domain": domain,
                    "ip": ip,
                    "url": url
                })
    return flagged

st.set_page_config(page_title="HAR File Analyzer for Product Support", layout="wide")
st.title("AI HAR File Analyzer for Product Support")

show_preview = st.checkbox("Show visual preview with keyword highlights")

def stringify_headers(headers):
    return ' '.join([f'{h.get("name")}: {h.get("value")}' for h in headers if h.get("name") and h.get("value")])

def match_keywords(text, keywords):
    matches = []
    for keyword in keywords:
        if keyword.lower() in text.lower():
            matches.append(keyword)
    return matches

def get_cookie_value(cookies, name="uid"):
    for cookie in cookies:
        if cookie.get('name') == name:
            return cookie.get('value')
    return None

def stringify_cookies(cookies):
    if not cookies:
        return ""
    # Example output: uid=12345; session=abcdef;
    return "; ".join(
        f"{cookie.get('name')}={cookie.get('value')}"
        for cookie in cookies
        if cookie.get('name') is not None and cookie.get('value') is not None
    )

# 1. Highlighting for TEXT (plain, CSS, JSON)

# 2. Highlighting for HTML or JSON, skipping <style> blocks for HTML
def highlight_keywords_in_html_or_json(text, keywords, skip_inside_words=True, mime_type=None):
    # If it's not a string, don't modify
    if not isinstance(text, str):
        return text, False

    # For JSON, pretty-print and highlight
    is_json = False
    if mime_type and mime_type.startswith('application/json'):
        try:
            obj = json.loads(text)
            text = json.dumps(obj, indent=2)
            is_json = True
        except Exception:
            pass

    # For text/css: skip highlighting
    if mime_type and mime_type.startswith('text/css'):
        return text, False  # Don't highlight in CSS files!

    # For HTML, remove <style> blocks before highlighting
    if mime_type and mime_type.startswith('text/html'):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(text, 'html.parser')
        for style_tag in soup.find_all('style'):
            style_tag.decompose()
        text = str(soup)

    # Now highlight keywords as usual
    highlighted = text
    for kw in keywords:
        if skip_inside_words:
            pattern = rf"(?<![A-Za-z0-9_])({re.escape(kw)})(?![A-Za-z0-9_])"
        else:
            pattern = rf"({re.escape(kw)})"
        highlighted = re.sub(pattern, r'<mark>\1</mark>', highlighted, flags=re.IGNORECASE)

    # Add <style> for <mark> tag
    style = "<style>mark { background-color: red; color: black; }</style>"
    return style + highlighted, is_json

# Keyword configuration
custom_keywords = st.text_area("Alert Keywords (comma-separated) - CMD-ENTER to apply", "Unauthorized, block, blocked, fail, failed, firewall, unable, uploadfailed, errorpage, error, zscaler, zscalertwo")
alert_keywords = [kw.strip().lower() for kw in custom_keywords.split(',') if kw.strip()]

skip_longer_word_matches = st.checkbox(
    "Skip keyword matches inside longer words (e.g. skip 'Blocked' in 'isDownloadBlockedByShieldAccessPolicy')",
    value=False
)

@st.cache_data
def parse_har_data(har_content, show_preview=False, alert_keywords=None, skip_longer_word_matches=True):
    if alert_keywords is None:
        alert_keywords = []
    entries = har_content['log']['entries']
    docs = []
    data = []
    alerts = []
    previews = []

    def decode_response(entry):
        try:
            content = entry['response'].get('content', {})
            mime_type = content.get('mimeType', '')
            text = content.get('text', '')
            encoding = content.get('encoding', '')

            if mime_type.startswith('text/html'):
                if encoding == 'base64':
                    decoded_bytes = base64.b64decode(text)
                    html = decoded_bytes.decode('utf-8', errors='ignore')
                else:
                    html = text
                soup = BeautifulSoup(html, 'html.parser')
                return soup.prettify()
            elif mime_type == 'application/json':
                if encoding == 'base64':
                    decoded_bytes = base64.b64decode(text)
                    return decoded_bytes.decode('utf-8', errors='ignore')
                return text
        except Exception as e:
            return f"Error decoding response: {e}"
        return "Not a supported response type."

    def extract_keyword_contexts(text, keywords, window=40, skip_inside_words=True):
        matches = []
        for keyword in keywords:
            if skip_inside_words:
                # Match only if NOT part of a larger alphanumeric or underscore word
                pattern = rf"(?<![A-Za-z0-9_])({re.escape(keyword)})(?![A-Za-z0-9_])"
            else:
                pattern = rf"({re.escape(keyword)})"
            for match in re.finditer(rf".{{0,{window}}}{pattern}.{{0,{window}}}", text, flags=re.IGNORECASE):
                ctx = re.sub(pattern, r"<mark>\1</mark>", match.group(0), flags=re.IGNORECASE)
                matches.append((keyword, ctx))
        return matches

    for i, entry in enumerate(entries):
        request = entry['request']
        response = entry['response']
        timings = entry.get('timings', {})
        time = entry.get('time')
        startedDateTime = entry.get('startedDateTime')

        url = request.get('url')
        method = request.get('method')
        status = response.get('status')
        status_text = response.get('statusText')

        response_headers = {h['name'].lower(): h['value'] for h in response.get('headers', [])}
        via = response_headers.get('via')
        server = response_headers.get('server')
        content_type = response_headers.get('content-type')
        ip_address = entry.get('serverIPAddress')

        query_params = request.get('queryString', [])
        request_cookies = request.get('cookies', [])
        response_cookies = response.get('cookies', [])
        uid_request = get_cookie_value(request_cookies, "uid")
        post_data = request.get('postData', {}).get('text', '')
        decoded_body = decode_response(entry)

        lower_body = decoded_body.lower() if isinstance(decoded_body, str) else ''
        highlighted_body = decoded_body

        safe_domains = {"pendo-data-prod.box.com", "pendo-prod.box.com", "cdn01.boxcdn.net"}
        domain = url.split('/')[2] if url and '//' in url else ''
        matches = []
        if (
            'text/css' not in (content_type or '') and
            domain not in safe_domains and
            not url.startswith("https://app.box.com/app-api/split-proxy/api/splitChanges")
        ):
            matches = extract_keyword_contexts(
                lower_body,
                alert_keywords,
                skip_inside_words=skip_longer_word_matches
            )

            if matches:
                match_texts = [f"- '{kw}': ...{ctx}..." for kw, ctx in matches]
                alert_msg = f"\U000026A0 Alert in Entry #{i}: Found keywords:\n" + "\n".join(match_texts)
                alerts.append(alert_msg)
                for kw, _ in matches:
                    highlighted_body = re.sub(rf"(?i)({re.escape(kw)})", r"<mark>\1</mark>", highlighted_body)
                highlighted_body = f"<style>mark {{ background-color: orange; color: black; }}</style>{highlighted_body}"

        text = f"""
        Entry #{i}
        [Request]
        URL: {url}
        Method: {method}
        Query Params: {query_params}
        Request Cookies: {request_cookies}
        Response Cookies: {response_cookies}
        Post Data: {post_data}
        Headers: {request.get('headers')}

        [Response]
        Status: {status} {status_text}
        Headers: {response.get('headers')}
        Time: {time} ms
        Timings: {timings}

        [Decoded Body Preview]
        {highlighted_body[:1000] if highlighted_body else 'N/A'}
        """
        docs.append(Document(page_content=text.strip()))

        if matches and show_preview:
            previews.append({
                "index": i,
                "highlighted_body": highlighted_body
            })

        data.append({
            "index": i,
            "url": url,
            "method": method,
            "status": status,
            "status_text": status_text,
            "time_ms": time,
            "startedDateTime": startedDateTime,
            "domain": domain,
            "ip_address": ip_address,
            "uid_request_cookie": uid_request,
            "response_cookies": response_cookies,
            "Server": server,
            "via": via,
            "content_type": content_type,
            "response_headers": response_headers,
            "request_headers": request.get('headers', [])
        })

# Build DataFrame:
    df = pd.DataFrame(data)
    df["startedDateTime"] = pd.to_datetime(df["startedDateTime"], errors='coerce')
    if df["startedDateTime"].dt.tz is None:
        df["startedDateTime"] = df["startedDateTime"].apply(lambda x: x.tz_localize('UTC') if x.tzinfo is None else x)

    df["request_headers_str"] = df["request_headers"].apply(stringify_headers)
    df["header_matches"] = df["request_headers_str"].apply(lambda text: match_keywords(text, alert_keywords))

    if "post_data" not in df.columns:
        df["post_data"] = ""
    df["post_data_matches"] = df["post_data"].apply(lambda text: match_keywords(str(text), alert_keywords))
    df["response_cookies_str"] = df["response_cookies"].apply(stringify_cookies)
    return docs, df, previews, alerts

def decode_html_response(entry):
    try:
        content = entry['response'].get('content', {})
        mime_type = content.get('mimeType', '')
        text = content.get('text', '')
        encoding = content.get('encoding', '')

        if encoding == 'base64':
            decoded_bytes = base64.b64decode(text)
            decoded_text = decoded_bytes.decode('utf-8', errors='ignore')
        else:
            decoded_text = text

        if mime_type.startswith('text/html'):
            soup = BeautifulSoup(decoded_text, 'html.parser')
            for tag in soup(['script', 'style']):
                tag.decompose()
            return soup.prettify(), mime_type

        elif mime_type == 'application/json':
            parsed_json = json.loads(decoded_text)
            return json.dumps(parsed_json, indent=2), mime_type
        
        elif mime_type.startswith('text/css'):
            return decoded_text, mime_type
        elif mime_type == 'application/javascript' or mime_type.startswith('application/javascript'):
            return decoded_text, mime_type

    except Exception as e:
        return f"Error decoding content in HAR file: {e}", None

    return "Unsupported or empty response.", None

uploaded_file = st.file_uploader("Upload a HAR file", type="har")

if uploaded_file:
    har_content = json.load(uploaded_file)
    docs, df, previews, alerts = parse_har_data(
        har_content, 
        show_preview,
        alert_keywords,
        skip_longer_word_matches
    )

# BOX IP PROXY CHECK
    flagged = check_box_com_proxy_usage(har_content)
    if flagged:
        with st.expander("Some Box domains use unexpected IP addresses (possible customer proxy/firewall detected):", expanded=True):
            st.warning("Review the flagged domains below:")
            for f in flagged:
                st.write(f"- {f['domain']} at IP {f['ip']} (URL: {f['url']})")

# --- Collapsible summary of domains with missing/empty IP addresses ---
    empty_ip_entries = [
        {
            "domain": entry['request']['url'].split('/')[2] if '//' in entry['request']['url'] else entry['request']['url'],
            "url": entry['request']['url']
        }
        for entry in har_content['log']['entries']
        if (
            not entry.get('serverIPAddress') and
            not entry['request']['url'].startswith("http://127.0.0.1") 
        )
    ]
    if empty_ip_entries:
        st.warning("Some requests have a missing or empty server IP address (possible proxy, VPN, or privacy block):")
        # Count unique domains and show count
        domain_counts = Counter(e['domain'] for e in empty_ip_entries)
        for domain, count in domain_counts.items():
            st.write(f"- {domain} ({count})")
    
    st.subheader("Filters")

    stored_timezone = load_timezone()
    all_timezones = pytz.all_timezones
    selected_timezone = st.selectbox(
        "Change/Select Time Zone",
        options=all_timezones,
        index=all_timezones.index(st.session_state.get("user_timezone", stored_timezone)),
        key="user_timezone"
    )
    save_timezone(st.session_state.user_timezone)

    df["startedDateTime"] = df["startedDateTime"].dt.tz_convert(selected_timezone)
    st.caption(f"Timestamps shown in: **{selected_timezone}**")

    unique_domains = sorted(df['domain'].dropna().unique())
    unique_statuses = sorted(df['status'].dropna().unique())
    unique_methods = sorted(df['method'].dropna().unique())

    selected_domains = st.multiselect("Filter by Domain - Click to hide from results", unique_domains, default=unique_domains)
    selected_statuses = st.multiselect("Filter by Status Code", unique_statuses, default=unique_statuses)
    selected_methods = st.multiselect("Filter by HTTP Method", unique_methods, default=unique_methods)

    min_time, max_time = df["startedDateTime"].dropna().min(), df["startedDateTime"].dropna().max()
    if pd.notnull(min_time) and pd.notnull(max_time):
        st.info(f"Time Range covered in HAR: **{min_time.strftime('%Y-%m-%d %H:%M:%S %Z')}** to **{max_time.strftime('%Y-%m-%d %H:%M:%S %Z')}**")
        # Use the full range automatically for filtering
        selected_time_range = (min_time.to_pydatetime(), max_time.to_pydatetime())
    else:
        st.info("Time Range covered in HAR: **no valid datetimes found**")
        selected_time_range = (datetime.now(), datetime.now())

# --- Search controls ---
    search_query = st.text_input("Search for any string in the URL")
    search_all = st.checkbox("Search entire HAR entry (all columns, including URL)?", value=False)
    use_regex = st.checkbox("Use REGEX in search?", value=False)

    filtered_df = df[
        df['domain'].isin(selected_domains) &
        df['status'].isin(selected_statuses) &
        df['method'].isin(selected_methods) &
        df['startedDateTime'].between(*selected_time_range)
    ].copy()

# --- Filtering with error handling for invalid regex ---
    if search_query:
        try:
            if search_all:
                # Search across all columns for any match in the row
                mask = filtered_df.apply(
                    lambda row: row.astype(str).str.contains(
                        search_query, case=False, na=False, regex=use_regex
                    ).any(),
                    axis=1
                )
                filtered_df = filtered_df[mask]
            else:
                filtered_df = filtered_df[
                    filtered_df['url'].str.contains(
                        search_query, case=False, na=False, regex=use_regex
                    )
                ]
        except re.error as e:
            st.error(f"Invalid REGEX: {e}")

    filtered_df = filtered_df.reset_index(drop=True)

    def prettify_matches(matches):
        if matches:
            return ", ".join(f"🟡 {m}" for m in matches)
        return ""

# Debug: prints to terminal:
    print(filtered_df["header_matches"].head())
#    print(filtered_df["post_data_matches"].head())

    filtered_df["header_matches_display"] = filtered_df["header_matches"].apply(prettify_matches)
    filtered_df["post_data_matches_display"] = filtered_df["post_data_matches"].apply(prettify_matches)

    def extract_headers_if_error(row):
        if row.get('status', 0) > 399:
            headers = row.get('response_headers', {})
            if isinstance(headers, dict):
                keys_of_interest = ["server", "content-type"]  # Add more if you wish
                items = [f"{k}: {v}" for k, v in headers.items() if k.lower() in keys_of_interest]
                return ", ".join(items)
        return ""

    filtered_df["error_response_headers"] = filtered_df.apply(extract_headers_if_error, axis=1)


#    print(filtered_df.columns)  # This will show all column names

    st.subheader("Parsed HAR Entries")
    st.caption("Double-click any long fields to expand and see more details. Alert Keywords may appear header_matches column.")

    def highlight_status(val):
        if isinstance(val, (int, float)):
            if val >= 500:
                return 'color: red; font-weight: bold; background-color: #ffff00'  # Yellow background
            elif val >= 400:
                return 'color: orange; font-weight: bold; background-color: #ffff00'
            elif val >= 307: # Flag a 307 in orange as it can be a proxy causing issues.
                return 'color: orange; font-weight: bold; background-color: #ffff00'
            elif val >= 300:
                return 'color: lightblue'
            elif val >= 200:
                return 'color: green'
        return ''

    def highlight_slow(val):
        try:
            if val > filtered_df['time_ms'].quantile(0.9):
                return 'background-color: #ffcccc'
        except:
            return ''
        return ''

    # Prepare columns for main HAR table display
    display_cols = list(filtered_df.columns)

    # Remove technical/raw columns from visible table
    cols_to_remove = [
        "request_headers",
        "request_headers_str",
        "response_cookies",
        "response_cookies_str",
        "header_matches"
    ]
    for col in cols_to_remove:
        if col in display_cols:
            display_cols.remove(col)
    
    # Insert stringified cookies column after uid_request_cookie
    if "uid_request_cookie" in display_cols:
        idx = display_cols.index("uid_request_cookie") + 1
        display_cols.insert(idx, "response_cookies_str")
    else:
        display_cols.append("response_cookies_str")
    
    # Remove any duplicate 'index' columns
    while display_cols.count("index") > 1:
        display_cols.remove("index")

    st.dataframe(
        filtered_df[display_cols].style
            .map(highlight_status, subset=['status'])
            .map(highlight_slow, subset=['time_ms']),
        use_container_width=True,
        hide_index=True
    )

    st.download_button("Download CSV", filtered_df.to_csv(index=False), file_name="har_entries.csv")

    st.subheader("Status Code Distribution")
    fig = px.histogram(filtered_df, x="status", title="Status Code Distribution", labels={"status": "HTTP Status Code"})
    st.plotly_chart(fig)

    st.subheader("Request Timeline")
    df_sorted = filtered_df.sort_values("startedDateTime")
    timeline_fig = px.line(df_sorted, x="startedDateTime", y="time_ms", title="Request Duration Over Time")
    st.plotly_chart(timeline_fig)

    st.subheader("Top Endpoints")
    filtered_df = filtered_df.copy()
    filtered_df['endpoint'] = filtered_df['url'].apply(lambda x: x.split('?')[0] if x else x)
    top_endpoints = filtered_df['endpoint'].value_counts().nlargest(10).reset_index()
    top_endpoints.columns = ['endpoint', 'count']
    endpoint_fig = px.bar(top_endpoints, x='endpoint', y='count', title="Top Requested Endpoints")
    st.plotly_chart(endpoint_fig)

    st.subheader("Slow Requests - sorted by time_ms")
    slow_requests = filtered_df[filtered_df['time_ms'] > filtered_df['time_ms'].quantile(0.9)]
    slow_requests = slow_requests.sort_values("time_ms", ascending=False).reset_index(drop=True)
    st.dataframe(slow_requests, use_container_width=True, hide_index=True)

    st.subheader("Additional Visual Summaries")
    col1, col2 = st.columns(2)
    with col1:
        avg_response_by_method = filtered_df.groupby("method")["time_ms"].mean().reset_index()
        fig_method_avg = px.bar(avg_response_by_method, x="method", y="time_ms", title="Average Response Time by Method")
        st.plotly_chart(fig_method_avg)
    with col2:
        count_by_domain = filtered_df["domain"].value_counts().reset_index()
        count_by_domain.columns = ["domain", "count"]
        fig_domain = px.pie(count_by_domain, names="domain", values="count", title="Requests by Domain")
        st.plotly_chart(fig_domain)

    st.subheader("Ask Questions About the HAR File")
    if 'last_question' not in st.session_state:
        st.session_state['last_question'] = ''
        st.session_state['last_answer'] = ''
        st.session_state['har_qa_input'] = ''
        st.session_state['matched_status_code'] = None
        st.session_state['matched_df_csv'] = None
    
    # Clear button (outside the form)
    clear_q = st.button("Clear Question")
    if clear_q:
        st.session_state['last_question'] = ''
        st.session_state['last_answer'] = ''
        st.session_state['har_qa_input'] = ''
        st.session_state['matched_status_code'] = None
        st.session_state['matched_df_csv'] = None
    
    with st.form("har_qa_form"):
        user_question = st.text_input("Enter your question about the HAR file", value=st.session_state['last_question'], key="har_qa_input")
        k_value = st.number_input("Number of relevant chunks to use (k). Lower only if errors occur.", min_value=1, max_value=20, value=3, step=1)
        submit_q = st.form_submit_button("Ask Question")
                
        status_explanations = {
            # 3xx explanations
            307: "This redirect may indicate a customer's proxy",
            # 4xx explanations
            400: "Bad Request: The server could not understand the request due to invalid syntax.",
            401: "Unauthorized: Authentication is required and has failed or has not yet been provided.",
            403: "Forbidden: The client does not have access rights to the content.",
            404: "Not Found: The server cannot find the requested resource.",
            407: "Proxy Authentication Required",
            408: "Request Timeout: The server timed out waiting for the request.",
            409: "Conflict: a request conflicts with the current status of the server.",
            # 5xx explanations
            500: "Internal Server Error: The server encountered an unexpected condition. NOTE: A customer's firewall or proxy can also generate this error code. If the 'Server' or 'via' columns are empty, this is more evidence it's not a Box-generated error.",
            501: "Not Implemented: The server does not support the functionality required.",
            502: "Bad Gateway: The server received an invalid response from the upstream server. NOTE: A customer's firewall or proxy can also generate this error code. If the 'Server' or 'via' columns are empty, this is more evidence it's not a Box-generated error.",
            503: "Service Unavailable: The server is temporarily unable to handle the request. NOTE: A customer's firewall or proxy can also generate this error code. If the 'Server' or 'via' columns are empty, this is more evidence it's not a Box-generated error.",
            504: "Gateway Timeout: The server was acting as a gateway or proxy and did not receive a timely response.",
            # ... add more as needed ...
        }
        generic_5xx = "5xx codes indicate server errors (an error occurred on the server side)."
        generic_4xx = "4xx codes indicate client errors (the request is incorrect or cannot be fulfilled by the server)."
        
        if submit_q and user_question:
            answer = None
            # Regex: matches 4xx or 5xx codes in the question
            status_pattern = re.compile(r'(show|list|display|find|status)?[^\d]*(4\d\d|5\d\d)', re.I)
            match = status_pattern.search(user_question)
            if match:
                status_code = int(match.group(2))
                filtered_status_df = filtered_df[filtered_df['status'] == status_code]
                st.markdown(f"**Table of Requests with Status {status_code}:**")
                st.dataframe(
                    filtered_status_df,
                    use_container_width=True,
                    hide_index=True
                )
                st.session_state['matched_status_code'] = status_code
                st.session_state['matched_df_csv'] = filtered_status_df.to_csv(index=False)
                # Pick the right explanation
                explanation = status_explanations.get(
                    status_code,
                    generic_5xx if 500 <= status_code < 600 else generic_4xx
                )
                st.info(f"**{status_code} Meaning:** {explanation}")
                answer = f"Displayed {len(filtered_status_df)} requests with status {status_code}. {explanation}"
                st.session_state['last_question'] = user_question
                st.session_state['last_answer'] = answer
            else:
                # --- Fallback: Use LLM (LangChain QA), but prevent token overflow ---
                with st.spinner("Thinking..."):
                    embeddings = OpenAIEmbeddings()
                    from langchain.text_splitter import CharacterTextSplitter
                    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                    chunks = splitter.split_documents(docs)
                    max_chunks = 50  # or lower if you need extra safety
                    if len(chunks) > max_chunks:
                        st.info(f"Using only the first {max_chunks} most relevant chunks to avoid exceeding model limits.")
                    chunks = chunks[:max_chunks]
                    capped_k = min(int(k_value), len(chunks), 5)
                    with tempfile.TemporaryDirectory() as temp_dir:
                        vectorstore = FAISS.from_documents(chunks, embeddings)
                        retriever = vectorstore.as_retriever(search_kwargs={"k": capped_k})
                        qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever)
                        response = qa.invoke(user_question)
                    st.session_state['last_question'] = user_question
                    st.session_state['last_answer'] = response
                st.session_state['matched_status_code'] = None
                st.session_state['matched_df_csv'] = None

    
    # ---- Show download button after the form ----
    if st.session_state.get('matched_status_code') is not None and st.session_state.get('matched_df_csv') is not None:
        st.download_button(
            f"Download status {st.session_state['matched_status_code']} requests as CSV",
            st.session_state['matched_df_csv'],
            file_name=f"har_status_{st.session_state['matched_status_code']}.csv"
        )
    
    if st.session_state['last_answer']:
        answer = st.session_state['last_answer']
        if isinstance(answer, dict) and "result" in answer:
            st.markdown(f"**Answer:** {answer['result']}")
        else:
            st.markdown(f"**Answer:** {answer}")


    if show_preview and previews:
        st.subheader("Keyword Matches (Preview)")
        for preview in previews:
            st.markdown(f"**Entry #{preview['index']}**", help="Highlighted keyword matches", unsafe_allow_html=True)
            st.components.v1.html(preview['highlighted_body'], height=300, scrolling=True)

    st.subheader("Decode HTML/JSON/CSS/Text Response from server")
    selected_index = st.number_input(
        "Enter Index from 'Parsed HAR Entries' table to decode a specific Server Response",
        min_value=0,
        max_value=len(har_content['log']['entries'])-1,
        step=1
    )
    show_only_matches = st.checkbox("Show only if keyword(s) match", value=False, key="show_only_matches")

    if st.button("Decode HTML/JSON/CSS/Text Response"):
        decoded, mime_type = decode_html_response(har_content['log']['entries'][selected_index])
        
        # --- Only run highlighting for HTML/JSON ---
        if mime_type is not None and mime_type.startswith("application/json"):
            highlighted, is_json = highlight_keywords_in_html_or_json(
                decoded,
                alert_keywords,
                skip_inside_words=skip_longer_word_matches
            )
        elif mime_type is not None and mime_type.startswith("text/html"):
            highlighted, is_json = highlight_keywords_in_html_or_json(
                decoded,
                alert_keywords,
                skip_inside_words=skip_longer_word_matches
            )
        else:
            # For text/css and any other type, skip highlighting
            highlighted, is_json = decoded, False
        
        # Detect if any keywords matched (only for HTML/JSON)
        def match_found(decoded, keywords, skip_inside_words):
            for kw in keywords:
                if skip_inside_words:
                    pattern = rf"(?<![A-Za-z0-9_]){re.escape(kw)}(?![A-Za-z0-9_])"
                else:
                    pattern = rf"{re.escape(kw)}"
                if re.search(pattern, decoded, flags=re.IGNORECASE):
                    return True
            return False
        
        entry = har_content['log']['entries'][selected_index]
        
        status = entry['response'].get('status')
        content_type = entry['response'].get('content', {}).get('mimeType')
        response_headers = {h['name']: h['value'] for h in entry['response'].get('headers', [])}

        response = entry.get('response', {})
        headers = {h['name'].lower(): h['value'] for h in response.get('headers', [])}
        # Select only specific headers:
        keys_of_interest = ["server", "content-type"]
        display_headers = {k: v for k, v in headers.items() if k in keys_of_interest}

        
        # Only apply for HTML/JSON, skip CSS
        if mime_type is not None and (
            mime_type.startswith("application/json") or
            mime_type.startswith("text/html") or
            mime_type.startswith("application/javascript")
        ):
            matched = match_found(decoded, alert_keywords, skip_longer_word_matches)
        else:
            matched = False
        
        if show_only_matches and not matched:
            st.info("No keyword match found in this decoded response.")
        else:
            st.subheader("Decoded HTML/JSON/CSS/Text Response (Preview)")
            try:
                if mime_type is not None and mime_type.startswith("application/json"):
                    st.markdown(highlighted, unsafe_allow_html=True)
                elif mime_type is not None and mime_type.startswith("text/html"):
                    st.components.v1.html(highlighted, height=400, scrolling=True)
                elif mime_type is not None and mime_type.startswith("application/javascript"):
                    st.code(decoded if decoded and decoded.strip() else "<empty response>", language='javascript')
                elif mime_type is not None and mime_type.startswith("text/css"):
                    st.code(decoded if decoded and decoded.strip() else "<empty response>", language='css')
                else:
                    st.code(decoded if decoded and decoded.strip() else "<empty response>")
            except Exception as e:
                st.error(f"Error rendering preview: {e}")

#zzzzz    
            with st.expander("Decoded HTML/JSON/CSS/Text Response (Code View) - Click to expand/collapse"):
                from bs4 import BeautifulSoup
                import html as html_mod
            
                view_as_code = st.checkbox(
                    "View as raw HTML code instead of rendered HTML",
                    value=False,
                    key="view_as_code_toggle"
                )
            
                def format_header_context(status, content_type, response_headers):
                    lines = [
                        f"Status: {status}" if status is not None else "Status: (unknown)",
                        f"Content-Type: {content_type}" if content_type else "Content-Type: (unknown)",
                    ]
                    if response_headers:
                        lines.append("Headers:")
                        for k, v in response_headers.items():
                            lines.append(f"  {k}: {v}")
                    return "\n".join(lines)
            
                if mime_type is not None and mime_type.startswith("application/json"):
                    def escape_but_keep_mark(text):
                        safe = html_mod.escape(text)
                        safe = safe.replace('&lt;mark&gt;', '<mark>').replace('&lt;/mark&gt;', '</mark>')
                        return safe
                    safe_highlighted = escape_but_keep_mark(highlighted)
                    if view_as_code:
                        code_to_show = decoded if decoded and decoded.strip() else "<empty response>"
                        context = format_header_context(status, content_type, response_headers)
                        st.code(f"{context}\n\n{code_to_show}", language="json")
                    else:
                        st.markdown(f"<pre>{safe_highlighted}</pre>", unsafe_allow_html=True)
            
                elif mime_type is not None and mime_type.startswith("text/html"):
                    soup = BeautifulSoup(highlighted, 'html.parser')
                    body = soup.body
                    if view_as_code:
                        code_to_show = (
                            decoded if decoded and decoded.strip()
                            else highlighted if highlighted and highlighted.strip()
                            else "<empty response>"
                        )
                        context = format_header_context(status, content_type, response_headers)
                        st.code(f"{context}\n\n{code_to_show}", language='html')
                    else:
                        rendered = False
                        if body and body.get_text(strip=True):
                            st.components.v1.html(str(body), height=400, scrolling=True)
                            rendered = True
                        if not rendered:
                            visible = body.decode_contents() if body else soup.decode_contents()
                            if visible and "<mark>" in visible:
                                st.markdown(visible, unsafe_allow_html=True)
                            else:
                                st.info("No visible HTML body content to render. Displaying raw HTML with context below:")
                                code_to_show = (
                                    decoded if decoded and decoded.strip()
                                    else highlighted if highlighted and highlighted.strip()
                                    else "<empty response>"
                                )
                                context = format_header_context(status, content_type, response_headers)
                                st.code(f"{context}\n\n{code_to_show}", language='html')
            
                #zzzzzz
                elif mime_type is not None and mime_type.startswith("text/css"):
                    st.code(decoded if decoded and decoded.strip() else "<empty response>", language='css')
                
                elif mime_type is not None and mime_type.startswith("application/javascript"):
                    st.code(decoded if decoded and decoded.strip() else "<empty response>", language='javascript')
                
                else:
                    st.code(decoded if decoded and decoded.strip() else "<empty response>")
