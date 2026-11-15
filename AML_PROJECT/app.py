from pathlib import Path
import io
import json
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_transactions, detect_column_mapping
from src.graph_builder import build_graph
from src.pattern_detection import detect_patterns
from src.risk_scoring import calculate_risk
from src.explainability import generate_explanation
from src.ml_model import train_model
from src.behavioral_baseline import calculate_behavioral_baseline

st.set_page_config(page_title="TRACEFIN AI | AML Command Center", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

# ---------- Design system ----------
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--bg:#17100d;--panel:#211711;--panel2:#2a1d15;--line:#493326;--muted:#b59a83;--text:#f7ead8;--gold:#f5a900;--gold2:#ffc34d;--red:#df6b63;--green:#82b77a;--cream:#f3dfc2}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:radial-gradient(ellipse at 18% -10%,#3b2417 0%,#1d130f 42%,#120d0b 100%);color:var(--text);animation:appIn .65s ease both}
@keyframes appIn{from{opacity:.25;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
@keyframes rise{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)}}
@keyframes glow{0%,100%{box-shadow:0 0 0 rgba(245,169,0,0)}50%{box-shadow:0 0 24px rgba(245,169,0,.12)}}
section[data-testid="stSidebar"]{background:#1a110d;border-right:1px solid #493326}
.block-container{padding-top:1.1rem;padding-bottom:2.2rem;max-width:1760px;transition:max-width .25s ease}
.hero{padding:23px 27px;border:1px solid #684326;border-radius:15px;background:linear-gradient(110deg,rgba(52,31,18,.98),rgba(35,23,17,.94) 60%,rgba(75,44,17,.65));box-shadow:0 16px 45px #0005;margin-bottom:15px;position:relative;overflow:hidden;animation:rise .55s ease both}
.hero:after{content:'◈';position:absolute;right:34px;top:-43px;font-size:190px;color:#ffca63;opacity:.06;font-family:serif}
.eyebrow{color:#f4b52d;font-size:10px;letter-spacing:2px;font-weight:700;text-transform:uppercase}
.hero h1{font-family:'Space Grotesk';font-size:clamp(27px,3vw,39px);letter-spacing:-1.3px;margin:5px 0 3px;color:#fff2df}
.hero p{color:#c9ad91;margin:0;font-size:13px}
.section-head{font-family:'Space Grotesk';font-size:19px;font-weight:700;margin:16px 0 10px;color:#ffedda}
.kpi{background:linear-gradient(145deg,#302016,#211711);border:1px solid #614128;border-radius:12px;padding:15px 16px;min-height:104px;box-shadow:0 8px 22px #0003;animation:rise .55s ease both}
.kpi-label{font-size:10px;text-transform:uppercase;letter-spacing:1.1px;color:#c3a486;font-weight:700}
.kpi-value{font-family:'Space Grotesk';font-size:clamp(23px,2.2vw,31px);font-weight:700;color:#fff0dc;margin-top:7px}
.kpi-foot{font-size:10px;color:#b39a81;margin-top:1px}
.panel{background:linear-gradient(145deg,#2b1d15,#211711);border:1px solid #503827;border-radius:13px;padding:15px 17px}
.pill{display:inline-block;padding:4px 9px;border-radius:30px;font-size:10px;font-weight:700;letter-spacing:.7px;background:#4b321b;color:#ffcf75}
div[data-testid="stMetric"]{background:#2b1d15;border:1px solid #503827;border-radius:12px;padding:11px}
div[data-testid="stMetricLabel"]{color:#c3a486}
.stTabs [data-baseweb="tab-list"]{gap:6px;background:#1d130f;padding:6px;border-radius:11px;border:1px solid #493326}
.stTabs [data-baseweb="tab"]{border-radius:8px;color:#c5a98d;font-size:12px}
.stTabs [aria-selected="true"]{background:#553719;color:#fff1d9}
.stButton>button,.stDownloadButton>button{border-radius:9px;border:1px solid #9a6724;background:linear-gradient(110deg,#8b5917,#593918);color:#fff4df;font-weight:700;transition:transform .18s ease,filter .18s ease}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:#ffd16e;color:white;transform:translateY(-2px);filter:brightness(1.12)}
[data-testid="stDataFrame"]{border:1px solid #503827;border-radius:10px;overflow:hidden}
[data-testid="stPlotlyChart"]{animation:rise .6s ease both}
hr{border-color:#503827}
small,.muted{color:#b59a83}
.stTextInput input,.stSelectbox>div>div,.stMultiSelect>div>div,.stDateInput input,.stNumberInput input{background:#211711!important;color:#f7ead8!important;border-color:#614128!important}
[data-testid="stFileUploader"]{background:#211711;border:1px dashed #79532e;border-radius:10px;padding:8px}
@media(max-width:760px){.block-container{padding:1rem .7rem 2rem}.hero{padding:18px}.hero h1{font-size:27px}.section-head{font-size:17px}.kpi{min-height:92px;padding:12px}.stTabs [data-baseweb="tab"]{font-size:10px;padding:7px}}
@media(prefers-reduced-motion:reduce){*,*:before,*:after{animation:none!important;transition:none!important;scroll-behavior:auto!important}}
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar / data ----------
st.sidebar.markdown("<div class='eyebrow'>EXPLAINABLE AML TRIAGE</div><h2 style='font-family:Space Grotesk;margin:3px 0 0'>TRACEFIN <span style='color:#56d9e8'>AI</span></h2><p style='color:#91a3bd;font-size:12px'>AML Command Center · Prototype</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader("Import transaction data", type=["csv"], help="Upload a transaction CSV. TRACEFIN automatically detects common header names and lets you map unfamiliar columns.")
source = uploaded if uploaded is not None else str(Path(__file__).parent / "data/raw/transactions.csv")
column_mapping = None
if uploaded is not None:
    try:
        uploaded.seek(0)
        preview_df = pd.read_csv(uploaded, sep=None, engine="python", encoding="utf-8-sig", nrows=5)
        uploaded.seek(0)
        detected = detect_column_mapping(preview_df.columns)
        st.sidebar.caption(f"Detected {len(detected)} field mappings from {len(preview_df.columns)} CSV columns.")
        with st.sidebar.expander("Review / map CSV columns", expanded=len([x for x in ["timestamp", "sender_id", "receiver_id", "amount"] if x not in detected]) > 0):
            st.caption("Map your file's headers to TRACEFIN's standard transaction fields. This lets different bank/export formats use the same analysis engine.")
            options = ["— Not available —"] + list(preview_df.columns)
            column_mapping = {}
            chosen_sources = {}
            for canonical, title in [("transaction_id", "Transaction ID (optional)"), ("timestamp", "Date / timestamp"), ("sender_id", "Sender / source account"), ("receiver_id", "Receiver / destination account"), ("amount", "Transaction amount"), ("label", "Fraud label (optional)")]:
                default = detected.get(canonical)
                index = options.index(default) if default in options else 0
                chosen = st.selectbox(title, options, index=index, key=f"csvmap_{canonical}")
                if chosen != options[0]:
                    column_mapping[canonical] = chosen
                    chosen_sources.setdefault(chosen, []).append(canonical)
            duplicates = [source_name for source_name, targets in chosen_sources.items() if len(targets) > 1]
            if duplicates:
                st.error("Each CSV column can map to only one TRACEFIN field. Please correct the repeated selections: " + ", ".join(duplicates))
                st.stop()
    except Exception as exc:
        st.error(f"Could not read CSV headers: {exc}"); st.stop()
try:
    if uploaded is not None:
        uploaded.seek(0)
    raw = load_transactions(source, column_mapping=column_mapping)
except Exception as exc:
    st.error(f"Could not load transactions: {exc}")
    st.info("Minimum required information: transaction date/time, sender account, receiver account and amount. Transaction ID is generated automatically if absent; fraud labels are optional.")
    st.stop()
if raw.empty:
    st.warning("No valid transactions found in this dataset."); st.stop()
if uploaded is not None:
    st.sidebar.success(f"Loaded {len(raw):,} valid transactions")
    st.sidebar.caption(f"{raw.sender_id.nunique():,} senders · {raw.receiver_id.nunique():,} receivers · {raw.timestamp.min().date()} to {raw.timestamp.max().date()}")

st.sidebar.markdown("---")
dashboard_scale = st.sidebar.select_slider("Dashboard window scale", options=["Compact", "Standard", "Large"], value="Standard")
scale_rules = {"Compact": "0.92", "Standard": "1", "Large": "1.08"}
st.markdown(f"<style>.block-container{{zoom:{scale_rules[dashboard_scale]}}}</style>", unsafe_allow_html=True)
st.sidebar.markdown("**Investigation filters**")
min_date, max_date = raw.timestamp.min().date(), raw.timestamp.max().date()
date_range = st.sidebar.date_input("Transaction date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(date_range, tuple) and len(date_range)==2:
    filtered = raw[(raw.timestamp.dt.date >= date_range[0]) & (raw.timestamp.dt.date <= date_range[1])].copy()
else: filtered = raw.copy()
max_amount = float(filtered.amount.max()) if not filtered.empty else 0
amount_floor = st.sidebar.number_input("Minimum transaction amount (₹)", min_value=0.0, max_value=max_amount if max_amount>0 else 0.0, value=0.0, step=500.0)
filtered = filtered[filtered.amount >= amount_floor].copy()
st.sidebar.caption(f"{len(filtered):,} of {len(raw):,} transactions in view")
if filtered.empty:
    st.warning("No transactions match the selected filters. Adjust the filters in the sidebar."); st.stop()

# ---------- Analysis ----------
G = build_graph(filtered)
alerts = detect_patterns(G)
risks = calculate_risk(alerts)
risk_map = {r['account']:r['risk_score'] for r in risks}
level_map = {r['account']:r['risk_level'] for r in risks}
accounts = sorted(set(filtered.sender_id) | set(filtered.receiver_id))
flagged = len(risks)
high_count = sum(r['risk_level']=='HIGH' for r in risks)
medium_count = sum(r['risk_level']=='MEDIUM' for r in risks)
pattern_names = {"FAN_IN":"Fan-in","FAN_OUT":"Fan-out","CIRCULAR_ROUND_TRIPPING":"Circular flow","RAPID_PASS_THROUGH":"Rapid pass-through"}

# ---------- Header ----------
st.markdown("""
<div class="hero"><div class="eyebrow">AML · GRAPH ANALYTICS · EXPLAINABLE RISK</div>
<h1>TRACEFIN <span style="color:#56d9e8">AI</span></h1>
<p>See the connections. Surface the signals. Investigate with context.</p>
<div style="margin-top:16px"><span class="pill">LIVE ANALYSIS</span> <span style="color:#9db0c9;font-size:11px;margin-left:8px">Graph-based financial crime detection workspace</span></div></div>
""", unsafe_allow_html=True)

# Navigation
tabs = st.tabs(["◈  Command Center", "◎  Network Explorer", "⚑  Alert Investigation", "▤  Transactions & Reports", "⌁  Model Insights", "⛨  Two-Stage Verification"])

with tabs[0]:
    # Reference-inspired AML command centre: summary, risk funnel, case queue,
    # live stream, explainable case detail and model/data safeguards.
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">TRACEFIN · EXPLAINABLE AML TRIAGE</div>
      <h1>Command Center</h1>
      <p>Thousands of transactions in — a short, explainable list of cases out. Updated from the currently selected dataset.</p>
      <div style="margin-top:13px"><span class="pill">DEMO ENVIRONMENT</span>
      <span style="color:#c3a486;font-size:10px;margin-left:8px">Figures are generated from the uploaded CSV and are not customer or regulatory data.</span></div>
    </div>
    """, unsafe_allow_html=True)

    intro1,intro2,intro3=st.columns(3)
    with intro1:
        st.markdown("<div class='panel'><div class='eyebrow'>THE SHARED SHELL</div><b>One workspace, one investigation story</b><p style='color:#c3a486;font-size:12px;margin:6px 0 0'>A compact overview connects the transaction funnel, account risk, case queue, network explorer and model evaluation.</p></div>",unsafe_allow_html=True)
    with intro2:
        st.markdown("<div class='panel'><div class='eyebrow'>INVESTIGATION WORKSPACE</div><b>Prioritise with context</b><p style='color:#c3a486;font-size:12px;margin:6px 0 0'>Ranked account signals include detected typologies, supporting transactions and human-readable explanations.</p></div>",unsafe_allow_html=True)
    with intro3:
        st.markdown("<div class='panel'><div class='eyebrow'>MODEL & INTERACTION SAFEGUARDS</div><b>Explainable, reviewable outputs</b><p style='color:#c3a486;font-size:12px;margin:6px 0 0'>CSV mapping, validation, account-specific baselines and downloadable reports support analyst review.</p></div>",unsafe_allow_html=True)

    st.markdown("<div class='section-head'>Monitoring snapshot</div>",unsafe_allow_html=True)
    total_volume=float(filtered.amount.sum())
    total_alerts=len(alerts)
    kpis=[
      ("TRANSACTIONS SCORED",f"{len(filtered):,}","Valid records in current view","#f5b52d"),
      ("CASES SURFACED",f"{flagged:,}","Accounts with detected signals","#e6a15c"),
      ("DETECTION SIGNALS",f"{total_alerts:,}","Graph typology alerts","#d87568"),
      ("MEDIAN TRANSACTION",f"₹{filtered.amount.median():,.0f}","Across selected records","#e5c07b"),
      ("TOTAL VALUE",f"₹{total_volume:,.0f}","Aggregate transaction amount","#a8b879")]
    kcols=st.columns(5)
    for i,(c,(label,value,foot,color)) in enumerate(zip(kcols,kpis)):
        c.markdown(f"<div class='kpi' style='animation-delay:{i*70}ms'><div class='kpi-label'>{label}</div><div class='kpi-value' style='color:{color}'>{value}</div><div class='kpi-foot'>{foot}</div></div>",unsafe_allow_html=True)

    st.markdown("<div class='section-head'>Transaction-to-case funnel & risk distribution</div>",unsafe_allow_html=True)
    left,right=st.columns([1.15,1])
    with left:
        funnel=pd.DataFrame({"Stage":["Ingested rows","Valid transaction rows","Graph accounts","Accounts with signals"],"Records":[len(raw),len(filtered),len(accounts),flagged]})
        figf=px.bar(funnel,x="Records",y="Stage",orientation="h",text="Records",template="plotly_dark")
        figf.update_traces(marker_color=["#8d6a3c","#c58a31","#e0a33a","#e66c54"],texttemplate="%{text:,}",textposition="outside",cliponaxis=False)
        figf.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#e8d2b7",margin=dict(l=8,r=35,t=20,b=8),height=285,xaxis_title="",yaxis_title="",showlegend=False)
        st.plotly_chart(figf,use_container_width=True,config={"displayModeBar":False})
        st.caption("Funnel stages are counts from the current dataset and detection pipeline; they are not a calibrated conversion or model-performance metric.")
    with right:
        levels=pd.Series([r["risk_level"] for r in risks]).value_counts().reindex(["HIGH","MEDIUM","LOW"],fill_value=0)
        risk_df=pd.DataFrame({"Risk level":levels.index,"Accounts":levels.values})
        if int(risk_df.Accounts.sum())>0:
            figr=px.pie(risk_df,names="Risk level",values="Accounts",hole=.66,color="Risk level",color_discrete_map={"HIGH":"#d96e61","MEDIUM":"#e4a342","LOW":"#82a978"},template="plotly_dark")
            figr.update_traces(textinfo="percent",textfont_size=12,marker=dict(line=dict(color="#211711",width=3)))
            figr.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#e8d2b7",margin=dict(l=5,r=5,t=12,b=5),height=285,legend_title_text="")
            st.plotly_chart(figr,use_container_width=True,config={"displayModeBar":False})
            st.caption(f"{int(levels['HIGH'])} high · {int(levels['MEDIUM'])} medium · {int(levels['LOW'])} low risk accounts")
        else:
            st.info("No accounts were flagged in the current selection.")

    st.markdown("<div class='section-head'>Case queue & live detection stream</div>",unsafe_allow_html=True)
    qleft,qright=st.columns([1.2,1])
    with qleft:
        st.markdown("<div class='eyebrow'>CASE QUEUE — RANKED BY SCORE</div>",unsafe_allow_html=True)
        if risks:
            queue=pd.DataFrame(risks).sort_values("risk_score",ascending=False).head(8).copy()
            queue["patterns"]=queue.patterns.apply(lambda ps:", ".join(pattern_names.get(p,p) for p in ps))
            st.dataframe(queue.rename(columns={"account":"ACCOUNT","risk_score":"RISK SCORE","risk_level":"RISK BAND","patterns":"DETECTED PATTERNS"}),use_container_width=True,hide_index=True)
        else:
            st.success("No accounts are currently in the investigation queue.")
    with qright:
        st.markdown("<div class='eyebrow'>LIVE DETECTION STREAM</div>",unsafe_allow_html=True)
        latest=filtered.sort_values("timestamp",ascending=False).head(7)
        for _,row in latest.iterrows():
            score=max(risk_map.get(row.sender_id,0),risk_map.get(row.receiver_id,0))
            dot="#d96e61" if score>=50 else ("#e4a342" if score>=25 else "#82a978")
            stamp=row.timestamp.strftime("%H:%M:%S")
            st.markdown(f"<div style='display:flex;justify-content:space-between;gap:8px;border-bottom:1px solid #493326;padding:7px 0;font-size:11px'><span><b style='color:{dot}'>●</b> {row.transaction_id} · {row.sender_id} → {row.receiver_id}<br><span style='color:#b59a83'>₹{row.amount:,.0f} · max linked risk {score}/100</span></span><span style='color:#b59a83;white-space:nowrap'>{stamp}</span></div>",unsafe_allow_html=True)
        st.caption("Recent records from the selected CSV, annotated with the highest linked account risk score.")

    st.markdown("<div class='section-head'>Case detail & contextual risk</div>",unsafe_allow_html=True)
    detail_left,detail_right=st.columns([1,1])
    with detail_left:
        if risks:
            chosen_case=st.selectbox("Select account for case detail",[r["account"] for r in sorted(risks,key=lambda x:x["risk_score"],reverse=True)],key="command_case")
            case=next(r for r in risks if r["account"]==chosen_case)
            score=int(case["risk_score"])
            st.markdown(f"<div class='panel'><div class='eyebrow'>EXPLAINABLE SCORE BREAKDOWN</div><div style='display:flex;align-items:center;gap:18px;margin:12px 0'><div style='font-size:34px;font-weight:700;color:#f3c36c'>{score}<span style='font-size:13px;color:#b59a83'>/100</span></div><div><b>{chosen_case}</b><br><span style='color:#c3a486;font-size:12px'>{case['risk_level']} · {len(case['patterns'])} detected signal(s)</span></div></div><p style='color:#e9d4b9;font-size:13px'>{generate_explanation(case)}</p><div style='height:7px;border-radius:8px;background:#493326;overflow:hidden'><div style='height:7px;width:{score}%;background:linear-gradient(90deg,#d9a43c,#d96e61)'></div></div></div>",unsafe_allow_html=True)
            related=filtered[(filtered.sender_id==chosen_case)|(filtered.receiver_id==chosen_case)].sort_values("timestamp",ascending=False).head(6)
            st.dataframe(related[["transaction_id","timestamp","sender_id","receiver_id","amount"]],use_container_width=True,hide_index=True)
        else:
            st.info("A case detail will appear when the selected data produces an account-level signal.")
    with detail_right:
        st.markdown("<div class='eyebrow'>NETWORK EXPLORER — REPLAY PREVIEW</div>",unsafe_allow_html=True)
        if risks:
            node=chosen_case
            neighbors=set(G.successors(node))|set(G.predecessors(node)) if node in G else set()
            neighbors=list(neighbors)[:8]
            sub_nodes=[node]+neighbors
            sub=G.subgraph(sub_nodes)
            if sub.number_of_edges():
                pos2=nx.spring_layout(sub,seed=21)
                ex=[];ey=[]
                for u,v in sub.edges():
                    ex.extend([pos2[u][0],pos2[v][0],None]);ey.extend([pos2[u][1],pos2[v][1],None])
                netfig=go.Figure()
                netfig.add_trace(go.Scatter(x=ex,y=ey,mode="lines",line=dict(color="#a77b49",width=1.5,dash="dot"),hoverinfo="skip"))
                netfig.add_trace(go.Scatter(x=[pos2[n][0] for n in sub.nodes()],y=[pos2[n][1] for n in sub.nodes()],mode="markers+text",text=list(sub.nodes()),textposition="bottom center",marker=dict(size=[27 if n==node else 17 for n in sub.nodes()],color=["#e5a52e" if n==node else "#a77b49" for n in sub.nodes()],line=dict(color="#f4dfbf",width=1)),textfont=dict(color="#f4dfbf",size=10),hovertemplate="%{text}<extra></extra>"))
                netfig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",height=260,margin=dict(l=5,r=5,t=5,b=5),showlegend=False,xaxis=dict(visible=False),yaxis=dict(visible=False))
                st.plotly_chart(netfig,use_container_width=True,config={"displayModeBar":False})
                st.caption(f"{node} and {len(neighbors)} immediate counterparties shown. Open Network Explorer for the full graph.")
            else:
                st.info("No immediate network edges are available for this account.")
        else:
            st.info("Network replay preview becomes available when an account is selected.")

    st.markdown("<div class='section-head'>Model, data safeguards & accessibility</div>",unsafe_allow_html=True)
    m1,m2,m3=st.columns(3)
    with m1:
        st.markdown("<div class='panel'><div class='eyebrow'>MODEL & EVALUATION</div><b>Rule-based signals + optional ML</b><p style='color:#c3a486;font-size:12px'>Graph typologies and contextual baselines help prioritise review. Supervised evaluation is available when the CSV contains labels.</p></div>",unsafe_allow_html=True)
    with m2:
        st.markdown("<div class='panel'><div class='eyebrow'>DATA & CSV SAFEGUARDS</div><b>Universal header mapping</b><p style='color:#c3a486;font-size:12px'>Upload different CSV schemas, map source columns, validate required fields and export filtered evidence for review.</p></div>",unsafe_allow_html=True)
    with m3:
        st.markdown("<div class='panel'><div class='eyebrow'>INTERACTION & ACCESSIBILITY</div><b>Responsive, reduced-motion aware</b><p style='color:#c3a486;font-size:12px'>Responsive layout, animated transitions, hover feedback and reduced-motion preferences improve usability across screen sizes.</p></div>",unsafe_allow_html=True)
    st.caption("TRACEFIN is a decision-support prototype. A high score or typology is an investigative lead, not proof of money laundering. Validate customer type, expected business activity, transaction purpose and source of funds before escalation.")

with tabs[1]:
    st.markdown("<div class='section-head'>Financial relationship graph</div>",unsafe_allow_html=True)
    st.caption("Nodes represent accounts; directed edges represent transactions. Select an account to focus its immediate network.")
    focus=st.selectbox("Focus account",["All accounts"]+accounts,key='network_focus')
    max_edges=st.slider("Maximum edges to render",min_value=50,max_value=max(50,min(1500,len(filtered))),value=min(500,max(50,len(filtered))),step=50)
    view=G if focus=="All accounts" else G.subgraph(list(nx.ego_graph(G,focus,radius=1,undirected=True).nodes())).copy()
    edges=list(view.edges(keys=True,data=True))[:max_edges]
    pos=nx.spring_layout(view,seed=17,k=0.8,iterations=60)
    edge_x=[];edge_y=[]
    for u,v,k,d in edges:
        x0,y0=pos[u];x1,y1=pos[v];edge_x += [x0,x1,None];edge_y += [y0,y1,None]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=edge_x,y=edge_y,mode='lines',line=dict(width=1,color='rgba(103,145,185,.36)'),hoverinfo='none'))
    node_x=[];node_y=[];node_text=[];node_color=[];node_size=[]
    # Four graph-only risk bands. The existing risk-scoring engine and all
    # other dashboard classifications remain unchanged.
    graph_risk_bands = [
        (25, '#35c98a', 'LOW'),
        (50, '#f2d34f', 'MODERATE'),
        (75, '#ff9638', 'HIGH'),
        (101, '#ff5364', 'CRITICAL'),
    ]
    for n in view.nodes():
        x,y=pos[n];node_x.append(x);node_y.append(y);score=risk_map.get(n,0)
        graph_level = next(level for upper, color, level in graph_risk_bands if score < upper)
        graph_color = next(color for upper, color, level in graph_risk_bands if score < upper)
        node_color.append(graph_color)
        node_size.append(21 if n==focus else 14)
        node_patterns=[p for r in risks if r['account']==n for p in r.get('patterns',[])]
        node_text.append(f"Account: {n}<br>Risk score: {score}/100<br>Graph risk category: {graph_level}<br>Signals: {', '.join(node_patterns) if node_patterns else 'No rule-based signal'}")
    fig.add_trace(go.Scatter(x=node_x,y=node_y,mode='markers+text',text=list(view.nodes()),textposition='top center',textfont=dict(size=10,color='#dce8f8'),hovertext=node_text,hoverinfo='text',marker=dict(size=node_size,color=node_color,line=dict(width=1,color='#dce8f8'))))
    fig.update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',height=610,margin=dict(l=10,r=10,t=15,b=10),showlegend=False,xaxis=dict(visible=False),yaxis=dict(visible=False))
    st.plotly_chart(fig,use_container_width=True)
    st.markdown("<div style='display:flex;flex-wrap:wrap;gap:18px;font-size:13px;margin:4px 0 8px'><span style='color:#35c98a'>● Low (0–24)</span><span style='color:#f2d34f'>● Moderate (25–49)</span><span style='color:#ff9638'>● High (50–74)</span><span style='color:#ff5364'>● Critical (75–100)</span></div>",unsafe_allow_html=True)
    st.caption(f"Rendering {len(edges):,} of {view.number_of_edges():,} edges in the selected network. Multiple transactions between the same accounts are retained in the transaction table.")

with tabs[2]:
    st.markdown("<div class='section-head'>Investigation queue</div>",unsafe_allow_html=True)
    if risks:
        q=pd.DataFrame(risks).sort_values('risk_score',ascending=False)
        f1,f2=st.columns(2)
        level_filter=f1.multiselect("Risk level",options=['HIGH','MEDIUM','LOW'],default=['HIGH','MEDIUM','LOW'])
        pattern_filter=f2.multiselect("Pattern type",options=sorted({p for r in risks for p in r['patterns']}),format_func=lambda x:pattern_names.get(x,x))
        if pattern_filter:q=q[q.patterns.apply(lambda ps:any(p in ps for p in pattern_filter))]
        q=q[q.risk_level.isin(level_filter)]
        st.dataframe(q.assign(patterns=q.patterns.apply(lambda ps:', '.join(pattern_names.get(x,x) for x in ps))).rename(columns={'account':'Account','risk_score':'Risk score','risk_level':'Risk level','patterns':'Detected signals'}),use_container_width=True,hide_index=True)
        st.markdown("<div class='section-head'>Account-level explanation</div>",unsafe_allow_html=True)
        chosen=st.selectbox("Choose an account",q.account.tolist() if not q.empty else [r['account'] for r in risks])
        result=next(r for r in risks if r['account']==chosen)
        c1,c2,c3=st.columns(3);c1.metric('Risk score',f"{result['risk_score']}/100");c2.metric('Risk classification',result['risk_level']);c3.metric('Patterns',len(result['patterns']))
        st.markdown(f"<div class='panel'><div class='eyebrow'>WHY THIS ACCOUNT WAS FLAGGED</div><p style='font-size:15px;margin:10px 0;color:#eaf2ff'>{generate_explanation(result)}</p></div>",unsafe_allow_html=True)
        for p in result['patterns']:
            st.markdown(f"<div class='panel' style='margin-top:9px'><b>{pattern_names.get(p,p)}</b><p style='color:#9db0c9;margin:5px 0 0'>{ {'FAN_IN':'Multiple distinct accounts send funds to this account. This can be consistent with aggregation, but may also describe an ordinary merchant or collection account.','FAN_OUT':'This account sends funds to multiple distinct accounts. This can reflect payroll, supplier payments, or normal business distribution.','CIRCULAR_ROUND_TRIPPING':'A time-ordered directed cycle was identified in the transaction network. Review transaction purpose and beneficial ownership.','RAPID_PASS_THROUGH':'Funds appear to leave shortly after arriving, with an outgoing-to-incoming amount ratio inside the configured range.'}.get(p,'Review the underlying transactions.')}</p></div>",unsafe_allow_html=True)
        st.warning("Analyst review required: validate customer type, expected business activity, transaction purpose, source of funds, and historical behaviour before escalating.")
    else: st.success("Investigation queue is clear for the current filters.")

with tabs[3]:
    st.markdown("<div class='section-head'>Transaction ledger</div>",unsafe_allow_html=True)
    search=st.text_input("Search transaction ID or account",placeholder="e.g. T001 or A014")
    ledger=filtered.copy()
    if search:
        mask=ledger.astype(str).apply(lambda col:col.str.contains(search,case=False,na=False)).any(axis=1);ledger=ledger[mask]
    ledger['sender_risk']=ledger.sender_id.map(risk_map).fillna(0).astype(int)
    ledger['receiver_risk']=ledger.receiver_id.map(risk_map).fillna(0).astype(int)
    ledger['transaction_signal']=ledger[['sender_risk','receiver_risk']].max(axis=1)
    st.dataframe(ledger.sort_values('timestamp',ascending=False),use_container_width=True,hide_index=True)
    csv_bytes=ledger.to_csv(index=False).encode('utf-8')
    st.download_button("↓ Export filtered transaction ledger",csv_bytes,"tracefin_transactions.csv","text/csv")
    risk_export=pd.DataFrame(risks)
    if not risk_export.empty:
        risk_export['patterns']=risk_export.patterns.apply(lambda ps:', '.join(ps))
        st.download_button("↓ Export account risk report",risk_export.to_csv(index=False).encode(),"tracefin_account_risk.csv","text/csv")
    alert_export=pd.DataFrame(alerts)
    if not alert_export.empty:
        st.download_button("↓ Export detection alerts",alert_export.to_csv(index=False).encode(),"tracefin_alerts.csv","text/csv")

with tabs[4]:
    st.markdown("<div class='section-head'>Machine learning & model diagnostics</div>",unsafe_allow_html=True)
    st.caption("A Random Forest model is trained on account-level aggregates derived from the uploaded dataset labels. Results are exploratory and are not production-grade validation.")
    if 'label' not in filtered.columns:
        st.info("The uploaded file has no 'label' column. Rule-based graph analysis remains available; supervised model evaluation is skipped.")
    else:
        try:
            model,account_data,results,baseline=train_model(filtered)
            metric_cols=st.columns(4)
            for c,label,key in zip(metric_cols,['Accuracy','Precision','Recall','F1 score'],['accuracy','precision','recall','f1_score']):c.metric(label,f"{results[key]:.1%}")
            comparison=pd.DataFrame({'Metric':['Precision','Recall','F1 score'],'Random Forest':[results['precision'],results['recall'],results['f1_score']],'Simple baseline':[baseline['precision'],baseline['recall'],baseline['f1_score']]})
            fig=px.bar(comparison.melt('Metric',var_name='Approach',value_name='Score'),x='Metric',y='Score',color='Approach',barmode='group',range_y=[0,1],template='plotly_dark')
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font_color='#b9c9dd',height=350)
            st.plotly_chart(fig,use_container_width=True)
            st.markdown("**Account-level model scores**")
            st.dataframe(account_data.sort_values('suspicious_probability',ascending=False),use_container_width=True,hide_index=True)
            with st.expander("Classification report"):st.text(results['report'])
            st.warning("Important: the dataset is small and synthetic/demo-scale. A random train/test split can produce unstable metrics; labels aggregated across accounts may also create leakage-like effects. Do not present these metrics as real-world fraud-detection accuracy.")
        except Exception as exc:st.info(f"Model evaluation unavailable for this dataset: {exc}")


with tabs[5]:
    st.markdown("<div class='section-head'>Two-stage intelligent verification</div>", unsafe_allow_html=True)
    st.caption("Stage 1 finds transaction-network typologies. Stage 2 checks whether those signals are unusual for each account's own history and network context.")

    flow_cols = st.columns([1, .18, 1, .18, 1])
    with flow_cols[0]:
        st.markdown("<div class='panel' style='border-color:#39718d'><div class='eyebrow'>STAGE 1 · DETECT</div><h3>Typology detection</h3><p style='color:#a9bad1;font-size:13px'>Fan-in · Fan-out · Circular transfers · Rapid pass-through</p><b style='color:#56d9e8'>Network structure signals</b></div>", unsafe_allow_html=True)
    with flow_cols[1]:
        st.markdown("<div style='text-align:center;padding-top:35px;color:#56d9e8;font-size:24px'>→</div>", unsafe_allow_html=True)
    with flow_cols[2]:
        st.markdown("<div class='panel' style='border-color:#397c68'><div class='eyebrow'>STAGE 2 · VERIFY</div><h3>Contextual risk analysis</h3><p style='color:#a9bad1;font-size:13px'>Behavioral baseline · Historical activity · Counterparty context</p><b style='color:#5ce0b0'>Account-specific evidence</b></div>", unsafe_allow_html=True)
    with flow_cols[3]:
        st.markdown("<div style='text-align:center;padding-top:35px;color:#56d9e8;font-size:24px'>→</div>", unsafe_allow_html=True)
    with flow_cols[4]:
        st.markdown("<div class='panel' style='border-color:#80633b'><div class='eyebrow'>OUTPUT</div><h3>Explainable alert</h3><p style='color:#a9bad1;font-size:13px'>Risk indicators, evidence and analyst review</p><b style='color:#ffbd69'>Decision support</b></div>", unsafe_allow_html=True)

    stage1_count = len(alerts)
    stage1_accounts = len({a["account"] for a in alerts})
    st.markdown("<div class='section-head'>Stage 1 · Detected typologies</div>", unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    s1.metric("Pattern alerts", f"{stage1_count:,}")
    s2.metric("Accounts with signals", f"{stage1_accounts:,}")
    s3.metric("Pattern types", f"{len({a['pattern'] for a in alerts}):,}")
    if alerts:
        alert_table = pd.DataFrame(alerts)
        alert_table["pattern"] = alert_table["pattern"].map(lambda p: pattern_names.get(p, p))
        st.dataframe(alert_table, use_container_width=True, hide_index=True)
    else:
        st.info("No Stage 1 typologies were detected in the current filtered data.")

    st.markdown("<div class='section-head'>Stage 2 · Behavioral baseline detection</div>", unsafe_allow_html=True)
    st.markdown("<div class='panel'><b>What is being compared?</b><p style='color:#a9bad1;margin:6px 0 0'>For each account, TRACEFIN compares its latest observed day's transaction count, total value and number of unique counterparties against its earlier active-day history. This is account-specific: a busy merchant is compared with its own normal activity, not with a personal account.</p></div>", unsafe_allow_html=True)
    baseline = calculate_behavioral_baseline(filtered, min_history_days=3)
    if not baseline.empty:
        b1, b2, b3 = st.columns(3)
        b1.metric("Accounts evaluated", f"{len(baseline):,}")
        b2.metric("Unusual activity", f"{(baseline.baseline_status == 'UNUSUAL').sum():,}")
        b3.metric("Insufficient history", f"{(baseline.baseline_status == 'INSUFFICIENT_HISTORY').sum():,}")
        chosen_baseline = st.selectbox("Inspect account baseline", baseline.account.tolist(), key="baseline_account")
        selected_baseline = baseline[baseline.account == chosen_baseline].iloc[0]
        if selected_baseline["baseline_status"] == "INSUFFICIENT_HISTORY":
            st.warning(f"{chosen_baseline}: {selected_baseline['baseline_reason']}. Upload multiple days of historical transactions to enable a meaningful account-specific comparison.")
        else:
            bm1, bm2, bm3 = st.columns(3)
            bm1.metric("Latest-day transactions", int(selected_baseline["current_transactions"]), f"Historical avg: {selected_baseline['baseline_avg_transactions']:.2f}")
            bm2.metric("Latest-day amount", f"₹{selected_baseline['current_amount']:,.0f}", f"Historical avg: ₹{selected_baseline['baseline_avg_amount']:,.0f}")
            bm3.metric("Unique counterparties", int(selected_baseline["current_counterparties"]), f"Historical avg: {selected_baseline['baseline_avg_counterparties']:.2f}")
            if selected_baseline["baseline_status"] == "UNUSUAL":
                st.error(f"Baseline status: UNUSUAL · Prototype anomaly score: {int(selected_baseline['anomaly_score'])}/100")
            elif selected_baseline["baseline_status"] == "ELEVATED":
                st.warning(f"Baseline status: ELEVATED · Prototype anomaly score: {int(selected_baseline['anomaly_score'])}/100")
            else:
                st.success(f"Baseline status: CONSISTENT · Prototype anomaly score: {int(selected_baseline['anomaly_score'])}/100")
            st.caption(selected_baseline["baseline_reason"])

        display_baseline = baseline.copy()
        display_baseline["latest_day"] = display_baseline["latest_day"].dt.strftime("%Y-%m-%d")
        st.dataframe(display_baseline, use_container_width=True, hide_index=True)
        st.download_button("↓ Export behavioral baseline report",
            baseline.to_csv(index=False).encode("utf-8"),
            "tracefin_behavioral_baseline.csv", "text/csv")
    else:
        st.info("No account history is available for baseline analysis.")

    st.markdown("<div class='section-head'>How to interpret the verification</div>", unsafe_allow_html=True)
    st.markdown("- **Fan-in / fan-out is a lead, not a verdict.** A merchant, marketplace or payroll account may naturally show these patterns.\n- **Behavioral baseline is account-specific.** Unusual means materially above that account's earlier activity, not automatically criminal.\n- **History matters.** This prototype requires at least three prior active days before assigning a baseline status.\n- **Human review remains essential.** Verify customer type, purpose, source of funds and business context before escalation.")
    st.caption("Prototype methodology: latest active day compared with prior active-day mean and standard deviation across transaction count, total amount and unique counterparties. The displayed anomaly score is a heuristic triage score, not a calibrated probability.")

st.markdown("---")
st.markdown("<div style='text-align:center;color:#7187a3;font-size:11px;letter-spacing:1px'>TRACEFIN AI · EXPLAINABLE AML TRIAGE · HACKATHON PROTOTYPE</div>",unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:#7187a3;font-size:10px;margin-top:5px'>Decision support only · Alerts are investigative leads, not determinations of criminal activity</div>",unsafe_allow_html=True)
