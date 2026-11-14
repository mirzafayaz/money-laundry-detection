from pathlib import Path
import io
import json
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_transactions
from src.graph_builder import build_graph
from src.pattern_detection import detect_patterns
from src.risk_scoring import calculate_risk
from src.explainability import generate_explanation
from src.ml_model import train_model

st.set_page_config(page_title="TRACEFIN AI | Financial Intelligence", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

# ---------- Design system ----------
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--bg:#080d18;--panel:#101a2b;--panel2:#131f33;--line:#24344d;--muted:#91a3bd;--text:#edf4ff;--cyan:#56d9e8;--violet:#9a8cff;--red:#ff647c;--amber:#ffbd69;--green:#5ce0b0}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:radial-gradient(ellipse at 15% -10%,#192d50 0%,#0b1220 42%,#080d18 78%);color:var(--text)}
section[data-testid="stSidebar"]{background:#0b1423;border-right:1px solid var(--line)}
.block-container{padding-top:1.7rem;padding-bottom:2.8rem;max-width:1500px}
.hero{padding:24px 28px;border:1px solid #2a4160;border-radius:22px;background:linear-gradient(115deg,rgba(24,48,78,.94),rgba(17,27,46,.86) 55%,rgba(44,35,78,.55));box-shadow:0 18px 55px #0003;margin-bottom:20px;position:relative;overflow:hidden}
.hero:after{content:'◈';position:absolute;right:36px;top:-34px;font-size:190px;color:#8bdde9;opacity:.07;font-family:serif}
.eyebrow{color:var(--cyan);font-size:11px;letter-spacing:2.4px;font-weight:700;text-transform:uppercase}
.hero h1{font-family:'Space Grotesk';font-size:38px;letter-spacing:-1.5px;margin:5px 0 3px;color:#f4f8ff}
.hero p{color:#a9bad1;margin:0;font-size:14px}
.section-head{font-family:'Space Grotesk';font-size:20px;font-weight:700;margin:18px 0 12px;color:#eef5ff}
.kpi{background:linear-gradient(145deg,#14243a,#101a2b);border:1px solid #293d58;border-radius:16px;padding:17px 18px;min-height:112px;box-shadow:0 8px 24px #0002}
.kpi-label{font-size:11px;text-transform:uppercase;letter-spacing:1.2px;color:#9db0c9;font-weight:700}
.kpi-value{font-family:'Space Grotesk';font-size:30px;font-weight:700;color:#f2f7ff;margin-top:7px}
.kpi-foot{font-size:11px;color:#8fa3bd;margin-top:1px}
.panel{background:linear-gradient(145deg,#111e31,#0e1828);border:1px solid #263a55;border-radius:16px;padding:17px 19px}
.pill{display:inline-block;padding:4px 9px;border-radius:30px;font-size:10px;font-weight:700;letter-spacing:.7px;background:#20334b;color:#bfeaf0}
div[data-testid="stMetric"]{background:#111e31;border:1px solid #263a55;border-radius:14px;padding:12px}
div[data-testid="stMetricLabel"]{color:#9db0c9}
.stTabs [data-baseweb="tab-list"]{gap:8px;background:#0c1626;padding:7px;border-radius:13px}
.stTabs [data-baseweb="tab"]{border-radius:9px;color:#aebed2}
.stTabs [aria-selected="true"]{background:#1d3450;color:#fff}
.stButton>button,.stDownloadButton>button{border-radius:10px;border:1px solid #385575;background:linear-gradient(110deg,#1d4560,#34305d);color:white;font-weight:700}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:#65d8e7;color:white}
[data-testid="stDataFrame"]{border:1px solid #293d58;border-radius:12px;overflow:hidden}
hr{border-color:#26364d}
small,.muted{color:#91a3bd}
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar / data ----------
st.sidebar.markdown("<div class='eyebrow'>FINANCIAL INTELLIGENCE</div><h2 style='font-family:Space Grotesk;margin:3px 0 0'>TRACEFIN <span style='color:#56d9e8'>AI</span></h2><p style='color:#91a3bd;font-size:12px'>Investigation workspace · Prototype</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader("Import transaction data", type=["csv"], help="Required: transaction_id, timestamp, sender_id, receiver_id, amount. Optional: label.")
source = uploaded if uploaded is not None else str(Path(__file__).parent / "data/raw/transactions.csv")
try:
    raw = load_transactions(source)
except Exception as exc:
    st.error(f"Could not load transactions: {exc}"); st.stop()
if raw.empty:
    st.warning("No valid transactions found in this dataset."); st.stop()

st.sidebar.markdown("---")
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
tabs = st.tabs(["◈  Command Center", "◎  Network Explorer", "⚑  Alert Investigation", "▤  Transactions & Reports", "⌁  Model Insights"])

with tabs[0]:
    st.markdown("<div class='section-head'>Portfolio overview</div>", unsafe_allow_html=True)
    cols=st.columns(4)
    kpis=[("TRANSACTIONS",f"{len(filtered):,}","Filtered transaction records","#56d9e8"),("CONNECTED ACCOUNTS",f"{len(accounts):,}","Unique senders and receivers","#9a8cff"),("ACCOUNTS TO REVIEW",f"{flagged:,}","At least one detected pattern","#ffbd69"),("HIGH-RISK SIGNALS",f"{high_count:,}","Rule-based risk classification","#ff647c")]
    for c,(label,value,foot,color) in zip(cols,kpis):
        c.markdown(f"<div class='kpi'><div class='kpi-label'>{label}</div><div class='kpi-value' style='color:{color}'>{value}</div><div class='kpi-foot'>{foot}</div></div>",unsafe_allow_html=True)
    st.markdown("<div class='section-head'>Activity & risk signals</div>",unsafe_allow_html=True)
    left,right=st.columns([1.45,1])
    with left:
        daily=filtered.assign(day=filtered.timestamp.dt.floor('D')).groupby('day').agg(transactions=('transaction_id','count'),volume=('amount','sum')).reset_index()
        fig=px.area(daily,x='day',y='transactions',template='plotly_dark',title='Transaction activity over time')
        fig.update_traces(line_color='#56d9e8',fillcolor='rgba(86,217,232,.12)'); fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font_color='#b9c9dd',margin=dict(l=10,r=10,t=50,b=10),height=310)
        st.plotly_chart(fig,use_container_width=True)
    with right:
        counts=pd.Series([a['pattern'] for a in alerts]).value_counts().rename_axis('pattern').reset_index(name='alerts') if alerts else pd.DataFrame(columns=['pattern','alerts'])
        if not counts.empty:
            counts['pattern']=counts.pattern.map(lambda x:pattern_names.get(x,x))
            fig2=px.pie(counts,names='pattern',values='alerts',hole=.68,template='plotly_dark',title='Detection pattern mix')
            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)',font_color='#b9c9dd',legend_title='',margin=dict(l=5,r=5,t=50,b=5),height=310)
            st.plotly_chart(fig2,use_container_width=True)
        else: st.info("No patterns detected in the selected data.")
    st.markdown("<div class='section-head'>Priority accounts</div>",unsafe_allow_html=True)
    if risks:
        top=pd.DataFrame(risks).head(7).copy(); top['patterns']=top.patterns.apply(lambda p:', '.join(pattern_names.get(x,x) for x in p))
        st.dataframe(top.rename(columns={'account':'Account','risk_score':'Risk score','risk_level':'Risk level','patterns':'Signals'}),use_container_width=True,hide_index=True)
    else: st.success("No accounts were flagged by the current detection rules.")
    st.caption("Risk signals are triage indicators, not proof of money laundering or other financial crime. High transaction volume alone is not evidence of wrongdoing.")

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
    for n in view.nodes():
        x,y=pos[n];node_x.append(x);node_y.append(y);score=risk_map.get(n,0)
        node_color.append('#ff647c' if score>=70 else '#ffbd69' if score>=40 else '#56d9e8')
        node_size.append(21 if n==focus else 14)
        node_patterns=[p for r in risks if r['account']==n for p in r.get('patterns',[])]
        node_text.append(f"Account: {n}<br>Risk score: {score}/100<br>Signals: {', '.join(node_patterns)}")
    fig.add_trace(go.Scatter(x=node_x,y=node_y,mode='markers+text',text=list(view.nodes()),textposition='top center',textfont=dict(size=10,color='#dce8f8'),hovertext=node_text,hoverinfo='text',marker=dict(size=node_size,color=node_color,line=dict(width=1,color='#dce8f8'))))
    fig.update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',height=610,margin=dict(l=10,r=10,t=15,b=10),showlegend=False,xaxis=dict(visible=False),yaxis=dict(visible=False))
    st.plotly_chart(fig,use_container_width=True)
    st.markdown("<span style='color:#56d9e8'>●</span> Low / no rule-based signal &nbsp;&nbsp; <span style='color:#ffbd69'>●</span> Medium &nbsp;&nbsp; <span style='color:#ff647c'>●</span> High",unsafe_allow_html=True)
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

st.markdown("---")
st.markdown("<div style='text-align:center;color:#7187a3;font-size:11px;letter-spacing:1px'>TRACEFIN AI · GRAPH-BASED FINANCIAL INTELLIGENCE · HACKATHON PROTOTYPE</div>",unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:#7187a3;font-size:10px;margin-top:5px'>Decision support only · Alerts are investigative leads, not determinations of criminal activity</div>",unsafe_allow_html=True)
