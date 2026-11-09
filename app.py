"""
EMA Signals - Subscriber Analytics Dashboard (Streamlit)
Run:  pip install streamlit pandas plotly
      streamlit run app.py
Data: data/clean_subscribers.csv (synthetic, produced by 01 and 02 scripts)
"""
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="EMA Signals Analytics", layout="wide")

@st.cache_data
def load():
    return pd.read_csv("data/clean_subscribers.csv", parse_dates=["signup_date"])

df_all = load()

# ---------------- Sidebar filters ----------------
st.sidebar.header("Filters")
ch = st.sidebar.multiselect("Acquisition channel", sorted(df_all.acquisition_channel.unique()),
                            default=sorted(df_all.acquisition_channel.unique()))
pl = st.sidebar.multiselect("Plan", ["Monthly", "Quarterly", "Annual"], default=["Monthly", "Quarterly", "Annual"])
ex = st.sidebar.multiselect("Experience", ["Beginner", "Intermediate", "Advanced"],
                            default=["Beginner", "Intermediate", "Advanced"])
tf = st.sidebar.multiselect("Timeframe", ["1m", "5m", "15m", "1H", "4H"], default=["1m", "5m", "15m", "1H", "4H"])
df = df_all[df_all.acquisition_channel.isin(ch) & df_all.plan.isin(pl)
            & df_all.experience_level.isin(ex) & df_all.preferred_timeframe.isin(tf)]
st.sidebar.caption("All data is synthetic, generated to validate the business strategy.")

st.title("EMA Signals: Subscriber Analytics")
st.caption("Descriptive and diagnostic analytics for a TradingView indicator subscription business")
if df.empty:
    st.warning("No subscribers match these filters."); st.stop()

# ---------------- KPIs ----------------
k = st.columns(6)
k[0].metric("Subscribers", f"{len(df):,}")
k[1].metric("Churn rate", f"{df.churn_flag.mean():.1%}")
k[2].metric("Total revenue", f"₹{df.total_revenue_inr.sum()/1e5:,.1f} L")
k[3].metric("Median LTV / CAC", f"{df.ltv_to_cac.median():.1f}x")
k[4].metric("Avg NPS score", f"{df.nps_score.mean():.1f} / 10")
k[5].metric("Profitable traders", f"{df.profitable_trader.mean():.0%}")

tab1, tab2, tab3, tab4 = st.tabs(["Descriptive", "Diagnostic: churn drivers", "Correlation", "Data"])

with tab1:
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(df.acquisition_channel.value_counts().reset_index(), x="acquisition_channel", y="count",
                           title="Subscribers by channel", labels={"count": "Subscribers", "acquisition_channel": ""}),
                    width="stretch")
    c2.plotly_chart(px.pie(df, names="plan", title="Plan mix", hole=.45), width="stretch")
    c3, c4 = st.columns(2)
    c3.plotly_chart(px.histogram(df, x="win_rate", nbins=30, title="Win rate experienced"), width="stretch")
    fig = px.histogram(df, x="net_pnl_inr", nbins=40, title="Net P&L per subscriber (₹)")
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    c4.plotly_chart(fig, width="stretch")
    m = df.groupby("signup_month").size().reset_index(name="signups")
    st.plotly_chart(px.line(m, x="signup_month", y="signups", markers=True, title="New subscribers per month"),
                    width="stretch")
    st.info("Most subscribers come from YouTube and Instagram, and over half choose the monthly plan. "
            "Win rates split into two groups: intraday users near 25% and 1H/4H users near 60%.")

with tab2:
    c1, c2 = st.columns(2)
    g = df.groupby("acquisition_channel").agg(churn=("churn_flag", "mean"), ltv_cac=("ltv_to_cac", "median")).reset_index()
    c1.plotly_chart(px.bar(g.sort_values("churn"), x="churn", y="acquisition_channel", orientation="h",
                           title="Churn rate by channel", labels={"churn": "Churn rate", "acquisition_channel": ""}),
                    width="stretch")
    c2.plotly_chart(px.bar(g.sort_values("ltv_cac"), x="ltv_cac", y="acquisition_channel", orientation="h",
                           title="Median LTV / CAC by channel", labels={"ltv_cac": "LTV / CAC", "acquisition_channel": ""}),
                    width="stretch")
    t = df.groupby("preferred_timeframe").agg(win_rate=("win_rate", "mean"), churn=("churn_flag", "mean")) \
          .reindex(["1m", "5m", "15m", "1H", "4H"]).dropna().reset_index()
    c3, c4 = st.columns(2)
    c3.plotly_chart(px.bar(t, x="preferred_timeframe", y=["win_rate", "churn"], barmode="group",
                           title="Timeframe: win rate vs churn", labels={"value": "Rate", "preferred_timeframe": ""}),
                    width="stretch")
    c4.plotly_chart(px.box(df, x="churned", y="monthly_logins", title="Monthly logins: churned vs retained"),
                    width="stretch")
    c5, c6 = st.columns(2)
    o = df.groupby("onboarding_completed").churn_flag.mean().reset_index()
    c5.plotly_chart(px.bar(o, x="onboarding_completed", y="churn_flag", title="Churn by onboarding completed",
                           labels={"churn_flag": "Churn rate"}), width="stretch")
    p = df.groupby("plan").churn_flag.mean().reindex(["Monthly", "Quarterly", "Annual"]).dropna().reset_index()
    c6.plotly_chart(px.bar(p, x="plan", y="churn_flag", title="Churn by plan", labels={"churn_flag": "Churn rate"}),
                    width="stretch")
    st.plotly_chart(px.scatter(df, x="win_rate", y="nps_score", opacity=.35, color="timeframe_group",
                               title="Win rate vs NPS score"), width="stretch")
    st.success("Diagnosis: churn is driven by poor outcomes (intraday timeframes), no onboarding, low engagement, "
               "monthly plans and support issues. Referral is the cheapest, stickiest channel.")

with tab3:
    cols = ["trading_capital_inr", "cac_inr", "onboarding_flag", "monthly_logins", "signals_followed", "win_rate",
            "net_pnl_inr", "support_tickets", "nps_score", "tenure_months", "total_revenue_inr", "churn_flag"]
    corr = df[cols].corr().round(2)
    st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                              title="Correlation matrix", aspect="auto", height=650), width="stretch")
    st.write("Correlation of each variable with churn:")
    st.bar_chart(corr["churn_flag"].drop("churn_flag").sort_values())
    st.caption("Tenure and total revenue are consequences of churn, not causes, so they are read as outcomes.")

with tab4:
    st.dataframe(df, width="stretch", height=450)
    st.download_button("Download filtered data (CSV)", df.to_csv(index=False), "filtered_subscribers.csv")
