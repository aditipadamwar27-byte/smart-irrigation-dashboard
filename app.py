import smtplib
from email.message import EmailMessage
import streamlit as st

import pandas as pd
import joblib
import plotly.express as px

def send_high_risk_email(row):
    try:
        email_config = st.secrets["email"]

        msg = EmailMessage()
        msg["Subject"] = "🚨 HIGH RISK: Irrigation Pump Alert"
        msg["From"] = email_config["sender"]
        msg["To"] = email_config["receiver"]

        timestamp = row.get("timestamp", "Not available")
        action = row.get(
            "recommended_action",
            "Inspect the pump and review sensor readings."
        )

        msg.set_content(
            f"""
SMART IRRIGATION PUMP ALERT

Risk Level: HIGH
Timestamp: {timestamp}

Maintenance Recommendation:
{action}

Please inspect the pump and verify its sensor readings.
"""
        )

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(
                email_config["sender"],
                email_config["app_password"]
            )
            server.send_message(msg)

        return True, "HIGH-risk email sent successfully."

    except Exception as e:
        return False, f"Email failed: {e}"
# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Irrigation Asset Monitoring",
    page_icon="💧",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 32px;
    font-weight: 700;
}

.subtitle {
    font-size: 17px;
    color: #666666;
}

div[data-testid="stMetric"] {
    background-color: #f5f7fa;
    padding: 15px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD FILES
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(
        "irrigation_pump_ai_model.pkl"
    )


@st.cache_data
def load_data():

    data = pd.read_csv(
        "irrigation_pump_dashboard_demo.csv"
    )

    data["timestamp"] = pd.to_datetime(
        data["timestamp"]
    )

    return data


@st.cache_data
def load_asset():

    return pd.read_csv(
        "asset_registry.csv"
    )


# ============================================================
# LOAD
# ============================================================

try:

    model_package = load_model()
    df = load_data()
    asset = load_asset()

except Exception as e:

    st.error("Unable to load project files.")
    st.code(str(e))
    st.stop()


# ============================================================
# MODEL PARAMETERS
# ============================================================

threshold_95 = model_package["threshold_95"]
threshold_99 = model_package["threshold_99"]


# ============================================================
# LATEST DATA
# ============================================================

latest = df.iloc[-1]

asset_id = latest["asset_id"]

health = float(
    latest["health_score"]
)

risk = str(
    latest["risk_level_final"]
)

anomaly = float(
    latest["anomaly_score_final"]
)

maintenance = str(
    latest["maintenance_status"]
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("💧 Smart Irrigation")

st.sidebar.caption(
    "Asset Monitoring & Predictive Maintenance"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Asset Management",
        "Sensor Monitoring",
        "Predictive Analytics",
        "Maintenance",
        "Failure Replay"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
AI Model
Isolation Forest

Purpose
Pump Condition Monitoring

Data
Historical Water Pump Sensors
"""
)



# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown(
        '<div class="main-title">'
        '💧 AI-Based Smart Irrigation Asset Monitoring'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-powered asset condition monitoring and predictive maintenance'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    # METRICS
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Asset ID", asset_id)

    with c2:
        st.metric("Health Score", f"{health:.1f}/100")

    with c3:
        st.metric("Risk Level", risk)

    with c4:
        st.metric("Anomaly Score", f"{anomaly:.4f}")

    st.markdown("---")

    # RISK ALERT
    if risk == "HIGH":
        st.error("🚨 HIGH RISK — Immediate maintenance recommended.")
    elif risk == "MEDIUM":
        st.warning("⚠️ MEDIUM RISK — Inspection required.")
    else:
        st.success("✅ LOW RISK — Asset operating normally.")

    # EMAIL ALERT
    st.subheader("📧 HIGH-Risk Email Alert")

    high_risk_rows = df[
        df["risk_level"].astype(str).str.upper() == "HIGH"
    ]

    if not high_risk_rows.empty:
        latest_high = high_risk_rows.iloc[-1]

        st.warning(
            f"High-risk pump event detected at "
            f"{latest_high.get('timestamp', 'Unknown time')}"
        )

        if st.button("Send HIGH-Risk Email"):
            success, message = send_high_risk_email(latest_high)

            if success:
                st.success(message)
            else:
                st.error(message)
    else:
        st.success("No HIGH-risk records found in the dataset.")

    st.markdown("---")

    # HEALTH TREND
    st.subheader("📈 AI Pump Health Trend")

    daily_health = (
        df.set_index("timestamp")
        .resample("D")["health_score"]
        .mean()
        .reset_index()
    )

    fig = px.line(
        daily_health,
        x="timestamp",
        y="health_score",
        markers=True,
        title="Daily Pump Health Score"
    )

    fig.update_yaxes(range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

    # LATEST SENSORS
    st.subheader("📊 Latest Sensor Values")

    sensor_map = {
        "Water Level": "water_level_Water Level (mm)",
        "Temperature": "temp_vibration.temp",
        "Pressure": "pressure_in_pressure.ps5",
        "Power P1": "power_power.p1",
        "Current I1": "current_power.i1",
        "X Vibration": "x_speed_X_speed",
        "Y Vibration": "y_speed_Y_speed",
        "Z Vibration": "z_speed_Z_speed"
    }

    sensor_table = []

    for name, column in sensor_map.items():
        if column in df.columns:
            sensor_table.append({
                "Sensor": name,
                "Latest Value": round(float(latest[column]), 3)
            })

    st.dataframe(
        pd.DataFrame(sensor_table),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ASSET MANAGEMENT
# ============================================================

elif page == "Asset Management":

    from datetime import date
    import pandas as pd
    import streamlit as st

    st.title("🔧 Asset Management")
    st.subheader("Installed Component Register")

    # KEEP YOUR EXISTING ASSET MANAGEMENT CODE BELOW THIS LINE

    # Existing asset registry information
    asset_info = asset.iloc[0]

    # Component tracking register.
    # Verify installation dates and condition before treating
    # template entries as confirmed physical installations.
    component_register = pd.DataFrame([
        {
            "Component ID": "COMP-001",
            "Component": "Water Pump",
            "Installation Date": "2023-07-01",
            "Condition": "Monitored",
            "Record Type": "Asset registry"
        },
        {
            "Component ID": "COMP-002",
            "Component": "Electric Motor",
            "Installation Date": "Not recorded",
            "Condition": "Not verified",
            "Record Type": "Tracking template"
        },
        {
            "Component ID": "COMP-003",
            "Component": "Power / Current Sensors",
            "Installation Date": "Not recorded",
            "Condition": "Not verified",
            "Record Type": "Tracking template"
        },
        {
            "Component ID": "COMP-004",
            "Component": "Temperature Sensor",
            "Installation Date": "Not recorded",
            "Condition": "Not verified",
            "Record Type": "Tracking template"
        },
        {
            "Component ID": "COMP-005",
            "Component": "Vibration Sensors",
            "Installation Date": "Not recorded",
            "Condition": "Not verified",
            "Record Type": "Tracking template"
        },
        {
            "Component ID": "COMP-006",
            "Component": "Water-Level Sensor",
            "Installation Date": "Not recorded",
            "Condition": "Not verified",
            "Record Type": "Tracking template"
        }
    ])

    st.dataframe(
        component_register,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "The pump record comes from the existing asset registry. "
        "Other component entries are tracking templates; verify "
        "installed hardware and dates before treating them as confirmed."
    )

    st.markdown("---")
    st.subheader("Asset Overview")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Asset Type",
            str(asset_info.get("asset_type", "Not recorded"))
        )

    with c2:
        st.metric(
            "Location",
            str(asset_info.get("location", "Not recorded"))
        )

    with c3:
        st.metric(
            "Asset Status",
            str(asset_info.get("status", "Not recorded"))
        )

    st.markdown("---")
    st.subheader("🛡️ Warranty Tracking")

    warranty_start = pd.to_datetime(
        asset_info.get("warranty_start"), errors="coerce"
    )
    warranty_end = pd.to_datetime(
        asset_info.get("warranty_end"), errors="coerce"
    )

    w1, w2, w3 = st.columns(3)

    with w1:
        st.metric(
            "Warranty Start",
            warranty_start.strftime("%d %b %Y")
            if pd.notna(warranty_start)
            else "Not recorded"
        )

    with w2:
        st.metric(
            "Warranty End",
            warranty_end.strftime("%d %b %Y")
            if pd.notna(warranty_end)
            else "Not recorded"
        )

    with w3:
        if pd.isna(warranty_end):
            st.warning("Warranty date unavailable")
        elif warranty_end.date() < date.today():
            st.error("Warranty Expired")
        else:
            st.success("Warranty Active")

    if pd.notna(warranty_end):
        if warranty_end.date() < date.today():
            st.info(
                "The recorded warranty has expired. Confirm the "
                "manufacturer's warranty terms before arranging "
                "paid maintenance."
            )
        else:
            st.info(
                "The recorded warranty end date has not passed. "
                "Confirm coverage with the manufacturer."
            )

    st.caption(
        "Warranty status is calculated from the recorded end date. "
        "It does not independently verify manufacturer coverage."
    )


# ============================================================
# SENSOR MONITORING
# ============================================================

elif page == "Sensor Monitoring":

    st.title("📊 Sensor Monitoring")

    sensor_options = {
        "Water Level":
            "water_level_Water Level (mm)",

        "Temperature":
            "temp_vibration.temp",

        "Pressure":
            "pressure_in_pressure.ps5",

        "Power":
            "power_power.p1",

        "Current":
            "current_power.i1",

        "X Vibration Speed":
            "x_speed_X_speed",

        "Y Vibration Speed":
            "y_speed_Y_speed",

        "Z Vibration Speed":
            "z_speed_Z_speed"
    }

    selected = st.selectbox(
        "Select Sensor",
        list(sensor_options.keys())
    )

    column = sensor_options[selected]

    sensor_data = df[
        ["timestamp", column]
    ].dropna()

    fig = px.line(
        sensor_data,
        x="timestamp",
        y=column,
        title=f"{selected} Trend"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "Sensor Statistics"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Minimum",
            f"{sensor_data[column].min():.3f}"
        )

    with c2:
        st.metric(
            "Maximum",
            f"{sensor_data[column].max():.3f}"
        )

    with c3:
        st.metric(
            "Mean",
            f"{sensor_data[column].mean():.3f}"
        )

    with c4:
        st.metric(
            "Latest",
            f"{sensor_data[column].iloc[-1]:.3f}"
        )


# ============================================================
# PREDICTIVE ANALYTICS
# ============================================================

elif page == "Predictive Analytics":

    st.title("🤖 Predictive Analytics")

    st.write(
        "Isolation Forest based anomaly detection "
        "for pump condition monitoring."
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Health",
            f"{health:.1f}/100"
        )

    with c2:

        st.metric(
            "Risk",
            risk
        )

    with c3:

        st.metric(
            "Anomaly",
            f"{anomaly:.4f}"
        )

    st.markdown("---")

    st.subheader(
        "AI Anomaly Score"
    )

    fig = px.line(
        df,
        x="timestamp",
        y="anomaly_score_final",
        title="Pump Anomaly Score"
    )

    fig.add_hline(
        y=threshold_95,
        line_dash="dash",
        annotation_text="95% Threshold"
    )

    fig.add_hline(
        y=threshold_99,
        line_dash="dash",
        annotation_text="99% Threshold"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "AI Maintenance Decision"
    )

    if risk == "HIGH":

        st.error(
            "Immediate Maintenance"
        )

    elif risk == "MEDIUM":

        st.warning(
            "Inspection Required"
        )

    else:

        st.success(
            "Normal Operation"
        )

    st.write(
        f"Current maintenance status: **{maintenance}**"
    )


# ============================================================
# MAINTENANCE
# ============================================================


# ============================================================
# MAINTENANCE
# ============================================================

elif page == "Maintenance":

    st.title("🔧 Maintenance Management")

    # Latest AI prediction
    
    # Determine the latest risk level
    
    # Determine the latest risk level
    if not df.empty:
        latest_df = df.copy()

        if "timestamp" in latest_df.columns:
            latest_df["timestamp"] = pd.to_datetime(
                latest_df["timestamp"], errors="coerce"
            )
            latest_df = latest_df.dropna(subset=["timestamp"])
            latest_df = latest_df.sort_values("timestamp")

        if not latest_df.empty:
            risk = str(
                latest_df.iloc[-1].get("risk_level_final", "LOW")
            ).strip().upper()
        else:
            risk = "LOW"
    else:
        risk = "LOW"

    # Temporary debugging information
    st.write("DEBUG — Total dataset rows:", len(df))
    st.write("DEBUG — Last CSV row:")
    st.dataframe(df.tail(1), use_container_width=True)

    st.write("DEBUG — Risk counts:")
    st.write(
        df["risk_level_final"]
        .fillna("LOW")
        .astype(str)
        .str.strip()
        .str.upper()
        .value_counts()
    )

    # Display the current risk banner
    if risk == "HIGH":
        st.error(
            "🚨 HIGH RISK — Immediate maintenance recommended."
        )
    elif risk == "MEDIUM":
        st.warning(
            "⚠️ MEDIUM RISK — Inspection required."
        )
    else:
        st.success(
            "✅ LOW RISK — No immediate maintenance indicated."
        )

    st.markdown("---")

    # Maintenance counts
    st.subheader("Maintenance Condition Distribution")

    maintenance_count = (
        df["maintenance_status"]
        .fillna("UNKNOWN")
        .value_counts()
        .rename_axis("Maintenance Status")
        .reset_index(name="Count")
    )

    fig = px.bar(
        maintenance_count,
        x="Maintenance Status",
        y="Count",
        title="Maintenance Status Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Map AI risk to a recommended action
   
    # AI risk to a recommended action
    def recommend_action(row):
        row_risk = str(
            row.get("risk_level_final", "LOW")
        ).strip().upper()

        if row_risk == "HIGH":
            return (
                "Inspect pump immediately; check motor, power/current, "
                "temperature and vibration before returning to service."
            )

        elif row_risk == "MEDIUM":
            return (
                "Schedule inspection and review recent sensor trends "
                "for developing abnormalities."
            )

        return (
            "Continue routine monitoring and scheduled preventive maintenance."
        )

   
    st.markdown("---")
    st.subheader("🚨 Alert History")

    # Prepare alert records
    alerts = df.copy()

    if "timestamp" in alerts.columns:
        alerts["timestamp"] = pd.to_datetime(
            alerts["timestamp"], errors="coerce"
        )

    if "risk_level_final" in alerts.columns:
        alerts["risk_level_final"] = (
            alerts["risk_level_final"]
            .fillna("LOW")
            .astype(str)
            .str.strip()
            .str.upper()
        )
    else:
        alerts["risk_level_final"] = "LOW"

    if "maintenance_status" in alerts.columns:
        alerts["maintenance_status"] = (
            alerts["maintenance_status"]
            .fillna("NORMAL")
            .astype(str)
            .str.strip()
            .str.upper()
        )
    else:
        alerts["maintenance_status"] = "NORMAL"

    alerts = alerts[
        alerts["risk_level_final"].isin(["HIGH", "MEDIUM"])
        | (alerts["maintenance_status"] != "NORMAL")
    ].copy()

    if not alerts.empty:
        alerts["Recommended Action"] = alerts.apply(
            recommend_action, axis=1
        )

        if "timestamp" in alerts.columns:
            alerts = alerts.sort_values(
                "timestamp", ascending=False
            )

        alerts = alerts.head(100).copy()

        # Session-only alert status tracking
        if "alert_statuses" not in st.session_state:
            st.session_state["alert_statuses"] = {}

        def get_alert_id(row):
            return (
                str(row.get("asset_id", "Unknown"))
                + "|"
                + str(row.get("timestamp", "Unknown"))
                + "|"
                + str(row.get("risk_level_final", "LOW"))
            )

        alerts["Alert ID"] = alerts.apply(get_alert_id, axis=1)

        for alert_id in alerts["Alert ID"]:
            if alert_id not in st.session_state["alert_statuses"]:
                st.session_state["alert_statuses"][alert_id] = "Open"

        alerts["Alert Status"] = alerts["Alert ID"].map(
            st.session_state["alert_statuses"]
        )

        high_count = (
            alerts["risk_level_final"] == "HIGH"
        ).sum()

        medium_count = (
            alerts["risk_level_final"] == "MEDIUM"
        ).sum()

        open_count = (alerts["Alert Status"] == "Open").sum()
        acknowledged_count = (
            alerts["Alert Status"] == "Acknowledged"
        ).sum()
        resolved_count = (
            alerts["Alert Status"] == "Resolved"
        ).sum()

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("HIGH", int(high_count))
        c2.metric("MEDIUM", int(medium_count))
        c3.metric("Open", int(open_count))
        c4.metric("Acknowledged", int(acknowledged_count))
        c5.metric("Resolved", int(resolved_count))

        st.dataframe(
            alerts[
                [
                    col for col in [
                        "timestamp",
                        "asset_id",
                        "health_score",
                        "risk_level_final",
                        "anomaly_score_final",
                        "maintenance_status",
                        "Recommended Action",
                        "Alert Status",
                    ]
                    if col in alerts.columns
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Update an alert")

        selected_alert = st.selectbox(
            "Select alert",
            options=alerts["Alert ID"].tolist(),
            format_func=lambda alert_id: (
                f"{alert_id} — "
                f"{st.session_state['alert_statuses'][alert_id]}"
            ),
        )

        new_status = st.selectbox(
            "New status",
            ["Open", "Acknowledged", "Resolved"],
            key="new_alert_status",
        )

        if st.button("Save Alert Status"):
            st.session_state["alert_statuses"][
                selected_alert
            ] = new_status
            st.success(f"Alert status updated to {new_status}.")
            st.rerun()

        st.download_button(
            "Download Alert History CSV",
            data=alerts.drop(
                columns=["Alert ID"], errors="ignore"
            ).to_csv(index=False).encode("utf-8"),
            file_name="smart_irrigation_alert_history.csv",
            mime="text/csv",
        )

    else:
        st.success("No HIGH- or MEDIUM-risk alerts found.")


    alerts = df.copy()

    # Safely normalize fields
    if "timestamp" in alerts.columns:
        alerts["timestamp"] = pd.to_datetime(
            alerts["timestamp"], errors="coerce"
        )

    if "risk_level_final" in alerts.columns:
        alerts["risk_level_final"] = (
            alerts["risk_level_final"]
            .fillna("LOW")
            .astype(str)
            .str.strip()
            .str.upper()
        )
    else:
        alerts["risk_level_final"] = "LOW"

    if "maintenance_status" in alerts.columns:
        alerts["maintenance_status"] = (
            alerts["maintenance_status"]
            .fillna("NORMAL")
            .astype(str)
            .str.strip()
            .str.upper()
        )
    else:
        alerts["maintenance_status"] = "NORMAL"

    # Include records requiring attention
    alerts = alerts[
        alerts["risk_level_final"].isin(["HIGH", "MEDIUM"])
        | (alerts["maintenance_status"] != "NORMAL")
    ].copy()

    if not alerts.empty:
        alerts["Recommended Action"] = alerts.apply(
            recommend_action, axis=1
        )

        display_columns = [
            "timestamp",
            "asset_id",
            "health_score",
            "risk_level_final",
            "anomaly_score_final",
            "maintenance_status",
            "Recommended Action",
        ]

        display_columns = [
            column for column in display_columns
            if column in alerts.columns
        ]

        
        # Sort alerts by timestamp: newest first
        if "timestamp" in alerts.columns:
            alerts = alerts.sort_values(
                "timestamp", ascending=False
            )

        alerts = alerts.head(100).copy()

        # Count risk levels among the displayed alerts
        high_count = (
            alerts["risk_level_final"] == "HIGH"
        ).sum()

        medium_count = (
            alerts["risk_level_final"] == "MEDIUM"
        ).sum()

        col1, col2 = st.columns(2)
        col1.metric("High-risk alerts shown", int(high_count))
        col2.metric("Medium-risk alerts shown", int(medium_count))

        if high_count:
            st.error(
                "🚨 HIGH RISK: HIGH-risk alerts require maintenance review."
            )

        if medium_count:
            st.warning(
                "⚠️ MEDIUM RISK: MEDIUM-risk alerts require inspection."
            )

        st.dataframe(
            alerts[display_columns],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download Maintenance Alerts CSV",
            data=alerts[display_columns].to_csv(
                index=False
            ).encode("utf-8"),
            file_name="smart_irrigation_maintenance_alerts.csv",
            mime="text/csv",
        )

    else:
        st.success(
            "No HIGH- or MEDIUM-risk maintenance alerts found."
        )

    # Total HIGH-risk records in the full dataset
    high_risk_count = (
        df["risk_level_final"]
        .fillna("LOW")
        .astype(str)
        .str.strip()
        .str.upper()
        .eq("HIGH")
        .sum()
    )

    st.caption(
        f"{high_risk_count} high-risk records exist in the full dataset. "
        "These are AI risk detections, not independently confirmed "
        "physical failures."
    )


# ============================================================
# FAILURE REPLAY
# ============================================================

elif page == "Failure Replay":

    st.title(
        "🔴 Historical Failure Replay"
    )

    st.info(
        "This section demonstrates the AI condition "
        "monitoring behavior around the documented "
        "March 13, 2024 pump failure."
    )

    replay = df[
        (df["timestamp"] >= pd.Timestamp("2024-03-10")) &
        (df["timestamp"] < pd.Timestamp("2024-03-14"))
    ].copy()

    if replay.empty:

        st.warning(
            "Failure replay data is not available."
        )

    else:

        # ----------------------------------------------------
        # HEALTH
        # ----------------------------------------------------

        st.subheader(
            "AI Health Score"
        )

        fig1 = px.line(
            replay,
            x="timestamp",
            y="health_score",
            title="March 10–13 Pump Health"
        )

        fig1.add_vline(
            x="2024-03-13",
            line_dash="dash",
            annotation_text="Documented Failure"
        )

        fig1.update_yaxes(
            range=[0, 100]
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

        # ----------------------------------------------------
        # ANOMALY
        # ----------------------------------------------------

        st.subheader(
            "AI Anomaly Score"
        )

        fig2 = px.line(
            replay,
            x="timestamp",
            y="anomaly_score_final",
            title="March 10–13 Anomaly Detection"
        )

        fig2.add_hline(
            y=threshold_95,
            line_dash="dash",
            annotation_text="95% Threshold"
        )

        fig2.add_hline(
            y=threshold_99,
            line_dash="dash",
            annotation_text="99% Threshold"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        # ----------------------------------------------------
        # DAILY SUMMARY
        # ----------------------------------------------------

        st.subheader(
            "Failure Period Summary"
        )

        summary = (
            replay
            .groupby(
                replay["timestamp"].dt.date
            )
            .agg(
                average_health=(
                    "health_score",
                    "mean"
                ),

                high_risk_minutes=(
                    "risk_level_final",
                    lambda x:
                    (x == "HIGH").sum()
                ),

                medium_risk_minutes=(
                    "risk_level_final",
                    lambda x:
                    (x == "MEDIUM").sum()
                )
            )
            .reset_index()
        )

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

        st.error(
            "Documented pump failure: 13 March 2024"
        )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.markdown("---")

st.sidebar.caption(
    "AI-Based Smart Irrigation Asset Monitoring "
    "and Predictive Maintenance System"
)