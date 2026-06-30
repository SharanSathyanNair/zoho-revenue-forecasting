import plotly.graph_objects as go
import streamlit as st


def metric_cards(
    current,
    forecast,
    growth,
    confidence,
):
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Current Revenue",
        f"₹{current:,.0f}",
    )

    c2.metric(
        "Forecast Revenue",
        f"₹{forecast:,.0f}",
    )

    c3.metric(
        "Growth",
        f"{growth:.2f}%"
    )

    c4.metric(
        "Confidence",
        confidence,
    )


def forecast_chart(
    history,
    forecast,
    lower,
    upper,
):
    fig = go.Figure()

    # Actual

    fig.add_trace(
        go.Scatter(
            x=history["week"],
            y=history["weekly_revenue"],
            mode="lines",
            name="Actual Revenue",
            line=dict(
                width=3,
                color="#00CC96",
            ),
        )
    )

    # Forecast

    fig.add_trace(
        go.Scatter(
            x=forecast["week"],
            y=forecast["weekly_revenue"],
            mode="lines",
            name="Forecast",
            line=dict(
                width=3,
                dash="dash",
                color="#EF553B",
            ),
        )
    )

    # Upper

    fig.add_trace(
        go.Scatter(
            x=forecast["week"],
            y=upper,
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Lower + Fill

    fig.add_trace(
        go.Scatter(
            x=forecast["week"],
            y=lower,
            fill="tonexty",
            fillcolor="rgba(239,85,59,0.20)",
            line=dict(width=0),
            name="Confidence Interval",
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=650,
        hovermode="x unified",
        xaxis_title="Week",
        yaxis_title="Revenue",
        legend=dict(
            orientation="h",
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )