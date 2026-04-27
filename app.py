
import streamlit as st
import pandas as pd
import plotly.express as px
from predictor import classify_investment, predict_price_5yr, get_investment_summary

st.set_page_config(page_title="Real Estate Investment Advisor", layout="wide", page_icon="🏠")

@st.cache_data
def load_data():
    return pd.read_csv("processed_data.csv")

df = load_data()

# ── Header
st.title("🏠 Real Estate Investment Advisor")
st.caption("Rule-based investment classifier & 5-year price forecast for Indian properties")
st.markdown("---")

# ── Sidebar Filters
st.sidebar.header("🔍 Filter Properties")
all_states = sorted(df["State"].dropna().unique())
state = st.sidebar.selectbox("State", all_states)

cities = sorted(df[df["State"] == state]["City"].dropna().unique())
city = st.sidebar.selectbox("City", cities)

bhk_options = sorted(df["BHK"].dropna().unique())
bhk = st.sidebar.multiselect("BHK", bhk_options, default=bhk_options)

price_min = float(df["Price_in_Lakhs"].min())
price_max = float(df["Price_in_Lakhs"].max())
price_range = st.sidebar.slider("Price Range (Lakhs)", price_min, price_max,
                                 (float(df["Price_in_Lakhs"].quantile(0.1)),
                                  float(df["Price_in_Lakhs"].quantile(0.9))))

prop_types = sorted(df["Property_Type"].dropna().unique())
selected_types = st.sidebar.multiselect("Property Type", prop_types, default=prop_types)

# ── Filter Data
filtered = df[
    (df["City"] == city) &
    (df["BHK"].isin(bhk)) &
    (df["Price_in_Lakhs"].between(*price_range)) &
    (df["Property_Type"].isin(selected_types))
]

# ── KPI Cards
st.subheader("📊 Market Overview")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Properties", len(filtered))
k2.metric("Avg Price (Lakhs)", f"₹{filtered['Price_in_Lakhs'].mean():.1f}")
k3.metric("Avg Price/SqFt", f"₹{filtered['Price_per_SqFt'].mean():.0f}")
k4.metric("Good Investments", f"{filtered['Good_Investment'].sum()} ({filtered['Good_Investment'].mean()*100:.1f}%)")
st.markdown("---")

# ── Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 EDA Insights", "🔮 Investment Predictor", "📋 Property Table", "📊 Location Analysis"])

# ───────────────────────────────────────────
with tab1:
    st.subheader("Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(filtered, x="Price_in_Lakhs", nbins=40,
                           title="Price Distribution", color_discrete_sequence=["#636EFA"])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(filtered, x="Property_Type", y="Price_per_SqFt",
                     title="Price/SqFt by Property Type", color="Property_Type")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.scatter(filtered, x="Size_in_SqFt", y="Price_in_Lakhs",
                         color=filtered["Good_Investment"].map({1: "Good", 0: "Not Good"}),
                         title="Size vs Price (Investment Quality)",
                         labels={"color": "Investment"}, opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        fig = px.box(filtered, x="Furnished_Status", y="Price_in_Lakhs",
                     title="Price by Furnished Status", color="Furnished_Status")
        st.plotly_chart(fig, use_container_width=True)

    col5, col6 = st.columns(2)
    with col5:
        fig = px.pie(filtered, names="Owner_Type", title="Properties by Owner Type", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    with col6:
        fig = px.pie(filtered, names="Availability_Status",
                     title="Availability Status", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    import plotly.figure_factory as ff
    num_cols = ["Price_in_Lakhs", "Size_in_SqFt", "Price_per_SqFt",
                "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals",
                "Public_Transport_Accessibility", "Amenity_Score", "Appreciation_Rate"]
    corr = filtered[num_cols].corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                    title="Correlation Heatmap")
    st.plotly_chart(fig, use_container_width=True)

# ───────────────────────────────────────────
with tab2:
    st.subheader("🔮 Enter Property Details to Get Prediction")

    col1, col2, col3 = st.columns(3)
    with col1:
        inp_price  = st.number_input("Current Price (Lakhs)", 5.0, 5000.0, 80.0)
        inp_size   = st.number_input("Size (SqFt)", 200, 10000, 1200)
        inp_bhk    = st.selectbox("BHK", [1, 2, 3, 4, 5])
        inp_ptype  = st.selectbox("Property Type", ["Apartment", "Villa", "House", "Plot"])
    with col2:
        inp_schools   = st.slider("Nearby Schools", 0, 20, 5)
        inp_hospitals = st.slider("Nearby Hospitals", 0, 15, 3)
        inp_transport = st.slider("Transport Accessibility (0-10)", 0, 10, 6)
    with col3:
        inp_amenities = st.slider("No. of Amenities", 0, 10, 3)
        inp_age       = st.number_input("Property Age (Years)", 0, 50, 5)
        inp_state     = st.selectbox("State", all_states)
        inp_furnished = st.selectbox("Furnished Status", ["Unfurnished", "Semi-Furnished", "Fully Furnished"])

    if st.button("🔍 Analyze Investment", type="primary", use_container_width=True):
        label, score = classify_investment(inp_schools, inp_hospitals, inp_transport, inp_amenities, inp_age)
        price_5yr, growth_rate = predict_price_5yr(inp_price, inp_state, inp_ptype)
        summary = get_investment_summary(score, label, inp_price, price_5yr, growth_rate)

        st.markdown("---")
        st.subheader("📋 Results")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Investment Decision", label)
        r2.metric("Investment Score", f"{score}")
        r3.metric("Predicted Price (5 Yrs)", f"₹{price_5yr} L", f"+{growth_rate}% CAGR")
        r4.metric("Expected Profit", f"₹{summary['expected_profit']} L", f"ROI: {summary['roi_percent']}%")

        # Score gauge chart
        fig = px.bar(x=["Your Score", "Threshold"],
                     y=[score, 0.85],
                     color=["Your Score", "Threshold"],
                     title="Investment Score vs Threshold",
                     labels={"x": "", "y": "Score"},
                     color_discrete_map={"Your Score": "#00CC96", "Threshold": "#EF553B"})
        st.plotly_chart(fig, use_container_width=True)

        # Price growth chart
        years = list(range(0, 6))
        rate = growth_rate / 100
        prices = [round(inp_price * ((1 + rate) ** y), 2) for y in years]
        fig2 = px.line(x=years, y=prices, markers=True,
                       title="Projected Price Growth Over 5 Years",
                       labels={"x": "Year", "y": "Price (Lakhs)"},
                       color_discrete_sequence=["#636EFA"])
        fig2.add_hline(y=inp_price, line_dash="dash", line_color="red",
                       annotation_text="Current Price")
        st.plotly_chart(fig2, use_container_width=True)

        st.info(f"💡 Score is based on schools, hospitals, transport, amenities & property age. Threshold = 0.85")

# ───────────────────────────────────────────
with tab3:
    st.subheader("📋 Filtered Property Data")
    st.write(f"Showing {len(filtered)} properties")
    show_cols = ["City", "Locality", "Property_Type", "BHK", "Size_in_SqFt",
                 "Price_in_Lakhs", "Price_per_SqFt", "Furnished_Status",
                 "Good_Investment", "Predicted_Price_5yr", "Appreciation_Rate"]
    st.dataframe(filtered[show_cols].reset_index(drop=True), use_container_width=True)

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Filtered Data", csv, "filtered_properties.csv", "text/csv")

# ───────────────────────────────────────────
with tab4:
    st.subheader("📊 Location-wise Analysis")

    col1, col2 = st.columns(2)
    with col1:
        state_inv = df.groupby("State")["Good_Investment"].mean().reset_index()
        state_inv.columns = ["State", "Good_Investment_Rate"]
        fig = px.bar(state_inv.sort_values("Good_Investment_Rate", ascending=False),
                     x="State", y="Good_Investment_Rate",
                     title="Good Investment Rate by State",
                     color="Good_Investment_Rate", color_continuous_scale="Greens")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        city_price = df.groupby("City")["Price_in_Lakhs"].mean().reset_index().sort_values("Price_in_Lakhs", ascending=False).head(15)
        fig = px.bar(city_price, x="City", y="Price_in_Lakhs",
                     title="Top 15 Cities by Avg Price",
                     color="Price_in_Lakhs", color_continuous_scale="Oranges")
        st.plotly_chart(fig, use_container_width=True)

    state_price = df.groupby("State")["Price_per_SqFt"].mean().reset_index()
    fig = px.choropleth(state_price, locations="State",
                        locationmode="geojson-id",
                        color="Price_per_SqFt",
                        title="Avg Price per SqFt by State",
                        color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)
