import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import io

# ----------------------------------------------------------------------
# 1. CORE SIMULATION ENGINE
# This is the heart of the model, adapted from our previous script.
# ----------------------------------------------------------------------

def run_simulation(
    N_nuclear, 
    Capacity_Wind_kW, 
    Capacity_Solar_kWp, 
    capacity_storage_kwh, 
    data_frame,
    params
    ):
    """
    Simulates the power system for 8760 hours.
    
    Inputs (decision variables):
    N_nuclear (int): Number of Nuclear SMR units
    Capacity_Wind_kW (float): Total installed Wind capacity (kW)
    Capacity_Solar_kWp (float): Total installed Solar capacity (kWp)
    capacity_storage_kwh (float): Total energy capacity of storage (kWh)
    data_frame (pd.DataFrame): The pre-processed hourly data
    params (dict): Dictionary of all system parameters
    
    Returns:
    tuple: (Total Gas Generation kWh, Total Curtailed Energy kWh, Total Investment Cost, SOC_timeseries, Gas_timeseries)
    """
    
    # 1. Calculate Investment Cost
    cost = (
        N_nuclear * params['COST_NUCLEAR_UNIT'] +
        Capacity_Wind_kW * params['COST_WIND_KW'] +
        Capacity_Solar_kWp * params['COST_SOLAR_KW'] +
        capacity_storage_kwh * params['COST_STORAGE_KWH']
    )
    
    # If the cost exceeds the budget, return a massive penalty
    if cost > params['BUDGET']:
        return (1e18, 0.0, cost, None, None) # Penalty: 10^18 kWh Gas Gen
    
    # 2. Calculate Total Generation Capacity (kW)
    P_nuclear_max = N_nuclear * params['SMR_RATED_POWER_KW']
    P_storage_max_dispatch = capacity_storage_kwh / params['STORAGE_DISPATCH_HOURS']
    
    # 3. Time Series Calculation
    Gen_Baseload_kWh = P_nuclear_max + params['HYDRO_CAPACITY_KW'] 
    Gen_Solar_kWh = data_frame['Solar_kW_per_kWp'] * Capacity_Solar_kWp
    Gen_Wind_kWh = data_frame['Wind_kW_per_kW'] * Capacity_Wind_kW
    
    Demand_kWh_ts = data_frame['Demand_kWh']
    
    SOC = np.zeros(len(data_frame))
    SOC[0] = capacity_storage_kwh * 0.5 # Start at 50%
    
    Gas_Generation_kWh = np.zeros(len(data_frame))
    Curtailed_Energy_kWh = np.zeros(len(data_frame))
    
    current_soc = SOC[0]
    
    for t in range(len(data_frame)):
        Demand_t = Demand_kWh_ts.iloc[t]
        Gen_Total_t = Gen_Baseload_kWh + Gen_Solar_kWh.iloc[t] + Gen_Wind_kWh.iloc[t]
        Balance_t = Gen_Total_t - Demand_t
        
        if Balance_t >= 0: # Surplus: Store or Curtail
            Surplus_t = Balance_t
            Max_Charge_t = capacity_storage_kwh - current_soc
            Storable_Energy_t = Surplus_t * params['STORAGE_EFFICIENCY']
            Energy_Charged_t = min(Storable_Energy_t, Max_Charge_t)
            
            current_soc += Energy_Charged_t
            Curtailed_Energy_kWh[t] = Surplus_t - (Energy_Charged_t / params['STORAGE_EFFICIENCY'])
            Gas_Generation_kWh[t] = 0
            
        else: # Deficit: Discharge Storage, then use Gas
            Deficit_t = abs(Balance_t)
            Max_Discharge_t = min(P_storage_max_dispatch, current_soc)
            
            # Energy to pull from battery (assumes 100% discharge efficiency)
            Energy_Discharged_t = min(Max_Discharge_t, Deficit_t) 
            
            current_soc -= Energy_Discharged_t
            Remaining_Deficit = Deficit_t - Energy_Discharged_t
            Gas_Generation_kWh[t] = Remaining_Deficit
            Curtailed_Energy_kWh[t] = 0
            
        SOC[t] = current_soc
        
    total_gas = np.sum(Gas_Generation_kWh)
    total_curtailed = np.sum(Curtailed_Energy_kWh)
    
    return total_gas, total_curtailed, cost, SOC, Gas_Generation_kWh

# ----------------------------------------------------------------------
# 2. OPTIMIZATION FUNCTION
# ----------------------------------------------------------------------

@st.cache_data # Cache the main computation
def find_optimal_mix(_data_df, params):
    """
    Runs a grid search to find the optimal mix.
    """
    best_gas_gen = 1e20
    best_config = None
    best_results = None
    
    # Create the grid for budget splits
    split_options = np.linspace(0.0, 1.0, params['GRID_SEARCH_DENSITY'])
    
    with st.spinner(f"Running optimization... Testing {len(params['N_NUC_OPTIONS']) * len(split_options)**2} combinations..."):
        for N_nuc in params['N_NUC_OPTIONS']:
            cost_nuc = N_nuc * params['COST_NUCLEAR_UNIT']
            remaining_budget = params['BUDGET'] - cost_nuc
            
            if remaining_budget < 0:
                continue

            for solar_share in split_options:
                for storage_share in split_options:
                    wind_share = 1.0 - solar_share - storage_share
                    
                    if wind_share < 0:
                        continue # This combination is not valid
                    
                    # Calculate capacities based on budget shares
                    cap_solar_kw = (remaining_budget * solar_share) / params['COST_SOLAR_KW']
                    cap_storage_kwh = (remaining_budget * storage_share) / params['COST_STORAGE_KWH']
                    cap_wind_kw = (remaining_budget * wind_share) / params['COST_WIND_KW']
                    
                    # Run simulation
                    total_gas, total_curtailed, final_cost, SOC, Gas_Gen = run_simulation(
                        N_nuc, cap_wind_kw, cap_solar_kw, cap_storage_kwh, _data_df, params
                    )
                    
                    # Update best configuration
                    if total_gas < best_gas_gen:
                        best_gas_gen = total_gas
                        best_config = {
                            'N_nuclear': N_nuc,
                            'Capacity_Wind_kW': cap_wind_kw,
                            'Capacity_Solar_kWp': cap_solar_kw,
                            'Storage_Capacity_kWh': cap_storage_kwh
                        }
                        best_results = {
                            'Total_Gas_kWh': total_gas,
                            'Total_Curtailed_kWh': total_curtailed,
                            'Final_Cost': final_cost,
                            'SOC': SOC,
                            'Gas_Generation_kWh_TS': Gas_Gen
                        }

    return best_config, best_results

# ----------------------------------------------------------------------
# 3. HELPER FUNCTIONS (PLOTTING, DATA LOADING)
# ----------------------------------------------------------------------

@st.cache_data
def load_data(demand_file, solar_file, wind_file):
    """
    Loads and merges the three uploaded CSV files.
    """
    try:
        # Read demand
        demand_df = pd.read_csv(demand_file)
        demand_kwh = demand_df['Mwh'].astype(float) * 1000 # Convert MWh to kWh

        # Read solar (skip headers from Renewables.ninja)
        solar_df = pd.read_csv(solar_file, skiprows=3)
        solar_kw_per_kwp = solar_df['electricity [kWh]'].astype(float)

        # Read wind (skip headers from Renewables.ninja)
        wind_df = pd.read_csv(wind_file, skiprows=3)
        wind_kw_per_kw = wind_df['electricity [kWh]'].astype(float)

        # Combine
        max_len = min(len(demand_kwh), len(solar_kw_per_kwp), len(wind_kw_per_kw))
        if max_len < 8760:
            st.warning(f"Data truncated to {max_len} hours based on shortest file.")
        
        data_df = pd.DataFrame({
            'Demand_kWh': demand_kwh.head(max_len),
            'Solar_kW_per_kWp': solar_kw_per_kwp.head(max_len),
            'Wind_kW_per_kW': wind_kw_per_kw.head(max_len)
        })
        return data_df
    
    except Exception as e:
        st.error(f"Error loading files: {e}")
        st.error("Please ensure files are correct CSVs and match the expected format.")
        return None

def plot_pie_chart(data, title):
    """
    Generates a Matplotlib pie chart.
    """
    labels = data.keys()
    sizes = data.values()
    
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.set_title(title)
    return fig

def get_hourly_timeseries(config, _data_df, params):
    """
    Re-creates the full hourly timeseries for the optimal mix for plotting.
    """
    # Get generation data
    Gen_Hydro_kWh = pd.Series(np.repeat(params['HYDRO_CAPACITY_KW'], len(_data_df)))
    Gen_Nuclear_kWh = pd.Series(np.repeat(config['N_nuclear'] * params['SMR_RATED_POWER_KW'], len(_data_df)))
    Gen_Solar_kWh = _data_df['Solar_kW_per_kWp'] * config['Capacity_Solar_kWp']
    Gen_Wind_kWh = _data_df['Wind_kW_per_kW'] * config['Capacity_Wind_kW']
    
    Demand_kWh_ts = _data_df['Demand_kWh']
    
    # Re-run simulation logic to get storage flows
    P_storage_max_dispatch = config['Storage_Capacity_kWh'] / params['STORAGE_DISPATCH_HOURS']
    
    SOC = np.zeros(len(_data_df))
    SOC[0] = config['Storage_Capacity_kWh'] * 0.5
    current_soc = SOC[0]
    
    Gas_kWh = np.zeros(len(_data_df))
    Curtailed_kWh = np.zeros(len(_data_df))
    Storage_Charge_kWh = np.zeros(len(_data_df))
    Storage_Discharge_kWh = np.zeros(len(_data_df))
    
    for t in range(len(_data_df)):
        Demand_t = Demand_kWh_ts.iloc[t]
        Gen_Total_t = Gen_Hydro_kWh.iloc[t] + Gen_Nuclear_kWh.iloc[t] + Gen_Solar_kWh.iloc[t] + Gen_Wind_kWh.iloc[t]
        Balance_t = Gen_Total_t - Demand_t
        
        if Balance_t >= 0: # Surplus
            Surplus_t = Balance_t
            Max_Charge_t = config['Storage_Capacity_kWh'] - current_soc
            Storable_Energy_t = Surplus_t * params['STORAGE_EFFICIENCY']
            Energy_Charged_t = min(Storable_Energy_t, Max_Charge_t)
            
            current_soc += Energy_Charged_t
            Storage_Charge_kWh[t] = Energy_Charged_t
            Curtailed_kWh[t] = Surplus_t - (Energy_Charged_t / params['STORAGE_EFFICIENCY'])
            
        else: # Deficit
            Deficit_t = abs(Balance_t)
            Max_Discharge_t = min(P_storage_max_dispatch, current_soc)
            Energy_Discharged_t = min(Max_Discharge_t, Deficit_t)
            
            current_soc -= Energy_Discharged_t
            Storage_Discharge_kWh[t] = Energy_Discharged_t
            Gas_kWh[t] = Deficit_t - Energy_Discharged_t
            
        SOC[t] = current_soc
        
    # Create final DataFrame
    hourly_df = pd.DataFrame({
        'Demand': Demand_kWh_ts,
        'Hydro': Gen_Hydro_kWh,
        'Nuclear': Gen_Nuclear_kWh,
        'Solar': Gen_Solar_kWh,
        'Wind': Gen_Wind_kWh,
        'Gas': Gas_kWh,
        'Storage_Discharge': Storage_Discharge_kWh,
        'Storage_Charge': Storage_Charge_kWh,
        'Curtailed': Curtailed_kWh,
        'SOC': SOC
    })
    
    return hourly_df

# ----------------------------------------------------------------------
# 4. STREAMLIT APP LAYOUT
# ----------------------------------------------------------------------

st.set_page_config(layout="wide")
st.title("⚡ Power System Optimization Tool")

# --- SIDEBAR ---
st.sidebar.header("Configuration")

# File Uploaders
st.sidebar.subheader("1. Upload Data Files")
demand_file = st.sidebar.file_uploader("Upload Demand CSV", type="csv")
solar_file = st.sidebar.file_uploader("Upload Solar CSV (Renewables.ninja)", type="csv")
wind_file = st.sidebar.file_uploader("Upload Wind CSV (Renewables.ninja)", type="csv")

# Gather parameters from user input
params = {}

st.sidebar.subheader("2. Financial Parameters")
params['BUDGET'] = st.sidebar.number_input("Total Investment Budget (€)", value=8_920_200_000, format="%d")

st.sidebar.subheader("3. Technology Costs (€)")
params['COST_NUCLEAR_UNIT'] = st.sidebar.number_input("Nuclear Cost (€/unit)", value=250_000_000, format="%d")
params['COST_WIND_KW'] = st.sidebar.number_input("Wind Cost (€/kW)", value=1300.0)
params['COST_SOLAR_KW'] = st.sidebar.number_input("Solar Cost (€/kWp)", value=1090.91)
params['COST_STORAGE_KWH'] = st.sidebar.number_input("Storage Cost (€/kWh)", value=100.0)

st.sidebar.subheader("4. Technology Specifications")
params['HYDRO_CAPACITY_KW'] = st.sidebar.number_input("Hydro Capacity (kW)", value=258_996, format="%d")
params['SMR_RATED_POWER_KW'] = st.sidebar.number_input("SMR Rated Power (kW/unit)", value=50_000, format="%d")
params['STORAGE_EFFICIENCY'] = st.sidebar.slider("Storage Roundtrip Efficiency", 0.0, 1.0, 0.85)
params['STORAGE_DISPATCH_HOURS'] = st.sidebar.number_input("Storage Dispatch Time (h)", value=6)

st.sidebar.subheader("5. Optimization Settings")
params['N_NUC_OPTIONS'] = st.sidebar.multiselect("Nuclear Units to Test", [0, 1, 2, 3], default=[0, 1])
params['GRID_SEARCH_DENSITY'] = st.sidebar.slider("Budget Split Density (N-steps)", 3, 11, 5, help="Number of steps for each budget share (e.g., 5 steps = [0, 0.25, 0.5, 0.75, 1.0])")

# --- MAIN PANEL ---
if not (demand_file and solar_file and wind_file):
    st.info("Welcome! Please upload Demand, Solar, and Wind data files in the sidebar to begin.")
else:
    # Load data
    data_df = load_data(demand_file, solar_file, wind_file)
    
    if data_df is not None:
        st.success(f"Successfully loaded {len(data_df)} hours of data.")
        
        # Run Button
        if st.sidebar.button("🚀 Run Optimization"):
            
            # 1. Run Optimization
            best_config, best_results = find_optimal_mix(data_df, params)
            
            if best_config is None:
                st.error("Optimization failed. No valid configuration found within the budget.")
            else:
                st.header("✨ Optimal Power Mix Found ✨")
                
                # 2. Calculate final metrics
                
                # Investment Breakdown
                cost_nuc = best_config['N_nuclear'] * params['COST_NUCLEAR_UNIT']
                cost_wind = best_config['Capacity_Wind_kW'] * params['COST_WIND_KW']
                cost_solar = best_config['Capacity_Solar_kWp'] * params['COST_SOLAR_KW']
                cost_storage = best_config['Storage_Capacity_kWh'] * params['COST_STORAGE_KWH']
                
                investment_breakdown = {
                    'Nuclear': cost_nuc,
                    'Wind': cost_wind,
                    'Solar': cost_solar,
                    'Storage': cost_storage,
                }
                
                # Generation Breakdown
                hourly_data = get_hourly_timeseries(best_config, data_df, params)
                
                annual_gen_kwh = {
                    'Hydro': hourly_data['Hydro'].sum(),
                    'Nuclear': hourly_data['Nuclear'].sum(),
                    'Solar': hourly_data['Solar'].sum(),
                    'Wind': hourly_data['Wind'].sum(),
                    'Gas': hourly_data['Gas'].sum(),
                }
                
                # Other Metrics
                total_demand = hourly_data['Demand'].sum()
                total_curtailed = hourly_data['Curtailed'].sum()
                total_potential_gen = annual_gen_kwh['Hydro'] + annual_gen_kwh['Nuclear'] + annual_gen_kwh['Solar'] + annual_gen_kwh['Wind']
                spared_percent = (total_curtailed / total_potential_gen) * 100
                
                peak_gas_power_kw = hourly_data['Gas'].max()
                gas_load_factor = 0
                if peak_gas_power_kw > 0:
                    gas_load_factor = annual_gen_kwh['Gas'] / (peak_gas_power_kw * 8760)
                    
                # CO2 Emissions
                co2_emissions_g_kwh = {'Nuclear': 12, 'Hydro': 24, 'Solar': 48, 'Wind': 11, 'Gas': 490}
                total_co2_g = sum(annual_gen_kwh[source] * co2_emissions_g_kwh[source] for source in annual_gen_kwh)
                avg_emissions_intensity = total_co2_g / total_demand

                # 3. Display Results
                
                st.subheader("Key Performance Metrics")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Gas Generation", f"{annual_gen_kwh['Gas']/1e6:,.2f} MWh")
                col2.metric("Peak Gas Power (Capacity)", f"{peak_gas_power_kw/1e3:,.2f} MW")
                col3.metric("Spared Energy", f"{spared_percent:,.2f} %")
                col4.metric("Avg. CO2 Intensity", f"{avg_emissions_intensity:,.2f} gCO2/kWh")

                st.subheader("Optimal Capacity & Investment")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Installed Capacities**")
                    st.info(
                        f"**Nuclear:** {best_config['N_nuclear']} units ({best_config['N_nuclear']*params['SMR_RATED_POWER_KW']/1e3:.0f} MW)\n\n"
                        f"**Wind:** {best_config['Capacity_Wind_kW']/1e3:,.0f} MW\n\n"
                        f"**Solar:** {best_config['Capacity_Solar_kWp']/1e3:,.0f} MWp\n\n"
                        f"**Storage:** {best_config['Storage_Capacity_kWh']/1e6:,.2f} GWh"
                    )
                
                with col2:
                    st.markdown("**Investment Distribution**")
                    fig = plot_pie_chart(investment_breakdown, "Investment Distribution")
                    st.pyplot(fig)
                
                st.subheader("Annual Energy Generation Mix")
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = plot_pie_chart(annual_gen_kwh, "Annual Generation Mix (kWh)")
                    st.pyplot(fig)
                
                with col2:
                    st.dataframe(pd.DataFrame.from_dict(annual_gen_kwh, orient='index', columns=['Total MWh']).apply(lambda x: x/1e6))

                # 4. Hourly Plots
                st.subheader("Hourly System Dynamics (Full Year)")
                
                # Prepare data for Altair
                plot_df = hourly_data.reset_index().rename(columns={'index': 'Hour'})
                
                # Melt for stacked area chart
                gen_sources = ['Hydro', 'Nuclear', 'Solar', 'Wind', 'Storage_Discharge', 'Gas']
                plot_df_melted = plot_df.melt(
                    'Hour', 
                    value_vars=gen_sources,
                    var_name='Source', 
                    value_name='kWh'
                )
                
                # Stacked generation chart
                gen_chart = alt.Chart(plot_df_melted).mark_area().encode(
                    x=alt.X("Hour", axis=alt.Axis(title="Hour of Year")),
                    y=alt.Y("kWh", stack=True, axis=alt.Axis(title="Energy (kWh)")),
                    color=alt.Color("Source", scale=alt.Scale(
                        domain=['Hydro', 'Nuclear', 'Solar', 'Wind', 'Storage_Discharge', 'Gas'],
                        range=['#0072B2', '#009E73', '#F0E442', '#56B4E9', '#CC79A7', '#D55E00']
                    )),
                    tooltip=["Hour", "Source", "kWh"]
                )
                
                # Demand line chart
                demand_chart = alt.Chart(plot_df).mark_line(color='black', strokeWidth=2).encode(
                    x=alt.X("Hour"),
                    y=alt.Y("Demand", axis=alt.Axis(title="Demand (kWh)")),
                    tooltip=["Hour", "Demand"]
                )
                
                final_chart = (gen_chart + demand_chart).properties(
                    title="Hourly Demand vs. Generation"
                ).interactive()
                
                st.altair_chart(final_chart, use_container_width=True)
                
                # Storage and Curtailment plots
                st.subheader("Storage and Curtailment")
                col1, col2 = st.columns(2)
                with col1:
                    st.line_chart(plot_df.set_index('Hour')['SOC'])
                    st.caption("Storage State of Charge (kWh)")
                with col2:
                    st.area_chart(plot_df.set_index('Hour')['Curtailed'])
                    st.caption("Curtailed (Spared) Energy (kWh)")
