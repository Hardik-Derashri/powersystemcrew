import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import io

# --- SKOPT IMPORTS ---
import skopt
from skopt import gp_minimize
from skopt.space import Integer
# ---------------------

# ----------------------------------------------------------------------
# 1. HELPER FUNCTIONS (MAX BOUNDS, PLOTTING, DATA LOADING)
# ----------------------------------------------------------------------

@st.cache_data
def load_data(demand_file, solar_file, wind_file):
    """
    Loads and merges the three uploaded CSV files.
    """
    try:
        # Read demand
        demand_df = pd.read_csv(demand_file)
        # Note: Assumes the demand file has an 'Mwh' column
        demand_kwh = demand_df['Mwh'].astype(float) * 1000 # Convert MWh to kWh

        # Read solar (skip headers from Renewables.ninja)
        solar_df = pd.read_csv(solar_file, skiprows=3)
        # Note: Assumes 'electricity [kWh]' column for per kWp output
        solar_kw_per_kwp = solar_df['electricity [kWh]'].astype(float)

        # Read wind (skip headers from Renewables.ninja)
        wind_df = pd.read_csv(wind_file, skiprows=3)
        # Note: Assumes 'electricity [kWh]' column for per kW output
        wind_kw_per_kw = wind_df['electricity [kWh]'].astype(float)

        # Combine
        max_len = min(len(demand_kwh), len(solar_kw_per_kwp), len(wind_kw_per_kw))
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

def calculate_max_bounds(params):
    """
    Calculates the maximum number of units possible within the budget.
    """
    Nmax_mill = params['BUDGET'] // params['COST_WIND_UNIT']
    Nmax_panel = params['BUDGET'] // params['COST_SOLAR_UNIT']
    Nmax_SMR = params['BUDGET'] // params['COST_NUCLEAR_UNIT']
    # Storage max is not needed as it uses the remainder of the budget
    
    # Set a practical upper limit for solar panels to prevent overflow/unrealistically large numbers
    Nmax_panel = min(Nmax_panel, 100_000_000) 

    return int(Nmax_mill), int(Nmax_panel), int(Nmax_SMR)

# ----------------------------------------------------------------------
# 2. CORE SIMULATION ENGINE (Objective Function for skopt)
# ----------------------------------------------------------------------

def gas_production_skopt(N_input, data_frame, params):
    
    # N_input is the tuple from the optimizer: (N_mill, N_panel, N_SMR)
    N_mill, N_panel, N_SMR = N_input
    
    # Calculate Capacities (kW)
    Capacity_Wind_kW = N_mill * params['WIND_RATED_POWER_KW']
    Capacity_Solar_kWp = N_panel * params['SOLAR_RATED_POWER_KWp']
    P_nuc = N_SMR * params['SMR_RATED_POWER_KW']
    
    # Calculate Total Cost based on number of units
    cost = (
        N_mill * params['COST_WIND_UNIT'] + 
        N_panel * params['COST_SOLAR_UNIT'] + 
        N_SMR * params['COST_NUCLEAR_UNIT']
    )
    
    # Apply hard constraint (penalty)
    if cost > params['BUDGET']:
        return 1e18  # Massive penalty for budget overrun
    
    # Remaining budget determines storage capacity (based on user's logic)
    remaining_budget = params['BUDGET'] - cost
    
    # Note: CostStorage is in €/kWh, so remaining_budget/CostStorage is kWh
    capacity_storage_kwh = remaining_budget / params['COST_STORAGE_KWH'] 
    
    # ----------- Start Hourly Simulation -----------
    
    P_hydro = params['HYDRO_CAPACITY_KW'] 
    P_storage_max_dispatch = capacity_storage_kwh / params['STORAGE_DISPATCH_HOURS']
    
    # Time Series Generation
    Gen_Baseload_kWh = P_nuc + P_hydro
    Gen_Solar_kWh = data_frame['Solar_kW_per_kWp'] * Capacity_Solar_kWp
    Gen_Wind_kWh = data_frame['Wind_kW_per_kW'] * Capacity_Wind_kW
    
    Demand_kWh_ts = data_frame['Demand_kWh']
    
    SOC = np.zeros(len(data_frame))
    SOC[0] = capacity_storage_kwh * params['STORAGE_INITIAL_SOC']
    
    Gas_Generation_kWh = np.zeros(len(data_frame))
    current_soc = SOC[0]
    
    for t in range(len(data_frame)):
        Demand_t = Demand_kWh_ts.iloc[t]
        Gen_Total_t = Gen_Baseload_kWh + Gen_Solar_kWh.iloc[t] + Gen_Wind_kWh.iloc[t]
        Balance_t = Gen_Total_t - Demand_t
        
        if Balance_t >= 0: # Surplus: Store or Curtail
            Surplus_t = Balance_t
            Max_Charge_t = capacity_storage_kwh - current_soc
            
            # The user's script applies efficiency upon discharge, let's keep it simple here
            # and use the charging efficiency logic from my previous script, as it's common.
            # Storable_Energy_t = Surplus_t * params['STORAGE_EFFICIENCY']
            
            # Storable energy based on surplus power (max C/T rate)
            Storable_Energy_t = min(Surplus_t, P_storage_max_dispatch) 
            
            Energy_Charged_t = min(Storable_Energy_t, Max_Charge_t)
            
            current_soc += Energy_Charged_t * params['STORAGE_EFFICIENCY'] # Apply efficiency on charging
            Gas_Generation_kWh[t] = 0
            
        else: # Deficit: Discharge Storage, then use Gas
            Deficit_t = abs(Balance_t)
            Max_Discharge_t = min(P_storage_max_dispatch, current_soc)
            
            Energy_Discharged_t = min(Max_Discharge_t, Deficit_t)
            
            # Note: We discharge the energy available (current_soc) and assume perfect discharge efficiency
            # as the round-trip efficiency was applied during charging.
            current_soc -= Energy_Discharged_t
            
            Remaining_Deficit = Deficit_t - Energy_Discharged_t
            Gas_Generation_kWh[t] = Remaining_Deficit
            
        # Ensure SOC stays within bounds
        current_soc = np.clip(current_soc, 0, capacity_storage_kwh)
        SOC[t] = current_soc
        
    total_gas = np.sum(Gas_Generation_kWh)
    
    # Skopt only needs the objective function value
    return total_gas

# ----------------------------------------------------------------------
# 3. OPTIMIZATION FUNCTION (The skopt Wrapper)
# ----------------------------------------------------------------------

@st.cache_resource
def gas_minimization(_data_df, params):
    """
    Runs Bayesian optimization using gp_minimize.
    """
    Nmax_mill, Nmax_panel, Nmax_SMR = calculate_max_bounds(params)
    
    st.info(f"Optimization space bounds:\nWind Units: 0 to {Nmax_mill}\nSolar Units: 0 to {Nmax_panel}\nNuclear Units: 0 to {Nmax_SMR}")
    
    # Define the search space (integer bounds for number of units)
    space = [
        Integer(0, Nmax_mill, name='N_mill'),
        Integer(0, Nmax_panel, name='N_panel'),
        Integer(0, Nmax_SMR, name='N_SMR')
    ]
    
    # Define the objective function wrapper for skopt
    def objective_function(N_input):
        return gas_production_skopt(N_input, _data_df, params)

    # Define initial starting points (similar to user's code, but adjusted for scale)
    initial_x0 = [
        min(10, Nmax_mill), 
        min(1000000, Nmax_panel), 
        min(15, Nmax_SMR)
    ]
    
    with st.spinner(f"Running Bayesian Optimization with {params['SKOPT_N_CALLS']} calls..."):
        
        result = gp_minimize(
            func = objective_function,
            dimensions = space,
            acq_func = "EI", 
            n_calls = params['SKOPT_N_CALLS'],
            n_random_starts = params['SKOPT_N_RANDOM_STARTS'],
            x0 = initial_x0,
            random_state = 1234
        )
        
    # Extract optimal configuration
    N_mill_opt, N_panel_opt, N_SMR_opt = result.x
    
    # Recalculate full results for the optimal mix
    best_config = {
        'N_nuclear': N_SMR_opt,
        'N_wind_units': N_mill_opt,
        'N_solar_units': N_panel_opt,
        # Calculate capacities and costs one last time
        'Capacity_Wind_kW': N_mill_opt * params['WIND_RATED_POWER_KW'],
        'Capacity_Solar_kWp': N_panel_opt * params['SOLAR_RATED_POWER_KWp'],
    }
    
    # Rerun the simulation to get the full time series (SOC, Gas_Gen, Cost)
    total_gas, total_curtailed, final_cost, SOC, Gas_Gen = run_simulation_full_results(
        N_mill_opt, N_panel_opt, N_SMR_opt, _data_df, params
    )
    
    # Add calculated storage capacity to config
    cost_opt = N_mill_opt * params['COST_WIND_UNIT'] + N_panel_opt * params['COST_SOLAR_UNIT'] + N_SMR_opt * params['COST_NUCLEAR_UNIT']
    storage_kwh = (params['BUDGET'] - cost_opt) / params['COST_STORAGE_KWH']
    best_config['Storage_Capacity_kWh'] = storage_kwh
    
    best_results = {
        'Total_Gas_kWh': total_gas,
        'Total_Curtailed_kWh': total_curtailed,
        'Final_Cost': final_cost,
        'SOC': SOC,
        'Gas_Generation_kWh_TS': Gas_Gen
    }

    return best_config, best_results

# ----------------------------------------------------------------------
# 4. FULL SIMULATION FUNCTION (For extracting results/plotting after optimization)
# ----------------------------------------------------------------------

# NOTE: This function is nearly identical to gas_production_skopt but returns
# the full set of simulation results and time series.
def run_simulation_full_results(N_mill, N_panel, N_SMR, data_frame, params):
    
    # Calculate Capacities (kW) and Costs
    Capacity_Wind_kW = N_mill * params['WIND_RATED_POWER_KW']
    Capacity_Solar_kWp = N_panel * params['SOLAR_RATED_POWER_KWp']
    P_nuc = N_SMR * params['SMR_RATED_POWER_KW']
    
    cost = (
        N_mill * params['COST_WIND_UNIT'] + 
        N_panel * params['COST_SOLAR_UNIT'] + 
        N_SMR * params['COST_NUCLEAR_UNIT']
    )
    
    remaining_budget = params['BUDGET'] - cost
    capacity_storage_kwh = remaining_budget / params['COST_STORAGE_KWH'] 
    
    # ----------- Start Hourly Simulation -----------
    P_hydro = params['HYDRO_CAPACITY_KW'] 
    P_storage_max_dispatch = capacity_storage_kwh / params['STORAGE_DISPATCH_HOURS']
    
    Gen_Baseload_kWh = P_nuc + P_hydro
    Gen_Solar_kWh = data_frame['Solar_kW_per_kWp'] * Capacity_Solar_kWp
    Gen_Wind_kWh = data_frame['Wind_kW_per_kW'] * Capacity_Wind_kW
    
    Demand_kWh_ts = data_frame['Demand_kWh']
    
    SOC = np.zeros(len(data_frame))
    SOC[0] = capacity_storage_kwh * params['STORAGE_INITIAL_SOC']
    
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
            
            Storable_Energy_t = min(Surplus_t, P_storage_max_dispatch) 
            Energy_Charged_t = min(Storable_Energy_t, Max_Charge_t)
            
            current_soc += Energy_Charged_t * params['STORAGE_EFFICIENCY']
            Curtailed_Energy_kWh[t] = Surplus_t - Energy_Charged_t
            Gas_Generation_kWh[t] = 0
            
        else: # Deficit: Discharge Storage, then use Gas
            Deficit_t = abs(Balance_t)
            Max_Discharge_t = min(P_storage_max_dispatch, current_soc)
            
            Energy_Discharged_t = min(Max_Discharge_t, Deficit_t)
            
            current_soc -= Energy_Discharged_t
            
            Remaining_Deficit = Deficit_t - Energy_Discharged_t
            Gas_Generation_kWh[t] = Remaining_Deficit
            Curtailed_Energy_kWh[t] = 0
            
        current_soc = np.clip(current_soc, 0, capacity_storage_kwh)
        SOC[t] = current_soc
        
    total_gas = np.sum(Gas_Generation_kWh)
    total_curtailed = np.sum(Curtailed_Energy_kWh)
    
    return total_gas, total_curtailed, cost, SOC, Gas_Generation_kWh

# ----------------------------------------------------------------------
# 5. STREAMLIT APP LAYOUT (Modified to handle unit costs/powers)
# ----------------------------------------------------------------------

# ... (Plotting helper functions remain the same) ...
def plot_pie_chart(data, title):
    # ... (same as before) ...
    labels = data.keys()
    sizes = data.values()
    
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal') 
    ax.set_title(title)
    return fig

def get_hourly_timeseries(config, _data_df, params, best_results):
    """
    Re-creates the full hourly timeseries for the optimal mix for plotting.
    Uses the results of the full simulation run.
    """
    Capacity_Wind_kW = config['N_wind_units'] * params['WIND_RATED_POWER_KW']
    Capacity_Solar_kWp = config['N_solar_units'] * params['SOLAR_RATED_POWER_KWp']
    P_nuc = config['N_nuclear'] * params['SMR_RATED_POWER_KW']
    capacity_storage_kwh = config['Storage_Capacity_kWh']
    
    P_storage_max_dispatch = capacity_storage_kwh / params['STORAGE_DISPATCH_HOURS']
    
    SOC = best_results['SOC']
    
    Gen_Hydro_kWh = pd.Series(np.repeat(params['HYDRO_CAPACITY_KW'], len(_data_df)))
    Gen_Nuclear_kWh = pd.Series(np.repeat(P_nuc, len(_data_df)))
    Gen_Solar_kWh = _data_df['Solar_kW_per_kWp'] * Capacity_Solar_kWp
    Gen_Wind_kWh = _data_df['Wind_kW_per_kW'] * Capacity_Wind_kW
    Demand_kWh_ts = _data_df['Demand_kWh']
    
    # Calculate flows based on SOC change (simplified for plotting)
    # The discharge/charge flows are complex to calculate *perfectly* from SOC alone due to efficiency.
    # We will approximate based on the full simulation run's logic to maintain consistency.
    
    Storage_Charge_kWh = np.zeros(len(_data_df))
    Storage_Discharge_kWh = np.zeros(len(_data_df))
    
    current_soc = SOC[0]
    
    for t in range(len(_data_df)):
        Demand_t = Demand_kWh_ts.iloc[t]
        Gen_Total_t = Gen_Hydro_kWh.iloc[t] + Gen_Nuclear_kWh.iloc[t] + Gen_Solar_kWh.iloc[t] + Gen_Wind_kWh.iloc[t]
        Balance_t = Gen_Total_t - Demand_t
        
        if Balance_t >= 0: 
            Surplus_t = Balance_t
            Max_Charge_t = capacity_storage_kwh - current_soc
            Storable_Energy_t = min(Surplus_t, P_storage_max_dispatch) 
            Energy_Charged_t = min(Storable_Energy_t, Max_Charge_t)
            
            # This is the energy that *actually* goes into the battery (after efficiency)
            Storage_Charge_kWh[t] = Energy_Charged_t * params['STORAGE_EFFICIENCY']
            current_soc += Storage_Charge_kWh[t]
            
        else: 
            Deficit_t = abs(Balance_t)
            Max_Discharge_t = min(P_storage_max_dispatch, current_soc)
            Energy_Discharged_t = min(Max_Discharge_t, Deficit_t)
            
            # This is the useful energy coming *out* of the battery
            Storage_Discharge_kWh[t] = Energy_Discharged_t
            current_soc -= Energy_Discharged_t
            
        current_soc = np.clip(current_soc, 0, capacity_storage_kwh)
    
    hourly_df = pd.DataFrame({
        'Demand': Demand_kWh_ts,
        'Hydro': Gen_Hydro_kWh,
        'Nuclear': Gen_Nuclear_kWh,
        'Solar': Gen_Solar_kWh,
        'Wind': Gen_Wind_kWh,
        'Gas': best_results['Gas_Generation_kWh_TS'], # Use the exact Gas production from the optimization run
        'Storage_Discharge': Storage_Discharge_kWh,
        'Storage_Charge': Storage_Charge_kWh,
        'Curtailed': best_results['Total_Curtailed_kWh'] * np.ones(len(_data_df)) / len(_data_df), # Crude representation
        'SOC': best_results['SOC']
    })
    
    return hourly_df


st.set_page_config(layout="wide")
st.title("⚡ Power System Optimization Tool (Bayesian)")

# --- SIDEBAR ---
st.sidebar.header("Configuration")

# File Uploaders
st.sidebar.subheader("1. Upload Data Files")
demand_file = st.sidebar.file_uploader("Upload Demand CSV", type="csv")
solar_file = st.sidebar.file_uploader("Upload Solar CSV (Renewables.ninja)", type="csv")
wind_file = st.sidebar.file_uploader("Upload Wind CSV (Renewables.ninja)", type="csv")

params = {}

# Technology Parameters (using unit costs and rated powers from your data)
st.sidebar.subheader("2. Technology Parameters")

# Nuclear
params['SMR_RATED_POWER_KW'] = st.sidebar.number_input("SMR Rated Power (kW/unit)", value=50_000, format="%d", key='P_NUC')
params['COST_NUCLEAR_UNIT'] = st.sidebar.number_input("Nuclear Unit Cost (€)", value=250_000_000, format="%d", key='C_NUC')

# Wind (Mill)
params['WIND_RATED_POWER_KW'] = st.sidebar.number_input("Wind Rated Power (kW/unit)", value=5000, format="%d", key='P_WIND')
params['COST_WIND_UNIT'] = st.sidebar.number_input("Wind Mill Unit Cost (€)", value=6_500_000, format="%d", key='C_WIND')

# Solar (Panel)
params['SOLAR_RATED_POWER_KWp'] = st.sidebar.number_input("Solar Rated Power (kWp/unit)", value=0.55, key='P_SOLAR')
params['COST_SOLAR_UNIT'] = st.sidebar.number_input("Solar Panel Unit Cost (€)", value=600.0, key='C_SOLAR')

# Storage
params['COST_STORAGE_KWH'] = st.sidebar.number_input("Storage Cost (€/kWh)", value=100.0, key='C_STORAGE')
params['STORAGE_DISPATCH_HOURS'] = st.sidebar.number_input("Storage Dispatch Time (h)", value=6, key='T_STORAGE')
params['STORAGE_EFFICIENCY'] = st.sidebar.slider("Storage Efficiency (Roundtrip)", 0.0, 1.0, 0.85, key='ETA_STORAGE')
params['STORAGE_INITIAL_SOC'] = st.sidebar.slider("Storage Initial SOC (%)", 0.0, 1.0, 0.5, key='SOC_INIT')

# Hydro (Baseload)
params['HYDRO_CAPACITY_KW'] = st.sidebar.number_input("Hydro Capacity (kW)", value=258_996, format="%d", key='P_HYDRO')

st.sidebar.subheader("3. Financial Parameters")
params['BUDGET'] = st.sidebar.number_input("Total Investment Budget (€)", value=8_920_200_000, format="%d", key='BUDGET')

st.sidebar.subheader("4. Optimization Settings (skopt)")
params['SKOPT_N_CALLS'] = st.sidebar.number_input("Total Optimizer Iterations", value=150, min_value=10, max_value=500)
params['SKOPT_N_RANDOM_STARTS'] = st.sidebar.number_input("Random Initialization Points", value=100, min_value=10, max_value=400)


# --- MAIN PANEL ---
if not (demand_file and solar_file and wind_file):
    st.info("Welcome! Please upload Demand, Solar, and Wind data files in the sidebar to begin.")
else:
    data_df = load_data(demand_file, solar_file, wind_file)
    
    if data_df is not None:
        st.success(f"Successfully loaded {len(data_df)} hours of data.")
        
        # Run Button
        if st.sidebar.button("🚀 Run Bayesian Optimization"):
            
            # 1. Run Optimization
            best_config, best_results = gas_minimization(data_df, params)
            
            if best_config is None:
                st.error("Optimization failed.")
            else:
                st.header("✨ Optimal Power Mix Found (Bayesian Optimization) ✨")
                
                # 2. Calculate final metrics
                
                # Investment Breakdown
                cost_nuc = best_config['N_nuclear'] * params['COST_NUCLEAR_UNIT']
                cost_wind = best_config['N_wind_units'] * params['COST_WIND_UNIT']
                cost_solar = best_config['N_solar_units'] * params['COST_SOLAR_UNIT']
                cost_storage = best_config['Storage_Capacity_kWh'] * params['COST_STORAGE_KWH']
                
                investment_breakdown = {
                    'Nuclear': cost_nuc,
                    'Wind': cost_wind,
                    'Solar': cost_solar,
                    'Storage': cost_storage,
                }
                
                # Generation Breakdown
                hourly_data = get_hourly_timeseries(best_config, data_df, params, best_results)
                
                annual_gen_kwh = {
                    'Hydro': hourly_data['Hydro'].sum(),
                    'Nuclear': hourly_data['Nuclear'].sum(),
                    'Solar': hourly_data['Solar'].sum(),
                    'Wind': hourly_data['Wind'].sum(),
                    'Gas (Required)': hourly_data['Gas'].sum(),
                }
                
                # Other Metrics
                total_demand = hourly_data['Demand'].sum()
                total_curtailed = best_results['Total_Curtailed_kWh']
                total_potential_gen = annual_gen_kwh['Hydro'] + annual_gen_kwh['Nuclear'] + annual_gen_kwh['Solar'] + annual_gen_kwh['Wind']
                spared_percent = (total_curtailed / total_potential_gen) * 100
                
                peak_gas_power_kw = hourly_data['Gas'].max()
                
                # CO2 Emissions (Using 490 gCO2/kWh for Gas from your data)
                co2_emissions_g_kwh = {'Nuclear': 12, 'Hydro': 24, 'Solar': 48, 'Wind': 11, 'Gas (Required)': 490}
                total_co2_g = sum(annual_gen_kwh[source] * co2_emissions_g_kwh.get(source, 0) for source in annual_gen_kwh)
                avg_emissions_intensity = total_co2_g / total_demand

                # 3. Display Results
                
                st.subheader("Key Performance Metrics")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Gas Generation", f"{annual_gen_kwh['Gas (Required)']/1e6:,.2f} MWh")
                col2.metric("Peak Gas Power (Capacity)", f"{peak_gas_power_kw/1e3:,.2f} MW")
                col3.metric("Spared Energy", f"{spared_percent:,.2f} %")
                col4.metric("Avg. CO2 Intensity", f"{avg_emissions_intensity:,.2f} gCO2/kWh")

                st.subheader("Optimal Capacity & Investment")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Installed Capacities**")
                    st.info(
                        f"**Nuclear:** {best_config['N_nuclear']} units ({best_config['Capacity_Wind_kW']/1e3:.0f} MW)\n\n"
                        f"**Wind:** {best_config['N_wind_units']} units ({best_config['Capacity_Wind_kW']/1e3:,.0f} MW)\n\n"
                        f"**Solar:** {best_config['N_solar_units']:,.0f} units ({best_config['Capacity_Solar_kWp']/1e3:,.0f} MWp)\n\n"
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
                    y=alt.Y("Demand"),
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
