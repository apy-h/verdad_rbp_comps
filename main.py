import rbp
import numpy as np
import pandas as pd
import streamlit as st
from helpers import prep_greg_data
from typing import List, Optional

st.set_page_config(page_title="Company Peer Analysis", layout="wide")

def load_data(start_date: Optional[str] = None, end_date: Optional[str] = None):
    """Load and cache the data with optional date filtering"""
    # Create a cache key based on date filters
    cache_key = f"df_{start_date}_{end_date}"

    if cache_key not in st.session_state:
        with st.spinner("Loading data..."):
            st.session_state[cache_key] = prep_greg_data(start_date=start_date, end_date=end_date)
    return st.session_state[cache_key]

def get_available_months(df: pd.DataFrame) -> List[str]:
    """Get list of available YEAR_MONTH values for dropdown"""
    if 'YEAR_MONTH' in df.columns:
        # Convert periods to strings and sort
        months = df['YEAR_MONTH'].dropna().unique()
        months_str = [str(month) for month in months]
        return sorted(months_str, reverse=True)
    return []

def get_company_options(df: pd.DataFrame) -> List[str]:
    """Get list of company names and tickers for autocomplete"""
    companies = []
    if 'COMPANY_NAME' in df.columns:
        companies.extend(df['COMPANY_NAME'].dropna().unique().tolist())
    # if 'TICKER' in df.columns:
    #     tickers = df['TICKER'].dropna().unique().tolist()
    #     companies.extend([f"{ticker} (Ticker)" for ticker in tickers])
    return sorted(companies, key=str.lower)  # Sort case-insensitively

def get_numerical_columns(df: pd.DataFrame) -> List[str]:
    """Get numerical columns excluding company identifiers"""
    exclude_cols = {'DAY_DATE', 'YEAR_MONTH'}

    # Get all columns that start with specific prefixes
    exclude_prefixes = ['INDUSTRY_', 'COMPANY_', 'SECTOR_']

    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Filter out excluded columns and columns with excluded prefixes
    filtered_cols = []
    for col in numerical_cols:
        if col not in exclude_cols and not any(col.startswith(prefix) for prefix in exclude_prefixes):
            filtered_cols.append(col)

    return filtered_cols

def find_company_row(df: pd.DataFrame, company_selection: str) -> Optional[pd.Series]:
    """Find the company row based on selection"""
    # if company_selection.endswith(" (Ticker)"):
    #     ticker = company_selection.replace(" (Ticker)", "")
    #     matches = df[df['TICKER'] == ticker]
    # else:
    # Case-insensitive matching
    matches = df[df['COMPANY_NAME'].str.lower() == company_selection.lower()]

    if len(matches) > 0:
        return matches.iloc[0]
    return None

def run_peer_analysis(df: pd.DataFrame, target_company: pd.Series, features: List[str],
                     use_most_relevant: bool = True):
    """Run relevance-based peer analysis on ALL companies in the industry"""

    # Get target company's industry sector
    target_industry = target_company.get('INDUSTRY_SECTOR_W_BIOTECH', None)

    if target_industry is None or pd.isna(target_industry):
        st.error("Target company is missing INDUSTRY_SECTOR_W_BIOTECH information")
        return None, None

    # Filter to only companies in the same industry sector
    industry_filtered_df = df[df['INDUSTRY_SECTOR_W_BIOTECH'] == target_industry].copy()

    if len(industry_filtered_df) == 0:
        st.error(f"No companies found in industry sector: {target_industry}")
        return None, None

    st.info(f"Filtering analysis to {target_industry} sector: {len(industry_filtered_df)} companies available")

    # Prepare data for RBP analysis
    # Filter out rows with NaN values in selected features
    analysis_df = industry_filtered_df.dropna(subset=features)

    if len(analysis_df) == 0:
        st.error("No valid data available for selected features in the target industry sector")
        return None, None

    # Check minimum data requirements (reduced since we're analyzing all companies)
    min_required_rows = 10
    if len(analysis_df) < min_required_rows:
        st.error(f"Insufficient data in {target_industry} sector: {len(analysis_df)} companies available, but need at least {min_required_rows} for reliable analysis")
        return None, None

    # Extract attributes (X) and create prediction circumstances (X_t)
    X = analysis_df[features].values
    target_values = target_company[features].values.flatten()

    # Check for NaN values in target company using pandas isna() which handles mixed types
    target_series = target_company[features]
    if target_series.isna().any():
        missing_features = [features[i] for i in range(len(features)) if target_series.iloc[i] is pd.NA or pd.isna(target_series.iloc[i])]
        st.error(f"Target company has missing values for: {missing_features}")
        return None, None

    # Ensure all target values are numeric
    try:
        target_values = target_values.astype(float)
    except (ValueError, TypeError) as e:
        st.error(f"Error converting target company features to numeric: {str(e)}")
        return None, None

    # Check for zero variance in features (which can cause division by zero)
    for i, feature in enumerate(features):
        feature_values = X[:, i]
        if np.var(feature_values) == 0:
            st.error(f"Feature '{feature}' has zero variance (all values are the same) in {target_industry} sector. This can cause division by zero errors.")
            return None, None

        # Check for extreme values that might cause numerical issues
        if np.any(np.isinf(feature_values)) or np.any(np.isnan(feature_values)):
            st.error(f"Feature '{feature}' contains infinite or NaN values after filtering in {target_industry} sector.")
            return None, None

    # Check target values for numerical issues
    if np.any(np.isinf(target_values)) or np.any(np.isnan(target_values)):
        st.error("Target company has infinite or NaN values in selected features.")
        return None, None

    # For peer analysis, we'll use a dummy outcome variable (not predicting outcomes, just finding similarity)
    # Use market cap or first available numerical column as dummy outcome
    outcome_cols = ['MARKET_CAP_FISCAL', 'IQ_TOTAL_ASSETS', 'TEV_FISCAL']
    y_col = None
    for col in outcome_cols:
        if col in analysis_df.columns:
            y_col = col
            break

    if y_col is None:
        # Use first numerical column as fallback
        numerical_cols = analysis_df.select_dtypes(include=[np.number]).columns
        available_cols = [col for col in numerical_cols if col not in features]
        if len(available_cols) > 0:
            y_col = available_cols[0]
        else:
            st.error("No suitable outcome variable found")
            return None, None

    y = analysis_df[y_col].values.reshape(-1, 1)

    # Check outcome variable for issues
    if np.var(y.flatten()) == 0:
        st.error(f"Outcome variable '{y_col}' has zero variance in {target_industry} sector.")
        return None, None

    try:
        # Add debugging info
        st.info(f"Running analysis with {len(analysis_df)} companies in {target_industry} sector, {len(features)} features")

        # Run prediction to get ALL relevance scores (analyze all companies in the industry)
        # Use a high threshold to get all companies
        n_all_companies = len(analysis_df) - 1  # Exclude target company
        _, pred_details = rbp.predict(X, target_values, y,
                                     thresh=n_all_companies,
                                     most=use_most_relevant,
                                     pct_thresh=False)

        # Extract relevance scores
        relevance_scores = pred_details['relevance'].flatten()

        # Extract similarity and informativeness using correct keys
        similarity_scores = pred_details['sim_it'].flatten()
        informativeness_scores = pred_details['info_t'].flatten()

        # Check if we got valid relevance scores
        if len(relevance_scores) == 0:
            st.error("No relevance scores computed. Try adjusting the threshold or features.")
            return None, None

        # Exclude the target company from results (it would be the most/least relevant to itself)
        if use_most_relevant:
            # Skip the first most relevant (target company), get all others
            peer_indices = np.argsort(relevance_scores)[:-1][::-1]  # All except last (least relevant)
        else:
            # Skip the last least relevant (target company), get all others
            peer_indices = np.argsort(relevance_scores)[1:]  # All except first (most relevant)

        # Get ALL peer companies with their scores
        all_peer_companies = analysis_df.iloc[peer_indices].copy()
        all_peer_companies['RELEVANCE_SCORE'] = relevance_scores[peer_indices]

        # Add similarity and informativeness scores
        all_peer_companies['SIMILARITY_SCORE'] = similarity_scores[peer_indices]
        all_peer_companies['INFORMATIVENESS_SCORE'] = informativeness_scores[peer_indices]

        # Sort by relevance score for consistent ordering
        all_peer_companies = all_peer_companies.sort_values('RELEVANCE_SCORE', ascending=False)

        return all_peer_companies, pred_details

    except ZeroDivisionError as e:
        st.error(f"Division by zero error in RBP analysis. This might be due to insufficient data variation or extreme values in {target_industry} sector. Try selecting different features or a different time period.")
        st.error(f"Technical details: {str(e)}")
        return None, None
    except Exception as e:
        st.error(f"Error in peer analysis: {str(e)}")
        # Add more detailed error info for debugging
        st.error(f"Dataset shape: {X.shape}, Target shape: {target_values.shape}, Features: {features}")
        return None, None

def validate_features_for_analysis(df: pd.DataFrame, features: List[str]) -> tuple[bool, List[str]]:
    """Validate that selected features are suitable for analysis"""
    issues = []

    for feature in features:
        if feature not in df.columns:
            issues.append(f"'{feature}' not found in dataset")
            continue

        # Check for sufficient non-null data
        non_null_count = df[feature].count()
        if non_null_count < 10:
            issues.append(f"'{feature}' has only {non_null_count} non-null values")
            continue

        # Check for variance
        feature_values = df[feature].dropna()
        if len(feature_values) > 1 and np.var(feature_values) == 0:
            issues.append(f"'{feature}' has zero variance (all values are the same)")
            continue

        # Check for extreme values
        if np.any(np.isinf(feature_values)):
            issues.append(f"'{feature}' contains infinite values")

    return len(issues) == 0, issues

def filter_peer_results(all_peer_companies, n_peers, use_most_relevant):
    """Filter peer results based on current settings without recalculating RBP"""
    if len(all_peer_companies) == 0:
        return all_peer_companies

    # The all_peer_companies is already sorted by relevance score (descending)
    if use_most_relevant:
        # Get top n_peers (highest relevance scores)
        filtered_peers = all_peer_companies.head(n_peers)
    else:
        # Get bottom n_peers (lowest relevance scores)
        filtered_peers = all_peer_companies.tail(n_peers).iloc[::-1]  # Reverse to show least relevant first

    return filtered_peers

# Main UI
st.title("🏢 Company Peer Analysis Dashboard")
st.write("Find the most relevant comp set for a company of interest using relevance-based prediction (RBP).")

# Sidebar configuration
st.sidebar.header("🔧 Analysis Configuration")

# Check if we have existing results to determine if controls should be frozen
has_results = 'peer_results' in st.session_state
analysis_in_progress = 'analysis_started' in st.session_state
freeze_controls = has_results or analysis_in_progress

# Date filtering section
st.sidebar.subheader("📅 Date Filter")

if freeze_controls:
    # Show current selection but disable changing it
    if has_results:
        current_month = st.session_state.peer_results['selected_month']
    else:
        current_month = st.session_state.get('temp_selected_month', '')
    st.sidebar.info(f"📊 Locked to: {current_month if current_month else 'All data'}")
    selected_month = current_month
    df = load_data(start_date=selected_month, end_date=selected_month) if selected_month else load_data()
else:
    # First load data without filters to get available months
    initial_df = load_data()
    available_months = get_available_months(initial_df)

    # Month-year selection
    selected_month = st.sidebar.selectbox(
        "Select Month-Year",
        options=[""] + available_months,
        index=0,
        help="Select a specific month-year to filter data, or leave blank to see all data."
    )

    # Load filtered data based on selection
    if selected_month == "":
        df = initial_df
        st.sidebar.info(f"📊 Using all available data")
    else:
        df = load_data(start_date=selected_month, end_date=selected_month)
        st.sidebar.info(f"📊 Data filtered to: {selected_month}")

st.sidebar.divider()

# Company selection with autocomplete
if freeze_controls:
    # Show current selection but disable changing it
    if has_results:
        current_company = st.session_state.peer_results['target_company']['COMPANY_NAME']
        target_company = st.session_state.peer_results['target_company']
    else:
        current_company = st.session_state.get('temp_selected_company', '')
        target_company = st.session_state.get('temp_target_company', None)
    st.sidebar.info(f"🎯 Locked to: {current_company}")
    selected_company = current_company
else:
    company_options = get_company_options(df)
    selected_company = st.sidebar.selectbox(
        "Select Target Company",
        options=[""] + company_options,
        index=0,
        help="Start typing to search for a company name"
    )

if selected_company:
    if not freeze_controls:
        target_company = find_company_row(df, selected_company)

    if target_company is not None:
        if freeze_controls:
            # Show current features but disable changing them
            if has_results:
                current_features = st.session_state.peer_results['selected_features']
            else:
                current_features = st.session_state.get('temp_selected_features', [])
            st.sidebar.info(f"📊 Features locked: {', '.join(current_features)}")
            selected_features = current_features
        else:
            # Feature selection
            numerical_cols = get_numerical_columns(df)
            selected_features = st.sidebar.multiselect(
                "Select Features for Analysis",
                options=numerical_cols,
                default=numerical_cols[:5] if len(numerical_cols) >= 5 else numerical_cols,
                help="Choose the financial metrics to use for finding similar companies"
            )

        # Number of peers - always interactive (unless analysis is in progress)
        if has_results:
            max_possible_peers = len(st.session_state.peer_results['all_peer_companies'])
        else:
            max_possible_peers = min(50, len(df)-1)

        n_peers = st.sidebar.slider(
            "Number of Peer Companies",
            min_value=3,
            max_value=max_possible_peers,
            value=min(10, max_possible_peers),
            help="Number of most similar companies to find",
            disabled=analysis_in_progress and not has_results
        )

        # Analysis type - always interactive (unless analysis is in progress)
        use_most_relevant = st.sidebar.radio(
            "Analysis Type",
            options=[True, False],
            format_func=lambda x: "Most Relevant" if x else "Least Relevant",
            help="Find most similar or most different companies",
            disabled=analysis_in_progress and not has_results
        ) == True

        # Run analysis button or reset button
        if freeze_controls and has_results:
            if st.sidebar.button("🔄 Start New Analysis", type="secondary"):
                # Clear all analysis states to unfreeze controls
                keys_to_remove = ['peer_results', 'analysis_started', 'temp_selected_month',
                                'temp_selected_company', 'temp_target_company', 'temp_selected_features']
                for key in keys_to_remove:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        elif analysis_in_progress and not has_results:
            st.sidebar.info("🔄 Analysis in progress...")
            # Show a disabled button during analysis
            st.sidebar.button("🔍 Find Peer Companies", type="primary", disabled=True)
        else:
            if st.sidebar.button("🔍 Find Peer Companies", type="primary"):
                if len(selected_features) == 0:
                    st.error("Please select at least one feature for analysis.")
                else:
                    # Lock controls immediately by setting analysis state
                    st.session_state.analysis_started = True
                    st.session_state.temp_selected_month = selected_month
                    st.session_state.temp_selected_company = selected_company
                    st.session_state.temp_target_company = target_company
                    st.session_state.temp_selected_features = selected_features

                    # Validate features before running analysis
                    features_valid, feature_issues = validate_features_for_analysis(df, selected_features)

                    if not features_valid:
                        st.error("Issues with selected features:")
                        for issue in feature_issues:
                            st.error(f"• {issue}")
                        st.info("Try selecting different features or a different time period.")
                        # Clear analysis state if validation fails
                        keys_to_remove = ['analysis_started', 'temp_selected_month',
                                        'temp_selected_company', 'temp_target_company', 'temp_selected_features']
                        for key in keys_to_remove:
                            if key in st.session_state:
                                del st.session_state[key]
                    else:
                        # Force rerun immediately to show locked state
                        st.rerun()

# Check if we should run the analysis (after the rerun)
if 'analysis_started' in st.session_state and 'peer_results' not in st.session_state:
    # Run the analysis
    temp_month = st.session_state.get('temp_selected_month', '')
    temp_company = st.session_state.get('temp_selected_company', '')
    temp_target = st.session_state.get('temp_target_company', None)
    temp_features = st.session_state.get('temp_selected_features', [])

    # Load appropriate data
    if temp_month:
        df = load_data(start_date=temp_month, end_date=temp_month)
    else:
        df = load_data()

    if temp_target is not None and temp_features:
        with st.spinner("Running peer analysis on all companies in industry..."):
            # Run analysis on ALL companies in the industry
            all_peer_companies, analysis_details = run_peer_analysis(
                df, temp_target, temp_features, True  # Default to most relevant for initial analysis
            )

            if all_peer_companies is not None:
                st.session_state.peer_results = {
                    'target_company': temp_target,
                    'all_peer_companies': all_peer_companies,  # Store ALL analyzed companies
                    'selected_features': temp_features,
                    'analysis_details': analysis_details,
                    'selected_month': temp_month
                }
                # Clear temporary state now that we have results
                keys_to_remove = ['analysis_started', 'temp_selected_month',
                                'temp_selected_company', 'temp_target_company', 'temp_selected_features']
                for key in keys_to_remove:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
            else:
                # Clear analysis state if analysis fails
                keys_to_remove = ['analysis_started', 'temp_selected_month',
                                'temp_selected_company', 'temp_target_company', 'temp_selected_features']
                for key in keys_to_remove:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

# Main content area
if selected_company and target_company is not None:
    # Display target company info
    st.subheader(f"🎯 Target Company: {selected_company}")

    col1, col2, col3 = st.columns(3)

    with col1:
        if 'INDUSTRY_SECTOR_W_BIOTECH' in target_company.index:
            st.metric("Industry", target_company['INDUSTRY_SECTOR_W_BIOTECH'])

    with col2:
        if 'MARKET_CAP_FISCAL' in target_company.index:
            market_cap = target_company['MARKET_CAP_FISCAL']
            if pd.notna(market_cap):
                st.metric("Market Cap", f"${market_cap:,.0f}M")

    with col3:
        if 'TEV_FISCAL' in target_company.index:
            tev = target_company['TEV_FISCAL']
            if pd.notna(tev):
                st.metric("TEV", f"${tev:,.0f}M")

    # Display results if available
    if 'peer_results' in st.session_state:
        results = st.session_state.peer_results

        # Filter peer companies based on current settings
        all_peer_companies = results['all_peer_companies']
        peer_companies = filter_peer_results(all_peer_companies, n_peers, use_most_relevant)

        selected_features = results['selected_features']

        st.subheader(f'📝 {"Most" if use_most_relevant else "Least"} Relevant Peer Companies')

        # Create display dataframe with target company included
        display_cols = ['COMPANY_NAME', 'RELEVANCE_SCORE', 'SIMILARITY_SCORE', 'INFORMATIVENESS_SCORE', 'MARKET_CAP_FISCAL']

        # Add selected features to display
        display_cols.extend([col for col in selected_features if col in peer_companies.columns])

        # Create target company row with same structure as peer companies
        target_row = target_company.copy()

        # Get actual scores for target company from analysis results
        analysis_details = results['analysis_details']

        # FIXME
        # Target company has maximum relevance (1.0) to itself
        target_row['RELEVANCE_SCORE'] = 1.0

        # Add actual similarity and informativeness scores if they exist
        if 'SIMILARITY_SCORE' in peer_companies.columns:
            # Target company has maximum similarity (1.0) to itself
            target_row['SIMILARITY_SCORE'] = 1.0

        if 'INFORMATIVENESS_SCORE' in peer_companies.columns:
            # Use info_i (informativeness of target) from prediction details
            if 'info_i' in analysis_details:
                target_informativeness = analysis_details['info_i']
                if hasattr(target_informativeness, 'flatten'):
                    target_informativeness = target_informativeness.flatten()[0]
                elif isinstance(target_informativeness, (list, np.ndarray)):
                    target_informativeness = target_informativeness[0]
                target_row['INFORMATIVENESS_SCORE'] = target_informativeness
            else:
                # Fallback - use average of peer informativeness or 1.0
                target_row['INFORMATIVENESS_SCORE'] = 1.0

        # Combine target company with peer companies
        combined_df = pd.concat([target_row.to_frame().T, peer_companies], ignore_index=True)

        # Format the dataframe for display
        display_df = combined_df[display_cols].copy()

        # Format all scores (including target)
        for idx in range(len(display_df)):
            # Format relevance score
            relevance_score = display_df.loc[idx, 'RELEVANCE_SCORE']
            display_df.loc[idx, 'RELEVANCE_SCORE'] = round(float(relevance_score), 4)

            sim_score = display_df.loc[idx, 'SIMILARITY_SCORE']
            display_df.loc[idx, 'SIMILARITY_SCORE'] = round(float(sim_score), 4)

            info_score = display_df.loc[idx, 'INFORMATIVENESS_SCORE']
            display_df.loc[idx, 'INFORMATIVENESS_SCORE'] = round(float(info_score), 4)

        if 'MARKET_CAP_FISCAL' in display_df.columns:
            display_df['MARKET_CAP_FISCAL'] = display_df['MARKET_CAP_FISCAL'].apply(
                lambda x: f"${x:,.0f}M" if pd.notna(x) and x != 'TARGET' else "N/A"
            )

        # Style the dataframe to highlight the target company (first row)
        def highlight_target(row):
            # Target company is always the first row (index 0)
            if row.name == 0:
                return ['background-color: #ff4b4b; font-weight: bold'] * len(row)
            else:
                return [''] * len(row)

        styled_df = display_df.style.apply(highlight_target, axis=1)

        st.dataframe(styled_df, use_container_width=True)

        # Feature comparison chart
        st.subheader("📈 Feature Comparison")

        if len(selected_features) > 0:
            # Create comparison dataframe
            comparison_data = []

            # Add target company
            target_row = {'Company': selected_company, 'Type': 'Target'}
            for feature in selected_features:
                target_row[feature] = target_company[feature] if feature in target_company.index else np.nan
            comparison_data.append(target_row)

            # Add peer companies
            for idx, (_, peer) in enumerate(peer_companies.iterrows()):
                peer_row = {'Company': peer['COMPANY_NAME'], 'Type': f'Peer {idx+1}'}
                for feature in selected_features:
                    peer_row[feature] = peer[feature] if feature in peer.index else np.nan
                comparison_data.append(peer_row)

            comparison_df = pd.DataFrame(comparison_data)

            # Create tabs for different visualizations
            tab1, tab2 = st.tabs(["📊 Bar Chart", "📋 Data Table"])

            with tab1:
                # Select feature to visualize
                feature_to_plot = st.selectbox(
                    "Select feature to visualize:",
                    options=selected_features,
                    key="feature_plot"
                )

                if feature_to_plot:
                    chart_data = comparison_df[['Company', 'Type', feature_to_plot]].copy()
                    chart_data = chart_data.dropna(subset=[feature_to_plot])

                    if len(chart_data) > 0:
                        st.bar_chart(
                            chart_data.set_index('Company')[feature_to_plot],
                            use_container_width=True
                        )

            with tab2:
                st.dataframe(comparison_df, use_container_width=True)

else:
    # Default view when no company is selected
    st.info("👈 Please select a company from the sidebar to begin peer analysis.")

    # Show data overview
    st.subheader("📋 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Total Companies", f"{df['COMPANY_ID'].nunique():,}")
    with col3:
        st.metric("Available Features", len(get_numerical_columns(df)))
    with col4:
        if 'INDUSTRY_SECTOR_W_BIOTECH' in df.columns:
            st.metric("Industries", df['INDUSTRY_SECTOR_W_BIOTECH'].nunique())

    # Show sample data
    st.subheader("📃 Sample Data")

    # Add controls for data exploration
    col1, col2 = st.columns(2)

    with col1:
        # Limit number of rows displayed
        max_rows = st.slider(
            "Number of rows to display",
            min_value=100,
            max_value=min(5000, len(df)),
            value=min(1000, len(df)),
            step=100,
            help="Adjust to control data size. Large datasets may cause performance issues."
        )

    with col2:
        # Column selection for display
        available_cols = [col for col in ['COMPANY_NAME', 'INDUSTRY_SECTOR_W_BIOTECH', 'MARKET_CAP_FISCAL', 'YEAR_MONTH'] if col in df.columns]
        # Add a few numerical columns
        numerical_sample = get_numerical_columns(df)[:5]  # First 5 numerical columns
        available_cols.extend([col for col in numerical_sample if col not in available_cols])

        selected_display_cols = st.multiselect(
            "Select columns to display",
            options=df.columns.tolist(),
            default=available_cols,
            help="Choose which columns to show. Fewer columns = faster loading."
        )

    if selected_display_cols:
        # Add sorting controls
        col_sort1, col_sort2, col_sort3 = st.columns(3)

        with col_sort1:
            sort_column = st.selectbox(
                "Sort by column",
                options=["None"] + selected_display_cols,
                index=0,
                help="Select a column to sort the entire dataset"
            )

        with col_sort2:
            sort_ascending = st.radio(
                "Sort order",
                options=[True, False],
                format_func=lambda x: "Ascending" if x else "Descending",
                index=0,
                help="Choose sort direction"
            )

        with col_sort3:
            # Add a search/filter option
            search_term = st.text_input(
                "Search in COMPANY_NAME",
                placeholder="Enter company name to filter...",
                help="Filter companies by name (case-insensitive)"
            )

        # Apply sorting and filtering to the full dataset
        filtered_df = df.copy()

        # Apply search filter if provided
        if search_term and 'COMPANY_NAME' in df.columns:
            filtered_df = filtered_df[
                filtered_df['COMPANY_NAME'].str.contains(search_term, case=False, na=False)
            ]
            st.info(f"Found {len(filtered_df)} companies matching '{search_term}'")

        # Apply sorting to the full filtered dataset
        if sort_column != "None":
            try:
                filtered_df = filtered_df.sort_values(
                    by=sort_column,
                    ascending=sort_ascending,
                    na_position='last'  # Put NaN values at the end
                )
                st.info(f"Dataset sorted by '{sort_column}' ({'ascending' if sort_ascending else 'descending'})")
            except Exception as e:
                st.warning(f"Could not sort by '{sort_column}': {str(e)}")

        # Now take the sample from the sorted/filtered data
        sample_df = filtered_df[selected_display_cols].head(max_rows)

        st.info(f"Showing {len(sample_df)} rows out of {len(filtered_df)} total rows with {len(selected_display_cols)} columns")

        # Display the data (remove height parameter to allow full interaction)
        st.dataframe(
            sample_df,
            use_container_width=True,
            height=400
        )

        # # Option to download the current filtered/sorted dataset
        # col_download1, col_download2 = st.columns(2)

        # with col_download1:
        #     if st.button("📥 Download Filtered Data (CSV)"):
        #         csv_data = filtered_df.to_csv(index=False)
        #         st.download_button(
        #             label="Click to Download Filtered Data",
        #             data=csv_data,
        #             file_name=f"filtered_data_{selected_month.replace(' ', '_') if selected_month != '' else 'all'}.csv",
        #             mime="text/csv"
        #         )

        # with col_download2:
        #     if st.button("📥 Download Full Dataset (CSV)"):
        #         csv_data = df.to_csv(index=False)
        #         st.download_button(
        #             label="Click to Download Full Dataset",
        #             data=csv_data,
        #             file_name=f"company_data_{selected_month.replace(' ', '_') if selected_month != '' else 'all'}.csv",
        #             mime="text/csv"
        #         )
    else:
        st.warning("Please select at least one column to display")

# Footer
st.markdown("---")
st.markdown("*Designed by Aarna Pal-Yadav*")