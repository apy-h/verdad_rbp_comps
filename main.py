import rbp
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from helpers import prep_greg_data
from typing import List, Optional

RED = '#ff4b4b'

st.set_page_config(page_title="Company Peer Analysis", layout="wide")

def clean_column_name(col_name: str) -> str:
    """Remove prefixes from column names for display purposes"""
    prefixes_to_remove = ['IQ_', 'D_VALUE_BBG_', 'D_APY_']

    for prefix in prefixes_to_remove:
        if col_name.startswith(prefix):
            return col_name[len(prefix):]

    return col_name

def create_column_mapping(columns: List[str]) -> dict:
    """Create a mapping from original column names to cleaned display names, handling duplicates"""
    mapping = {}
    cleaned_names_count = {}

    for col in columns:
        cleaned = clean_column_name(col)

        # Check if this cleaned name already exists
        if cleaned in cleaned_names_count:
            cleaned_names_count[cleaned] += 1
            # Add a suffix to make it unique
            mapping[col] = f"{cleaned}_{cleaned_names_count[cleaned]}"
        else:
            cleaned_names_count[cleaned] = 1
            mapping[col] = cleaned

    return mapping

def reverse_column_mapping(display_name: str, original_columns: List[str]) -> str:
    """Find the original column name from a cleaned display name"""
    prefixes = ['IQ_', 'D_VALUE_BBG_', 'D_APY_']

    # First check if the display_name exists as-is (no prefix was removed)
    if display_name in original_columns:
        return display_name

    # Handle numbered suffixes (e.g., "TOTAL_REV_2")
    if '_' in display_name and display_name.split('_')[-1].isdigit():
        base_name = '_'.join(display_name.split('_')[:-1])
    else:
        base_name = display_name

    # Then check with each prefix
    for prefix in prefixes:
        original_name = prefix + base_name
        if original_name in original_columns:
            return original_name

    # If not found, return as-is (shouldn't happen in normal usage)
    return display_name

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

def get_target_multiple_options(df: pd.DataFrame) -> List[str]:
    """Get columns that start with D_VALUE_BBG_ for target multiple selection"""
    d_value_bbg_cols = [col for col in df.columns if col.startswith('D_VALUE_BBG_')]
    return sorted(d_value_bbg_cols)

def format_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply consistent formatting rules to dataframe columns for display.
    Also rename columns to remove prefixes.

    Formatting rules:
    - DAY_DATE: year-month-day format
    - IQ_*, D_APY_FCF*, D_APY_LOG_ASSETS*, MARKET_CAP_FISCAL, TEV_FISCAL: ${x:,.2f}M format (negative sign outside $)
    - PRICE_CLOSE_USD: ${x:,.2f} format (negative sign outside $)
    - *_RET, D_APY_GP_REV, D_APY_GR_*: decimal to percentage with 2 decimal points
    - SECTOR_*: 0 or 1, no decimal points
    - Everything else: 4 decimal points
    """
    display_df = df.copy()

    for col in display_df.columns:
        if col in display_df.columns and len(display_df) > 0:
            try:
                # DAY_DATE formatting
                if col == 'DAY_DATE':
                    display_df[col] = pd.to_datetime(display_df[col], errors='coerce').dt.strftime('%Y-%m-%d')
                    display_df[col] = display_df[col].fillna('N/A')

                # Dollar amounts in millions: IQ_*, D_APY_FCF*, D_APY_LOG_ASSETS*, MARKET_CAP_FISCAL, TEV_FISCAL
                elif (col.startswith('IQ_') or
                      col.startswith('D_APY_FCF') or
                      col.startswith('D_APY_LOG_ASSETS') or
                      col in ['MARKET_CAP_FISCAL', 'TEV_FISCAL']):
                    display_df[col] = display_df[col].apply(
                        lambda x: f"-${abs(x):,.2f}M" if pd.notna(x) and x < 0 else f"${x:,.2f}M" if pd.notna(x) else "N/A"
                    )

                # Dollar amounts: PRICE_CLOSE_USD
                elif col == 'PRICE_CLOSE_USD':
                    display_df[col] = display_df[col].apply(
                        lambda x: f"-${abs(x):,.2f}" if pd.notna(x) and x < 0 else f"${x:,.2f}" if pd.notna(x) else "N/A"
                    )

                # Percentage formatting (from decimal): *_RET, D_APY_GP_REV, D_APY_GR_*
                elif (col.endswith('_RET') or
                      col == 'D_APY_GP_REV' or
                      col.startswith('D_APY_GR_')):
                    display_df[col] = display_df[col].apply(
                        lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A"
                    )

                # Integer formatting: SECTOR_*
                elif col.startswith('SECTOR_'):
                    display_df[col] = display_df[col].apply(
                        lambda x: f"{int(x)}" if pd.notna(x) else "N/A"
                    )

                # Everything else: 4 decimal places (for numerical columns)
                else:
                    # Only format if it's a numerical column
                    if pd.api.types.is_numeric_dtype(display_df[col]):
                        display_df[col] = display_df[col].apply(
                            lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
                        )

            except Exception as e:
                # If formatting fails for any column, keep original values
                st.warning(f"Could not format column '{col}': {str(e)}")
                continue

    # Rename columns to remove prefixes, handling duplicates
    column_mapping = create_column_mapping(display_df.columns.tolist())
    display_df = display_df.rename(columns=column_mapping)

    return display_df

def run_peer_analysis(df: pd.DataFrame, target_company: pd.Series, features: List[str],
                     use_most_relevant: bool = True, target_multiple: str = 'D_VALUE_BBG_EBITDA_EV'):
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

    # Use the selected target multiple as the outcome variable for RBP
    if target_multiple in analysis_df.columns:
        y_col = target_multiple
    else:
        # Fallback to other outcome columns if target multiple not available
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

        st.warning(f"Selected target multiple '{target_multiple}' not available. Using '{y_col}' instead.")

    y = analysis_df[y_col].values.reshape(-1, 1)

    # Check outcome variable for issues
    if np.var(y.flatten()) == 0:
        st.error(f"Outcome variable '{y_col}' has zero variance in {target_industry} sector.")
        return None, None

    try:
        # Add debugging info
        st.info(f"Running analysis with {len(analysis_df)} companies in {target_industry} sector, {len(features)} features, target multiple: {y_col}")

        # Run prediction to get ALL relevance scores (analyze all companies in the industry)
        # Use a high threshold to get all companies
        n_all_companies = len(analysis_df)  # Include all companies including target
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

        # Get ALL companies with their scores (including target company)
        all_companies = analysis_df.copy()
        all_companies['RELEVANCE'] = relevance_scores
        all_companies['SIMILARITY'] = similarity_scores
        all_companies['INFORMATIVENESS'] = informativeness_scores

        # Sort by relevance score for consistent ordering (target should be at the top)
        all_companies = all_companies.sort_values('RELEVANCE', ascending=False)

        # Find the target company in the results
        target_company_name = target_company.get('COMPANY_NAME', '')
        target_company_matches = all_companies[all_companies['COMPANY_NAME'] == target_company_name]

        if len(target_company_matches) == 0:
            st.error("Target company not found in analysis results")
            return None, None

        return all_companies, pred_details

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

def get_default_features(df: pd.DataFrame, preferred_features: List[str], target_count: int = 5) -> List[str]:
    """Get default features prioritizing preferred list, then filling with other numerical columns"""
    # Get all available numerical columns
    numerical_cols = get_numerical_columns(df)

    # Find which preferred features are available in the dataset
    available_preferred = [feat for feat in preferred_features if feat in numerical_cols]

    # If we have enough or more preferred features than target_count, use all available preferred features
    if len(available_preferred) >= target_count:
        return available_preferred

    # If we have fewer preferred features than target_count, add other numerical columns to reach target_count
    default_features = available_preferred.copy()

    # Add remaining features from numerical_cols that aren't in preferred list
    remaining_features = [col for col in numerical_cols if col not in preferred_features]

    # Add features until we reach target_count or run out of features
    needed_count = target_count - len(default_features)
    default_features.extend(remaining_features[:needed_count])

    return default_features

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

# Define preferred features for analysis
PREFERRED_FEATURES = [
    "D_APY_FCF_EBITDA",
    "D_APY_EBITDA_GP",
    "D_APY_GP_REV",
    "D_APY_DEBT_MARKET_CAP",
    "VOLATILITY",
    "EARNINGS_VOLATILITY",
    "D_APY_LOG_ASSETS",
    "D_APY_GR_REV",
    "D_APY_GR_EBITDA",
    "D_APY_GR_NI",
    "D_APY_GR_ASSETS",
    "D_APY_GP_ASSETS"
]

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
                current_target_multiple = st.session_state.peer_results.get('target_multiple', 'D_VALUE_BBG_EBITDA_EV')
            else:
                current_features = st.session_state.get('temp_selected_features', [])
                current_target_multiple = st.session_state.get('temp_target_multiple', 'D_VALUE_BBG_EBITDA_EV')

            # Show cleaned feature names for display
            current_features_display = [clean_column_name(feat) for feat in current_features]
            current_target_multiple_display = clean_column_name(current_target_multiple)

            st.sidebar.info(f"✏️ Features locked: {', '.join(current_features_display)}")
            st.sidebar.info(f"🖋️ Target multiple locked: {current_target_multiple_display}")
            selected_features = current_features
            selected_target_multiple = current_target_multiple
        else:
            # Feature selection
            numerical_cols = get_numerical_columns(df)
            default_features = get_default_features(df, PREFERRED_FEATURES, target_count=5)

            # Create display options with cleaned names
            numerical_cols_mapping = create_column_mapping(numerical_cols)
            display_options = list(numerical_cols_mapping.values())
            default_features_display = [clean_column_name(feat) for feat in default_features]

            selected_features_display = st.sidebar.multiselect(
                "Select Features for Analysis",
                options=display_options,
                default=default_features_display,
                help="Choose the financial metrics to use for finding similar companies. Preferred features are pre-selected when available."
            )

            # Convert back to original column names
            selected_features = [reverse_column_mapping(display_name, numerical_cols) for display_name in selected_features_display]

            # Target multiple selection
            target_multiple_options = get_target_multiple_options(df)
            target_multiple_mapping = create_column_mapping(target_multiple_options)
            target_multiple_display_options = list(target_multiple_mapping.values())

            default_target_multiple = 'D_VALUE_BBG_EBITDA_EV' if 'D_VALUE_BBG_EBITDA_EV' in target_multiple_options else (target_multiple_options[0] if target_multiple_options else 'MARKET_CAP_FISCAL')
            default_target_multiple_display = clean_column_name(default_target_multiple)

            selected_target_multiple_display = st.sidebar.selectbox(
                "Select Target Multiple",
                options=target_multiple_display_options if target_multiple_display_options else ['MARKET_CAP_FISCAL'],
                index=target_multiple_display_options.index(default_target_multiple_display) if default_target_multiple_display in target_multiple_display_options else 0,
                help="Choose the target multiple for RBP analysis. This doesn't affect peer group selection but is necessary for the RBP algorithm to function properly."
            )

            # Convert back to original column name
            selected_target_multiple = reverse_column_mapping(selected_target_multiple_display, target_multiple_options if target_multiple_options else ['MARKET_CAP_FISCAL'])

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
            help="Size of peer group to find",
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
                                'temp_selected_company', 'temp_target_company', 'temp_selected_features', 'temp_target_multiple']
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
                    st.session_state.temp_target_multiple = selected_target_multiple

                    # Validate features before running analysis
                    features_valid, feature_issues = validate_features_for_analysis(df, selected_features)

                    if not features_valid:
                        st.error("Issues with selected features:")
                        for issue in feature_issues:
                            st.error(f"• {issue}")
                        st.info("Try selecting different features or a different time period.")
                        # Clear analysis state if validation fails
                        keys_to_remove = ['analysis_started', 'temp_selected_month',
                                        'temp_selected_company', 'temp_target_company', 'temp_selected_features', 'temp_target_multiple']
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
    temp_target_multiple = st.session_state.get('temp_target_multiple', 'D_VALUE_BBG_EBITDA_EV')

    # Load appropriate data
    if temp_month:
        df = load_data(start_date=temp_month, end_date=temp_month)
    else:
        df = load_data()

    if temp_target is not None and temp_features:
        with st.spinner("Running peer analysis on all companies in industry..."):
            # Run analysis on ALL companies in the industry
            all_peer_companies, analysis_details = run_peer_analysis(
                df, temp_target, temp_features, True, temp_target_multiple  # Pass target multiple
            )

            if all_peer_companies is not None:
                st.session_state.peer_results = {
                    'target_company': temp_target,
                    'all_peer_companies': all_peer_companies,  # Store ALL analyzed companies
                    'selected_features': temp_features,
                    'target_multiple': temp_target_multiple,  # Store target multiple
                    'analysis_details': analysis_details,
                    'selected_month': temp_month
                }
                # Clear temporary state now that we have results
                keys_to_remove = ['analysis_started', 'temp_selected_month',
                                'temp_selected_company', 'temp_target_company', 'temp_selected_features', 'temp_target_multiple']
                for key in keys_to_remove:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
            else:
                # Clear analysis state if analysis fails
                keys_to_remove = ['analysis_started', 'temp_selected_month',
                                'temp_selected_company', 'temp_target_company', 'temp_selected_features', 'temp_target_multiple']
                for key in keys_to_remove:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

# Main content area
if selected_company and target_company is not None:
    # Display target company info
    st.subheader(f"🎯 Target Company: {selected_company}")

    # Add warning when no date filter is selected
    if not selected_month or selected_month == "":
        total_rows = len(df)
        st.warning(
            f"**No date filter selected**: Analysis will process {total_rows:,} rows across all time periods. "
            f"This not only takes a long time, but also isn't cross-sectional analysis. "
            f"Consider selecting a specific month-year for better performance and more meaningful comparisons.",
            icon="⚠️"
        )

    col1, col2 = st.columns(2)

    with col1:
        if 'INDUSTRY_SECTOR_W_BIOTECH' in target_company.index:
            st.metric("Industry", target_company['INDUSTRY_SECTOR_W_BIOTECH'])
        if 'PRICE_CLOSE_USD' in target_company.index:
            st.metric("Price (Close)", f"${target_company['PRICE_CLOSE_USD']:,.2f}")

    with col2:
        if 'MARKET_CAP_FISCAL' in target_company.index:
            market_cap = target_company['MARKET_CAP_FISCAL']
            if pd.notna(market_cap):
                if market_cap < 0:
                    st.metric("Market Cap", f"-${abs(market_cap):,.0f}M")
                else:
                    st.metric("Market Cap", f"${market_cap:,.0f}M")
        if 'TEV_FISCAL' in target_company.index:
            tev = target_company['TEV_FISCAL']
            if pd.notna(tev):
                if tev < 0:
                    st.metric("TEV", f"-${abs(tev):,.0f}M")
                else:
                    st.metric("TEV", f"${tev:,.0f}M")

    # Display results if available
    if 'peer_results' in st.session_state:
        results = st.session_state.peer_results

        # Get all companies from analysis (includes target company)
        all_companies = results['all_peer_companies']

        # Find target company in the results
        target_company_name = selected_company
        target_company_matches = all_companies[all_companies['COMPANY_NAME'] == target_company_name]

        if len(target_company_matches) > 0:
            target_company_row = target_company_matches.iloc[0]
            # Remove target company from all_companies for peer filtering
            peer_companies_only = all_companies[all_companies['COMPANY_NAME'] != target_company_name]
        else:
            st.error("Target company not found in analysis results")
            target_company_row = None
            peer_companies_only = all_companies

        # Filter peer companies based on current settings (excluding target)
        if len(peer_companies_only) > 0:
            if use_most_relevant:
                # Get top n_peers (highest relevance scores)
                filtered_peers = peer_companies_only.head(n_peers)
            else:
                # Get bottom n_peers (lowest relevance scores)
                filtered_peers = peer_companies_only.tail(n_peers).iloc[::-1]
        else:
            filtered_peers = peer_companies_only

        selected_features = results['selected_features']
        selected_target_multiple = results.get('target_multiple', 'D_VALUE_BBG_EBITDA_EV')

        st.subheader(f'📝 {"Most" if use_most_relevant else "Least"} Relevant Peer Companies')

        # Create display dataframe with target company included
        display_cols = ['COMPANY_NAME', 'RELEVANCE', 'SIMILARITY', 'INFORMATIVENESS', 'MARKET_CAP_FISCAL', 'TEV_FISCAL']

        # Add selected features to display columns first
        display_cols.extend([col for col in selected_features if col in all_companies.columns])

        # Add the selected target multiple and related columns at the end (rightmost)
        target_multiple_cols = []
        if selected_target_multiple in all_companies.columns:
            target_multiple_cols.append(selected_target_multiple)

            # Add predicted target multiple and residual if available
            pred_col_name = f"{selected_target_multiple}_PRED"
            residual_col_name = f"{selected_target_multiple}_RESIDUAL"

            # Check if prediction results are available and add predicted values and residuals
            if 'analysis_details' in results and results['analysis_details'] is not None:
                analysis_details = results['analysis_details']

                # Get predicted values from RBP results
                if 'y' in analysis_details:
                    predicted_values = analysis_details['y'].flatten()
                    all_companies[pred_col_name] = predicted_values
                    target_multiple_cols.append(pred_col_name)

                    # Calculate residuals (actual - predicted)
                    actual_values = all_companies[selected_target_multiple].values
                    residuals = actual_values - predicted_values
                    all_companies[residual_col_name] = residuals
                    target_multiple_cols.append(residual_col_name)

        # Add target multiple columns to the end of display_cols (rightmost position)
        display_cols.extend(target_multiple_cols)

        # Update the filtered peers and target company row to include new columns
        if len(peer_companies_only) > 0:
            if use_most_relevant:
                # Get top n_peers (highest relevance scores)
                filtered_peers = all_companies[all_companies['COMPANY_NAME'] != target_company_name].head(n_peers)
            else:
                # Get bottom n_peers (lowest relevance scores)
                filtered_peers = all_companies[all_companies['COMPANY_NAME'] != target_company_name].tail(n_peers).iloc[::-1]
        else:
            filtered_peers = all_companies[all_companies['COMPANY_NAME'] != target_company_name]

        # Update target company row to include new columns
        if len(target_company_matches) > 0:
            target_company_row = all_companies[all_companies['COMPANY_NAME'] == target_company_name].iloc[0]

        # Combine target company with peer companies (target first)
        if target_company_row is not None:
            combined_df = pd.concat([target_company_row.to_frame().T, filtered_peers], ignore_index=True)
        else:
            combined_df = filtered_peers

        # Format the dataframe for display using the new formatting function
        display_df = combined_df[display_cols].copy()

        score_columns = ['RELEVANCE', 'SIMILARITY', 'INFORMATIVENESS']
        for col in score_columns:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: round(float(x), 4))

        # Apply comprehensive formatting (this will also clean column names)
        display_df = format_dataframe_for_display(display_df)

        # Check for duplicate column names and warn if found
        if len(display_df.columns) != len(set(display_df.columns)):
            duplicate_cols = [col for col in display_df.columns if list(display_df.columns).count(col) > 1]
            st.warning(f"Warning: Duplicate column names detected after cleaning: {set(duplicate_cols)}. This may cause display issues.")

        # Style the dataframe to highlight the target company (first row)
        def highlight_target(row):
            # Target company is always the first row (index 0) when present
            if row.name == 0 and target_company_row is not None:
                return [f'background-color: {RED}; color: white; font-weight: bold'] * len(row)
            else:
                return [''] * len(row)

        # Check if columns are unique before applying styling
        if len(display_df.columns) == len(set(display_df.columns)):
            styled_df = display_df.style.apply(highlight_target, axis=1)
            st.dataframe(styled_df, use_container_width=True)
        else:
            # If we have duplicate columns, just display without styling
            st.warning("Displaying table without highlighting due to duplicate column names.")
            st.dataframe(display_df, use_container_width=True)

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

            # Add peer companies (use filtered_peers instead of peer_companies)
            for idx, (_, peer) in enumerate(filtered_peers.iterrows()):
                peer_row = {'Company': peer['COMPANY_NAME'], 'Type': f'Peer {idx+1}'}
                for feature in selected_features:
                    peer_row[feature] = peer[feature] if feature in peer.index else np.nan
                comparison_data.append(peer_row)

            comparison_df = pd.DataFrame(comparison_data)

            # Create display options for feature selection with cleaned names
            feature_display_options = [clean_column_name(feat) for feat in selected_features]
            feature_mapping = {clean_column_name(feat): feat for feat in selected_features}

            feature_to_plot_display = st.selectbox(
                "Select feature to visualize:",
                options=feature_display_options,
                key="feature_plot"
            )

            # Convert back to original column name
            feature_to_plot = feature_mapping[feature_to_plot_display]

            if feature_to_plot:
                chart_data = comparison_df[['Company', 'Type', feature_to_plot]].copy()
                chart_data = chart_data.dropna(subset=[feature_to_plot])

                if len(chart_data) > 0:
                    # Create colors for bars - red for target company, blue for peers
                    colors = [RED if row['Type'] == 'Target' else '#1f77b4' for _, row in chart_data.iterrows()]

                    # Create bar chart with custom colors
                    fig = go.Figure(data=[
                        go.Bar(
                            x=chart_data['Company'],
                            y=chart_data[feature_to_plot],
                            marker_color=colors,
                            text=chart_data[feature_to_plot].round(4),
                            textposition='auto',
                            hovertemplate='<extra></extra>'  # Remove tooltip
                        )
                    ])

                    fig.update_layout(
                        title="",
                        xaxis_title="",
                        yaxis_title=feature_to_plot_display,  # Use cleaned name for chart label
                        showlegend=False,
                        height=400
                    )

                    # Rotate x-axis labels for better readability
                    fig.update_xaxes(tickangle=45)

                    st.plotly_chart(fig, use_container_width=True)

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
        # Define preferred columns for display
        preferred_display_cols = [
            "YEAR_MONTH",
            "COMPANY_NAME",
            "INDUSTRY_SECTOR_W_BIOTECH",
            "PRICE_CLOSE_USD",
            "IQ_TOTAL_REV",
            "D_APY_GR_REV",
            "D_APY_GP_REV",
            "IQ_EBITDA",
            "IQ_NI",
            "IQ_TOTAL_ASSETS",
            "IQ_TOTAL_DEBT",
            "MARKET_CAP_FISCAL",
            "TEV_FISCAL",
            "D_VALUE_BBG_EARN_PRICE",
            "D_VALUE_BBG_EBITDA_EV"
        ]

        # Get available preferred columns from the dataset
        available_preferred_display = [col for col in preferred_display_cols if col in df.columns]

        # If we don't have enough preferred columns, add some numerical columns as fallback
        if len(available_preferred_display) < 5:
            numerical_sample = get_numerical_columns(df)[:5]  # First 5 numerical columns
            available_preferred_display.extend([col for col in numerical_sample if col not in available_preferred_display])

        # Create display options with cleaned names
        all_columns_mapping = create_column_mapping(df.columns.tolist())
        available_preferred_display_cleaned = [clean_column_name(col) for col in available_preferred_display]
        all_display_options = list(all_columns_mapping.values())

        selected_display_cols_cleaned = st.multiselect(
            "Select columns to display",
            options=all_display_options,
            default=available_preferred_display_cleaned,
            help="Choose which columns to show. Fewer columns = faster loading."
        )

        # Convert back to original column names
        selected_display_cols = [reverse_column_mapping(display_name, df.columns.tolist()) for display_name in selected_display_cols_cleaned]

    if selected_display_cols:
        # Add sorting controls
        col_sort1, col_sort2, col_sort3 = st.columns(3)

        with col_sort1:
            # Convert selected display columns to cleaned names for the sort dropdown
            selected_display_cols_cleaned_for_sort = [clean_column_name(col) for col in selected_display_cols]

            sort_column_display = st.selectbox(
                "Sort by column",
                options=["None"] + selected_display_cols_cleaned_for_sort,
                index=0,
                help="Select a column to sort the entire dataset"
            )

            # Convert back to original column name
            sort_column = reverse_column_mapping(sort_column_display, selected_display_cols) if sort_column_display != "None" else "None"

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
                sort_column_display_for_info = clean_column_name(sort_column)
                st.info(f"Dataset sorted by '{sort_column_display_for_info}' ({'ascending' if sort_ascending else 'descending'})")
            except Exception as e:
                st.warning(f"Could not sort by '{sort_column}': {str(e)}")

        # Now take the sample from the sorted/filtered data
        sample_df = filtered_df[selected_display_cols].head(max_rows)

        st.info(f"Showing {len(sample_df)} rows out of {len(filtered_df)} total rows with {len(selected_display_cols)} columns")

        # Apply formatting to the sample data as well (this will also clean column names)
        formatted_sample_df = format_dataframe_for_display(sample_df)

        # Display the data (remove height parameter to allow full interaction)
        st.dataframe(
            formatted_sample_df,
            use_container_width=True,
            height=400
        )

    else:
        st.warning("Please select at least one column to display")

# Footer
st.markdown("---")
st.markdown("*Designed by Aarna Pal-Yadav.*")