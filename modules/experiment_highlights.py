import json
import os
try:
    from dotenv import load_dotenv
    # Loads .env from project root (recommended location)
    load_dotenv()
except ImportError:
    pass  # If dotenv is not installed, skip (user must set env manually)
import pandas as pd
import numpy as np
import sys
import os
import builtins
import requests
import re
import ast
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances_argmin_min
from sklearn.decomposition import PCA
from prince import MCA
from ._correlation_utils import (
    compute_correlations_with_components,
    compute_eta_squared_with_components,
    detect_variable_types
)
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import f_oneway
import shap
from ._correlation_utils import (
    detect_variable_types,
    compute_relationship_measure,
    compute_relationship_matrix
)
import warnings
import json

QUIET = os.environ.get("QUIET_MODE", "1") == "0"

if QUIET:
    builtins.print = lambda *args, **kwargs: None
def convert_runs_data_to_csv(runs):
    """Convert in-memory runs list/dict to workflows DataFrame.

    This is the core implementation used by both the file-based helper
    and the gRPC entrypoint where the client sends the JSON payload
    directly instead of a file path.
    """

    # If a dict with a top-level key is provided, try to unwrap
    if isinstance(runs, dict):
        # Common patterns: {"runs": [...]} or similar
        for key in ("runs", "data", "items"):
            if key in runs and isinstance(runs[key], list):
                runs = runs[key]
                break

    # Ensure we are working with a list of runs
    if not isinstance(runs, list):
        raise ValueError("Expected 'runs' to be a list of run objects")

    # Collect all unique parameter and metric names
    param_names = set()
    metric_names = set()

    # Process each run
    workflow_list = []

    for run in runs:
        # Initialize workflow dictionary with ID
        run_id = run.get('id') if isinstance(run, dict) else None
        workflow = {'workflowId': run_id}

        # Extract parameters
        for param in run.get('params', []):
            param_name = param.get('name')
            param_value = param.get('value')

            if param_name is None:
                continue

            # Convert parameter name format: fairness_method -> fairness method
            param_name_formatted = param_name.replace('_', ' ')
            param_names.add(param_name_formatted)

            # Try to convert numeric strings to numbers
            try:
                if isinstance(param_value, str) and '.' in param_value:
                    param_value = float(param_value)
                else:
                    param_value = int(param_value)
            except (ValueError, TypeError):
                # Keep original value if conversion fails
                pass

            workflow[param_name_formatted] = param_value

        # Extract metrics
        metrics = run.get('metrics', [])

        # Some runs may have missing or placeholder metrics (e.g. None, "N/A").
        # In that case, skip the entire run to avoid downstream errors.
        if metrics is None or isinstance(metrics, str):
            # Do not include this run in workflow_list
            continue

        for metric in metrics:
            metric_name = metric.get('name')
            metric_value = metric.get('value')
            if metric_name is None:
                continue

            # Exclude rating metric from features
            if str(metric_name).lower() == 'rating':
                continue

            metric_names.add(metric_name)
            workflow[metric_name] = metric_value

        # Add status if available
        if isinstance(run, dict) and 'status' in run:
            workflow['status'] = run['status']

        workflow_list.append(workflow)

    # Create DataFrame
    df = pd.DataFrame(workflow_list)
    # Some datasets may not include status
    df.drop(["status"], axis=1, inplace=True, errors='ignore')
    df.drop(["rating"], axis=1, inplace=True, errors='ignore')


    return df, param_names, metric_names


def convert_runs_json_to_csv(json_file):
    """Convert runs.json on disk to workflows DataFrame.

    This is a thin wrapper around convert_runs_data_to_csv used by
    the original CLI-style script. For gRPC usage you should prefer
    convert_runs_data_to_csv with in-memory data.
    """

    with open(json_file, 'r') as f:
        runs = json.load(f)

    return convert_runs_data_to_csv(runs)

class ClusteringPipeline:
    """
    Modular pipeline for ML workflow clustering.
    Allows enabling/disabling and reordering steps for testing.
    Steps can be optional and dependent steps handle missing results gracefully.
    """
    
    def __init__(self):
        """Initialize pipeline state."""
        self.steps = []
        self.results = {}
        self.enabled_steps = set()
        self.skip_missing_deps = True  # Enable graceful handling of missing dependencies
    
    def add_step(self, name, func, enabled=True):
        """
        Add a step to the pipeline.
        
        Args:
            name: Unique identifier for the step
            func: Callable that performs the step logic
            enabled: Whether this step is enabled by default
        """
        self.steps.append({'name': name, 'func': func})
        if enabled:
            self.enabled_steps.add(name)
    
    def enable_step(self, name):
        """Enable a specific step."""
        if name in [s['name'] for s in self.steps]:
            self.enabled_steps.add(name)
        else:
            print(f"✗ Step not found: {name}")
    
    def disable_step(self, name):
        """Disable a specific step."""
        self.enabled_steps.discard(name)
        print(f"✓ Disabled: {name}")
    
    def set_steps(self, step_names):
        """Enable only specific steps (disable all others)."""
        self.enabled_steps = set(step_names)
    
    def list_steps(self):
        """Print all available steps and their status."""
       
        for step in self.steps:
            status = "✓ ENABLED" if step['name'] in self.enabled_steps else "✗ DISABLED"
    
    def get_result(self, step_name, key=None, default=None):
        """
        Safely retrieve results from a previous step.
        
        Args:
            step_name: Name of the step to get results from
            key: Optional key within the step's result dict
            default: Default value if step or key not found
        
        Returns:
            Result value or default if not found
        """
        if step_name not in self.results:
            return default
        
        result = self.results[step_name]
        if key is None:
            return result
        
        if isinstance(result, dict):
            return result.get(key, default)
        
        return default
    
    def run(self, **kwargs):
        """
        Execute the pipeline, running only enabled steps in order.
        
        Args:
            **kwargs: Arguments passed to each step function
        
        Returns:
            Dictionary of results from all executed steps
        """
       
        
        for step in self.steps:
            step_name = step['name']
            
            if step_name not in self.enabled_steps:
                print(f"\n⊘ Skipping: {step_name}")
                continue
            
            
            try:
                # Pass both general kwargs and previously stored results
                result = step['func'](self.results, self, **kwargs)
                self.results[step_name] = result
                print(f"✓ Completed: {step_name}")
            except KeyError as e:
                print(f"✗ Error in {step_name}: Missing dependency {e}")
                if self.skip_missing_deps:
                    print(f"  Skipping this step due to missing results from a disabled step.")
                    continue
                else:
                    raise
            except Exception as e:
                print(f"✗ Error in {step_name}: {e}")
                raise
        
        return self.results

def load_and_preprocess_data(data_folder='data', params_file=None, metrics_file=None):
    """Load workflow data and separate hyperparameters from metrics."""
    
    # If params_file and metrics_file not provided, use defaults from data_folder
    if params_file is None:
        params_file = os.path.join(data_folder, 'parameter_names.txt')
    if metrics_file is None:
        metrics_file = os.path.join(data_folder, 'metric_names.txt')
    
    # Load workflows CSV from folder
    filepath = os.path.join(data_folder, 'workflows.csv')
    df = pd.read_csv(filepath, on_bad_lines='skip')

    # Load hyperparameters from file if provided
    if os.path.exists(params_file):
        with open(params_file, 'r') as f:
            hyperparams = [line.strip() for line in f.readlines() if line.strip()]
    else:
        # Define hyperparameters and metrics (fallback)
        hyperparams = ['criterion', 'fairness method', 'random state', 
                       'max depth', 'normalization', 'n estimators']

    # Load metrics from file if provided
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics = [line.strip() for line in f.readlines() if line.strip()]
    else:
        # Exclude system metrics and identifiers
        exclude_cols = ['workflowId'] + hyperparams 
        metrics = [col for col in df.columns if col not in exclude_cols]


    return df, hyperparams, metrics

def step_load_data(results, pipeline, **kwargs):
    """Step 1: Load and preprocess data."""
    # Preferred path: accept precomputed objects directly (no folders/files needed)
    df_converted = kwargs.get('df_converted')
    param_names = kwargs.get('param_names')
    metric_names = kwargs.get('metric_names')

    if df_converted is not None and param_names is not None and metric_names is not None:
        hyperparam_cols = sorted(list(param_names))
        metric_cols = sorted(list(metric_names))
        return {
            'df': df_converted,
            'hyperparam_cols': hyperparam_cols,
            'metric_cols': metric_cols,

        }
    else:
        print("⚠️  Warning: Missing precomputed data. Attempting to load from default files...")

def filter_low_and_high_variance_metrics(df, metric_cols, low_cv_threshold=0.05, high_cv_threshold=1.5):
    """
    Filter out metrics with low variance using Coefficient of Variation (CV).
    
    CV = std / mean, which is scale-independent and works across metrics with different ranges.
    Metrics with CV below the threshold are considered uninformative for clustering.
    
    Args:
        df: Input DataFrame
        metric_cols: List of metric column names
        cv_threshold: Coefficient of Variation threshold. Metrics with CV below 
                     this are filtered out (default: 0.1 = 10%)
    
    Returns:
        filtered_metrics: List of metric column names after low variance filter
        filtered_info: Dictionary with filtering details
    """
    metric_variances = []
    
    for col in metric_cols:
        # Calculate variance statistics only for numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            mean_val = df[col].mean()
            std_val = df[col].std()
            variance = df[col].var()
            
            # Coefficient of Variation: std / mean (scale-independent)
            # Handle edge cases: if mean is 0 or very close to 0, use variance as fallback
            if abs(mean_val) > 1e-10:
                cv = std_val / abs(mean_val)
            else:
                cv = variance  # Fallback for metrics with mean near 0
            
            metric_variances.append({
                'metric': col,
                'mean': mean_val,
                'std': std_val,
                'variance': variance,
                'cv': cv,
                'min': df[col].min(),
                'max': df[col].max(),
                'range': df[col].max() - df[col].min()
            })
        else:
            # Non-numeric columns are kept (will be handled as categorical)
            metric_variances.append({
                'metric': col,
                'mean': np.nan,
                'std': np.nan,
                'variance': np.inf,
                'cv': np.inf,  # Non-numeric columns are always kept
                'min': np.nan,
                'max': np.nan,
                'range': np.nan
            })
    
    variance_df = pd.DataFrame(metric_variances)
    
    # Identify low variance metrics based on CV threshold
    low_variance_metrics = variance_df[(variance_df['cv'] < low_cv_threshold) & (variance_df['cv'] != np.inf)]['metric'].tolist()
    high_variance_metrics = variance_df[(variance_df['cv'] > high_cv_threshold) & (variance_df['cv'] != np.inf)]['metric'].tolist()
    filtered_metrics = [col for col in metric_cols if col not in low_variance_metrics]
    filtered_metrics = [col for col in filtered_metrics if col not in high_variance_metrics]
    
    # Print filtering results
   
    
    if low_variance_metrics:
        print(f"\nRemoved metrics (low coefficient of variation):")
        for metric in low_variance_metrics:
            var_info = variance_df[variance_df['metric'] == metric].iloc[0]
            print(f"  - {metric:30s} (CV: {var_info['cv']:.4f}, mean: {var_info['mean']:10.4f}, std: {var_info['std']:10.6f}, range: [{var_info['min']:.4f}, {var_info['max']:.4f}])")
    
    if high_variance_metrics:
        print(f"\nRemoved metrics (high coefficient of variation):")
        for metric in high_variance_metrics:
            var_info = variance_df[variance_df['metric'] == metric].iloc[0]
            print(f"  - {metric:30s} (CV: {var_info['cv']:.4f}, mean: {var_info['mean']:10.4f}, std: {var_info['std']:10.6f}, range: [{var_info['min']:.4f}, {var_info['max']:.4f}])")
    
    print(f"\nRetained metrics:")
    for metric in filtered_metrics:
        var_info = variance_df[variance_df['metric'] == metric].iloc[0]
        if pd.api.types.is_numeric_dtype(df[metric]):
            print(f"  - {metric:30s} (CV: {var_info['cv']:.4f}, mean: {var_info['mean']:10.4f}, std: {var_info['std']:10.6f}, range: [{var_info['min']:.4f}, {var_info['max']:.4f}])")
        else:
            print(f"  - {metric:30s} (categorical)")
    
    filtered_info = {
        'total_original': len(metric_cols),
        'total_removed': len(low_variance_metrics),
        'total_retained': len(filtered_metrics),
        'removed_metrics': low_variance_metrics,
        'low_cv_threshold': low_cv_threshold,
        'high_cv_threshold': high_cv_threshold,
        'variance_details': variance_df.to_dict('records')
    }
    
    return filtered_metrics, filtered_info


def step_filter_low_variance_metrics(results, pipeline, **kwargs):
    """Step 1.5a: Filter out metrics with low coefficient of variation."""
    load_result = pipeline.get_result('step_load_data', default={})
    df = load_result.get('df')
    metric_cols = load_result.get('metric_cols')
    
    if df is None or metric_cols is None:
        raise KeyError("step_load_data: 'df' or 'metric_cols' not available. Load data first.")
    
    low_cv_threshold = kwargs.get('low_cv_threshold', 0.05)
    
 
    
    filtered_metrics, filter_info = filter_low_and_high_variance_metrics(
        df[metric_cols], metric_cols, 
        low_cv_threshold=low_cv_threshold, 
        high_cv_threshold=float('inf')  # Don't filter high in this step
    )
    
    return {
        'metric_cols': filtered_metrics,
        'filter_info': filter_info,
        'removed_low_variance': filter_info.get('removed_metrics', [])
    }


def step_filter_high_variance_metrics(results, pipeline, **kwargs):
    """Step 1.5b: Filter out metrics with high coefficient of variation."""
    # Get metric columns from low variance filtering step
    low_filter_result = pipeline.get_result('step_filter_low_variance_metrics', default=None)
    
    if low_filter_result is None:
        # Fallback to raw metrics from load step
        load_result = pipeline.get_result('step_load_data', default={})
        metric_cols = load_result.get('metric_cols', [])
       
    else:
        metric_cols = low_filter_result.get('metric_cols', [])
       
    
    load_result = pipeline.get_result('step_load_data', default={})
    df = load_result.get('df')
    
    if df is None or not metric_cols:
        raise KeyError("step_load_data or step_filter_low_variance_metrics: Data not available.")
    
    high_cv_threshold = kwargs.get('high_cv_threshold', 1.5)
    
   
    
    filtered_metrics, filter_info = filter_low_and_high_variance_metrics(
        df[metric_cols], metric_cols,
        low_cv_threshold=0.0,  # Don't filter low in this step (already done)
        high_cv_threshold=high_cv_threshold
    )

   
    
    return {
        'metric_cols': filtered_metrics,
        'filter_info': filter_info,
        'removed_high_variance': filter_info.get('removed_metrics', [])
    }

def apply_elbow_method_pca(X_scaled, max_components=None, variance_threshold=0.95):
    """
    Apply elbow method to determine optimal number of PCA components
    for numerical variables (SPCA simulation using standard PCA).
    
    Args:
        X_scaled: Scaled numerical data matrix
        max_components: Maximum number of components to test (default: min(n_features-1, n_samples-1))
        variance_threshold: Stop when cumulative variance exceeds this threshold
    
    Returns:
        optimal_n_components: Optimal number of components based on elbow
        pca_fit: Fitted PCA object
        explained_variance_ratio: Array of explained variance ratios
        cumulative_variance: Array of cumulative explained variance
    """
    if max_components is None:
        max_components = min(X_scaled.shape[0] - 1, X_scaled.shape[1])
    
    pca = PCA(n_components=max_components)
    pca.fit(X_scaled)
    
    # Calculate cumulative variance explained
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Find optimal number of components based on variance threshold
    optimal_n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    # If threshold not reached, use elbow method (largest decrease in variance)
    if cumulative_variance[-1] < variance_threshold:
        variance_diffs = np.diff(pca.explained_variance_ratio_)
        # Find elbow: steepest drop in variance explained
        elbow_point = np.argmax(np.diff(variance_diffs))
        optimal_n_components = max(elbow_point + 2, 2)  # At least 2 components
    
    
    
    return optimal_n_components, pca, pca.explained_variance_ratio_, cumulative_variance


def reduce_dimensions(df, metric_cols, pca_variance_threshold, mca_inertia_threshold,
                      corr_threshold, eta_threshold):
    """
    Apply combined dimensionality reduction using SPCA for numerical variables
    and MCA for categorical variables with automatic component selection via elbow method.
    
    This function:
    1. Separates numerical and categorical variables
    2. Standardizes numerical variables and applies PCA (SPCA simulation)
    3. Applies MCA to categorical variables
    4. Uses elbow method to automatically determine optimal components
    5. Identifies relevant original variables based on correlation/eta-squared thresholds
    6. Names derived components based on most correlated original variables
    
    Args:
        df: Input DataFrame
        metric_cols: List of metric column names to include in dimensionality reduction (already filtered)
        pca_variance_threshold: Target cumulative variance for PCA
        mca_inertia_threshold: Target cumulative inertia for MCA
        corr_threshold: Correlation threshold for naming PCA components
        eta_threshold: Eta-squared threshold for naming MCA components
    
    Returns:
        reduced_df: DataFrame with derived components
        reduction_info: Dictionary with reduction details and component names
        component_correlations: Dictionary of correlations/eta-squared for each component
    """
    
    # Select only metric columns for reduction
    df_metrics = df[metric_cols].copy()
    
    # Identify numerical and categorical variables using unified approach
    var_types = detect_variable_types(df_metrics)
    numerical_cols = [col for col, vtype in var_types.items() if vtype == 'continuous']
    categorical_cols = [col for col, vtype in var_types.items() if vtype == 'categorical']
    
   
    
    # Initialize dataframe for reduced components
    reduced_data = pd.DataFrame(index=df.index)
    component_names = []
    component_correlations = {}
    
    # ===== Process Numerical Variables with PCA =====
    if numerical_cols:
        X_numerical = df_metrics[numerical_cols].dropna(axis=0, how='all')
        if not QUIET:
            print(f"Applying PCA to numerical variables: {X_numerical}")
        
        # Note: Low and high variance filtering is now done in separate pipeline steps
        # (step_filter_low_variance_metrics and step_filter_high_variance_metrics)
        # metric_cols passed here are already filtered
        
        # Standardize numerical variables
        scaler = StandardScaler()
        X_numerical_scaled = scaler.fit_transform(X_numerical)
        X_numerical_scaled_df = pd.DataFrame(X_numerical_scaled, columns=numerical_cols, index=X_numerical.index)
        
        # Apply elbow method for PCA with target variance threshold
        optimal_pca_components, pca_fit, explained_var, cumulative_var = apply_elbow_method_pca(
            X_numerical_scaled, 
            max_components=len(numerical_cols),
            variance_threshold=pca_variance_threshold
        )
        
        # Refit PCA with optimal number of components determined by elbow method
        pca_final = PCA(n_components=optimal_pca_components)
        pca_components = pca_final.fit_transform(X_numerical_scaled)
        
        # Create component names based on correlations
        pca_component_correlations = compute_correlations_with_components(
            X_numerical_scaled_df, pca_components, 
            [f'PC_{i+1}' for i in range(optimal_pca_components)],
            correlation_threshold=corr_threshold
        )

        # Generate meaningful names for PCA components
        for i in range(optimal_pca_components):
            relevant_vars = pca_component_correlations[f'PC_{i+1}']
            component_name = f"PC_{i+1}"
            # if relevant_vars:
                # Create a set of all relevant variable names
                # component_name = f"PC_{i+1}"
                # var_names = {var['variable'] for var in relevant_vars}
                # component_name = f"PCA_{i+1}_{{{','.join(sorted(var_names))}}}"
            # else:
            #     continue
            
            component_names.append(component_name)
            reduced_data[component_name] = pca_components[:, i]
            component_correlations[component_name] = relevant_vars
        
    
    
    # ===== Process Categorical Variables with MCA =====
    if categorical_cols:
        X_categorical = df_metrics[categorical_cols].copy()
        
        # Handle missing values in categorical data
        X_categorical = X_categorical.fillna('Unknown')
        # Apply elbow method for MCA with target inertia threshold
        optimal_mca_components, mca_fit, inertia, cumulative_inertia = apply_elbow_method_mca(
            X_categorical,
            max_components=len(categorical_cols),
            inertia_threshold=mca_inertia_threshold
        )
        
        # Refit MCA with optimal number of components determined by elbow method
        mca_final = MCA(n_components=optimal_mca_components, random_state=42)
        mca_components = mca_final.fit_transform(X_categorical).values
        
        # Create component names based on eta-squared
        mca_component_eta = compute_eta_squared_with_components(
            X_categorical, mca_components,
            [f'MCA_{i+1}' for i in range(optimal_mca_components)],
            eta_threshold=eta_threshold
        )
        
        # Generate meaningful names for MCA components
        for i in range(optimal_mca_components):
            relevant_cats = mca_component_eta[f'MCA_{i+1}']
            component_name = f'MCA_{i+1}'
            # if relevant_cats:
            #     # Create a set of all relevant categorical variable names
            #     cat_names = {cat['variable'] for cat in relevant_cats}
            #     component_name = f"MCA_{i+1}_{{{','.join(sorted(cat_names))}}}"
            # else:
            #     component_name = f'MCA_{i+1}'
            
            component_names.append(component_name)
            reduced_data[component_name] = mca_components[:, i]
            component_correlations[component_name] = relevant_cats
        
    
    
    # Summary statistics

    for comp_name in component_names:
        print(f"  - {comp_name}")

    # Neither the PCA nor the MCA branch ran (no numerical_cols, no categorical_cols),
    # so `reduced_data` has zero columns. Left unchecked, that 0-column DataFrame
    # reaches KMeans.fit_predict() several steps later and fails with a cryptic
    # numpy/sklearn error ("at least one array or dtype is required") that gives no
    # indication of the real cause. Fail here instead, with a message that actually
    # explains what's wrong: no metric/param column had enough variation to be
    # classified as numerical or categorical after filtering.
    if reduced_data.shape[1] == 0:
        raise ValueError(
            "No usable metrics or parameters remain for clustering: "
            f"{len(metric_cols)} column(s) were provided ({', '.join(metric_cols) or 'none'}), "
            "but none had enough variation to be classified as numerical or categorical — "
            "they may be constant, entirely missing, or filtered out by the low/high "
            "variance thresholds. This experiment needs at least one metric or "
            "hyperparameter that varies across its completed runs to compute highlights."
        )

    # Create reduction info dictionary
    reduction_info = {
        'n_original_variables': len(metric_cols),
        'n_derived_variables': len(component_names),
        'n_numerical_original': len(numerical_cols),
        'n_categorical_original': len(categorical_cols),
        'n_pca_components': len([c for c in component_names if 'PCA' in c]),
        'n_mca_components': len([c for c in component_names if 'MCA' in c]),
        'component_names': component_names,
        'numerical_variables': numerical_cols,
        'categorical_variables': categorical_cols,
        'kept_metric_cols': numerical_cols + categorical_cols  # Track which metrics were actually kept after filtering
    }
    
    return reduced_data, reduction_info, component_correlations


def step_dimensionality_reduction(results, pipeline, **kwargs):
    """Step 2: Apply dimensionality reduction."""
    # Get filtered metrics from high variance filtering step
    high_filter_result = pipeline.get_result('step_filter_high_variance_metrics', default=None)
    
    if high_filter_result is None:
        # Fallback to low variance filtering
        low_filter_result = pipeline.get_result('step_filter_low_variance_metrics', default=None)
        if low_filter_result is None:
            # Fallback to raw metrics from load step
            load_result = pipeline.get_result('step_load_data', default={})
            metric_cols = load_result.get('metric_cols')
        else:
            metric_cols = low_filter_result.get('metric_cols')
    else:
        metric_cols = high_filter_result.get('metric_cols')

    load_result = pipeline.get_result('step_load_data', default={})
    df = load_result.get('df')

    if df is None or metric_cols is None:
        raise KeyError("step_load_data: 'df' or 'metric_cols' not available. Load data first.")
    
    pca_variance_threshold = kwargs.get('pca_variance_threshold', 0.95)
    mca_inertia_threshold = kwargs.get('mca_inertia_threshold', 0.95)
    corr_threshold = kwargs.get('corr_threshold', 0.5)
    eta_threshold = kwargs.get('eta_threshold', 0.5)
    
   
    
    reduced_data, reduction_info, component_correlations = reduce_dimensions(
        df, 
        metric_cols=metric_cols,
        pca_variance_threshold=pca_variance_threshold,  
        mca_inertia_threshold=mca_inertia_threshold,   
        corr_threshold=corr_threshold,
        eta_threshold=eta_threshold
    )
    
    return {
        'reduced_data': reduced_data,
        'reduction_info': reduction_info,
        'component_correlations': component_correlations
    }


def step_save_correlations(results, pipeline, **kwargs):
    """Step 2.5: Save component correlation information. (Optional - skipped if dim reduction is skipped)"""
    # Gracefully handle missing dimensionality reduction
    dim_result = pipeline.get_result('step_dimensionality_reduction', default=None)
    if dim_result is None:
        print("Skipping correlation save: dimensionality reduction not performed.")
        return {'correlation_summary': []}
    
    component_correlations = dim_result.get('component_correlations', {})
    
    # Get data_folder from load_data step
    load_result = pipeline.get_result('step_load_data', default={})
    # data_folder = load_result.get('data_folder', 'data')
    # print(f"Data folder for saving correlations: {data_folder}")
    correlation_summary = []
    for comp_name, correlations in component_correlations.items():
        for corr_info in correlations:
            if 'correlation' in corr_info:
                correlation_summary.append({
                    'component': comp_name,
                    'variable': corr_info['variable'],
                    'correlation': corr_info['correlation']
                })
            elif 'eta_squared' in corr_info:
                correlation_summary.append({
                    'component': comp_name,
                    'variable': corr_info['variable'],
                    'eta_squared': corr_info['eta_squared']
                })
    
    if correlation_summary:
        corr_df = pd.DataFrame(correlation_summary)
        # corr_file = os.path.join(data_folder, 'workflows_component_correlations.csv')
        # corr_df.to_csv(corr_file, index=False)
        # print(f"Saved component correlations to: {corr_file}")
    
    return {'correlation_summary': correlation_summary}


def step_prepare_clustering_data(results, pipeline, **kwargs):
    """Step 3: Prepare and scale data for clustering."""
    # Try to get reduced data from dimensionality reduction
    dim_result = pipeline.get_result('step_dimensionality_reduction', default=None)
    
    if dim_result is None:
        # Fallback: use raw data from load step
        print("Dimensionality reduction not performed. Using raw metric data.")
        load_result = pipeline.get_result('step_load_data', default={})
        df = load_result.get('df')
        metric_cols = load_result.get('metric_cols')
        
        if df is None or metric_cols is None:
            raise KeyError("step_load_data: 'df' or 'metric_cols' not available. Load data first.")
        
        reduced_data = df[metric_cols].copy()
    else:
        reduced_data = dim_result.get('reduced_data')
    
   
    
    X_scaled = reduced_data.values
    
    return {'X_scaled': X_scaled, 'reduced_data': reduced_data}

def step_find_optimal_clusters(results, pipeline, **kwargs):
    """Step 4: Find optimal number of clusters using silhouette score."""
    cluster_result = pipeline.get_result('step_prepare_clustering_data', default=None)
    if cluster_result is None:
        raise KeyError("step_prepare_clustering_data: Data preparation required before finding optimal clusters.")
    
    X_scaled = cluster_result.get('X_scaled')
    reduced_data = cluster_result.get('reduced_data')
    
    min_k = kwargs.get('min_k', 3)
    max_k = kwargs.get('max_k', 9)

    n_samples = len(reduced_data)
    # silhouette_score requires 2 <= n_labels <= n_samples - 1, so k can never
    # reach n_samples itself (that's every point in its own cluster, which
    # isn't scoreable). Cap the search range to the data actually available
    # instead of letting KMeans/silhouette_score crash with an opaque
    # "Number of labels is N. Valid values are 2 to n_samples - 1" error.
    effective_max_k = min(max_k, n_samples)
    if effective_max_k <= min_k:
        raise ValueError(
            f"Not enough completed runs to search for clusters: {n_samples} run(s) available, "
            f"but at least {min_k + 1} are needed to try cluster counts from {min_k} up to "
            "n_samples - 1."
        )

    print("\nFinding optimal number of clusters...")
    silhouette_scores = []

    for k in range(min_k, effective_max_k):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(reduced_data)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
        print(f"  k={k}, Silhouette Score: {score:.3f}")

    optimal_k = list(range(min_k, effective_max_k))[np.argmax(silhouette_scores)]
    print(f"\nOptimal number of clusters: {optimal_k}")

    return {'optimal_k': optimal_k, 'silhouette_scores': silhouette_scores}

def cluster_workflows(X_scaled, n_clusters=4, random_state=42):
    """
    Cluster workflows based on performance metrics using KMeans,
    then find medoids (actual data points closest to centroids).
    """
    # Perform KMeans clustering
    print(f"\nClustering into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Find medoids (actual workflows closest to centroids)
    medoid_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X_scaled)
    print(f"Clustering complete. Silhouette score: {compute_silhouette(X_scaled, cluster_labels):.3f}")

    return cluster_labels, medoid_indices

def compute_silhouette(X, labels):
    """Compute silhouette score for clustering quality."""
    try:
        return silhouette_score(X, labels)
    except:
        return 0.0


def step_perform_clustering(results, pipeline, **kwargs):
    """Step 5: Perform clustering with optimal k or use provided k."""
    cluster_result = pipeline.get_result('step_prepare_clustering_data', default=None)
    if cluster_result is None:
        raise KeyError("step_prepare_clustering_data: Data preparation required before clustering.")
    
    X_scaled = cluster_result.get('X_scaled')
    
    # Try to get optimal_k from previous step, otherwise use kwargs or default
    optimal_k = kwargs.get('n_clusters', None)
    if optimal_k is None:
        opt_result = pipeline.get_result('step_find_optimal_clusters', default=None)
        if opt_result is not None:
            optimal_k = opt_result.get('optimal_k')
        else:
            optimal_k = kwargs.get('n_clusters', 4)  # Default to 4 if not found
    
    cluster_labels, medoid_indices = cluster_workflows(
        X_scaled, n_clusters=optimal_k
    )
    
    return {
        'cluster_labels': cluster_labels,
        'medoid_indices': medoid_indices,
        'n_clusters': optimal_k
    }


def identify_small_clusters(cluster_labels, n_std=1.5):
    """
    Statistically identify small clusters based on size distribution.
    
    A cluster is considered "small" if its size is more than n_std standard 
    deviations below the mean cluster size.
    
    Args:
        cluster_labels: Array of cluster assignments
        n_std: Number of standard deviations below mean to define "small"
    
    Returns:
        small_cluster_ids: Set of cluster IDs that are statistically small
        cluster_sizes: Dict of cluster_id -> size
        stats_info: Dict with mean, std, threshold for reporting
    """
    unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
    cluster_sizes = dict(zip(unique_clusters, cluster_counts))
    
    sizes = np.array(cluster_counts)
    mean_size = sizes.mean()
    std_size = sizes.std()
    
    # Small cluster threshold: below mean - n_std * std
    small_threshold = max(mean_size - n_std * std_size, 2) # Ensure at least 2
    small_cluster_ids = set()
    for cluster_id, size in cluster_sizes.items():
        if size < small_threshold:
            small_cluster_ids.add(cluster_id)
    
    stats_info = {
        'mean_size': mean_size,
        'std_size': std_size,
        'small_threshold': small_threshold,
        'n_std': n_std
    }
    
    return small_cluster_ids, cluster_sizes, stats_info


def step_identify_small_clusters(results, pipeline, **kwargs):
    """Step 6: Identify small clusters statistically. (Optional)"""
    cluster_result = pipeline.get_result('step_perform_clustering', default=None)
    if cluster_result is None:
        print("Skipping small cluster identification: clustering not performed.")
        return {
            'small_clusters': set(),
            'cluster_sizes': {},
            'stats_info': {}
        }
    
    cluster_labels = cluster_result.get('cluster_labels')
    n_std = kwargs.get('n_std', 1.5)
    
    small_clusters, cluster_sizes, stats_info = identify_small_clusters(
        cluster_labels, n_std=n_std
    )
    
    return {
        'small_clusters': small_clusters,
        'cluster_sizes': cluster_sizes,
        'stats_info': stats_info
    }


def step_create_cluster_metadata(results, pipeline, **kwargs):
    """Step 7: Create cluster metadata with outlier flags. (Optional)"""
    cluster_result = pipeline.get_result('step_perform_clustering', default=None)
    if cluster_result is None:
        print("Skipping cluster metadata: clustering not performed.")
        return {'cluster_metadata_df': pd.DataFrame()}
    
    cluster_labels = cluster_result.get('cluster_labels')
    optimal_k = cluster_result.get('n_clusters')
    
    small_result = pipeline.get_result('step_identify_small_clusters', default={})
    small_clusters = small_result.get('small_clusters', set())
    cluster_sizes = small_result.get('cluster_sizes', {})
    
    # Get data_folder from load_data step
    load_result = pipeline.get_result('step_load_data', default={})
    # data_folder = load_result.get('data_folder', 'data')
    
    cluster_metadata = []
    for cluster_id in range(optimal_k):
        cluster_metadata.append({
            'cluster_id': cluster_id,
            'size': cluster_sizes.get(cluster_id, 0),
            'is_small': cluster_id in small_clusters,
            'outlier_status': 'SMALL' if cluster_id in small_clusters else 'NORMAL'
        })
    
    cluster_metadata_df = pd.DataFrame(cluster_metadata)
    # cluster_metadata_file = os.path.join(data_folder, 'workflows_cluster_metadata.csv')
    # cluster_metadata_df.to_csv(cluster_metadata_file, index=False)
    # print(f"\nSaved cluster metadata to: {cluster_metadata_file}")
    
    return {'cluster_metadata_df': cluster_metadata_df}


def save_clustering_results(df, cluster_labels, medoid_indices, X, column_names, kept_cols=None):
    """Save clustering results to files in the output folder, including scaled dataset.
    
    Args:
        df: Original dataframe with all columns
        cluster_labels: Cluster assignments for each sample
        medoid_indices: Indices of medoid samples
        X: Scaled data used for clustering
        column_names: Names of columns in X (after dimensionality reduction)
        output_folder: Output directory
        output_prefix: Prefix for output filenames
        kept_metrics: List of original metric columns that were kept after low-variance filtering.
                     If provided, only these metrics are saved in the clustered output.
    """

    # Add cluster labels to original data
    df_clustered = df.copy()
    df_clustered['cluster'] = cluster_labels
    
    # Filter to only kept metrics if provided
    if kept_cols is not None:
        # Keep workflowId and kept metrics, then add cluster label
        cols_to_keep = ['workflowId'] + kept_cols + ['cluster']
        # Only include columns that exist in the dataframe
        cols_to_keep = [col for col in cols_to_keep if col in df_clustered.columns]
        df_clustered = df_clustered[cols_to_keep]

    # Save clustered workflows

    # Save processed dataset for reuse in stats generation
    processed_df = pd.DataFrame(X, columns=column_names)
    processed_df['cluster'] = cluster_labels

    # Save medoid information
    medoid_data = []
    for cluster_id, medoid_idx in enumerate(medoid_indices):
        medoid_data.append({
            'cluster_id': cluster_id,
            'medoid_index': medoid_idx,
            'workflow_id': df.iloc[medoid_idx]['workflowId']
        })

    medoid_df = pd.DataFrame(medoid_data)

    return df_clustered, processed_df, medoid_df



def step_save_results(results, pipeline, **kwargs):
    """Step 8: Save all clustering results."""
    # Required dependencies
    load_result = pipeline.get_result('step_load_data', default={})
    df = load_result.get('df')
    hyperparam_cols = load_result.get('hyperparam_cols', [])

    cluster_result = pipeline.get_result('step_perform_clustering', default=None)
    if cluster_result is None:
        print("Skipping save: clustering not performed.")
        return {'df_clustered': None}
    
    cluster_labels = cluster_result.get('cluster_labels')
    medoid_indices = cluster_result.get('medoid_indices')
    

    prepare_result = pipeline.get_result('step_prepare_clustering_data', default={})
    X_scaled = prepare_result.get('X_scaled')
    
    # Get component names from dimensionality reduction if available
    dim_result = pipeline.get_result('step_dimensionality_reduction', default={})
    reduction_info = dim_result.get('reduction_info', {})
    component_names = reduction_info.get('component_names', [])
    
    # Handle kept_cols properly when dimensionality reduction is skipped
    kept_metric_cols = reduction_info.get('kept_metric_cols', None)
    if kept_metric_cols is not None:
        metric_cols_for_analysis = kept_metric_cols
        kept_cols = hyperparam_cols + kept_metric_cols
    else:
        # If no dimensionality reduction, use filtered metrics
        high_filter_result = pipeline.get_result('step_filter_high_variance_metrics', default={})
        low_filter_result = pipeline.get_result('step_filter_low_variance_metrics', default={})
        
        if high_filter_result:
            filtered_metrics = high_filter_result.get('metric_cols', [])
        elif low_filter_result:
            filtered_metrics = low_filter_result.get('metric_cols', [])
        else:
            filtered_metrics = load_result.get('metric_cols', [])

        metric_cols_for_analysis = filtered_metrics
        
        kept_cols = hyperparam_cols + filtered_metrics
    
    # Fallback to metric columns if no component names
    if not component_names:
        component_names = load_result.get('metric_cols', [])
    
    df_clustered, processed_df, medoid_df = save_clustering_results(
        df, cluster_labels, medoid_indices, X_scaled, component_names,
         kept_cols=kept_cols
    )
    
    
    
    return {
        'df_clustered': df_clustered,
        'processed_df': processed_df,
        'medoid_df': medoid_df,
        'metric_cols': metric_cols_for_analysis,
        'hyperparam_cols': hyperparam_cols,
    }



def build_default_pipeline():
    """Build the default clustering pipeline with all steps."""
    pipeline = ClusteringPipeline()
    
    # Add all steps in order
    pipeline.add_step('step_load_data', step_load_data, enabled=True)
    pipeline.add_step('step_filter_low_variance_metrics', step_filter_low_variance_metrics, enabled=True)
    pipeline.add_step('step_filter_high_variance_metrics', step_filter_high_variance_metrics, enabled=True)
    pipeline.add_step('step_dimensionality_reduction', step_dimensionality_reduction, enabled=True)
    pipeline.add_step('step_save_correlations', step_save_correlations, enabled=True)
    pipeline.add_step('step_prepare_clustering_data', step_prepare_clustering_data, enabled=True)
    pipeline.add_step('step_find_optimal_clusters', step_find_optimal_clusters, enabled=True)
    pipeline.add_step('step_perform_clustering', step_perform_clustering, enabled=True)
    pipeline.add_step('step_identify_small_clusters', step_identify_small_clusters, enabled=True)
    pipeline.add_step('step_create_cluster_metadata', step_create_cluster_metadata, enabled=True)
    pipeline.add_step('step_save_results', step_save_results, enabled=True)
    
    return pipeline

class InsightsPipeline:
    """
    Modular pipeline for ML workflow classification & feature selection.
    Allows enabling/disabling and reordering steps for testing.
    Steps can be optional and dependent steps handle missing results gracefully.
    """
    
    def __init__(self):
        """Initialize pipeline state."""
        self.steps = []
        self.results = {}
        self.enabled_steps = set()
        self.skip_missing_deps = True
    
    def add_step(self, name, func, enabled=True):
        """Add a step to the pipeline."""
        self.steps.append({'name': name, 'func': func})
        if enabled:
            self.enabled_steps.add(name)
        print(f"✓ Added step: {name} (enabled={enabled})")
    
    def enable_step(self, name):
        """Enable a specific step."""
        if name in [s['name'] for s in self.steps]:
            self.enabled_steps.add(name)
            print(f"✓ Enabled: {name}")
        else:
            print(f"✗ Step not found: {name}")
    
    def disable_step(self, name):
        """Disable a specific step."""
        self.enabled_steps.discard(name)
        print(f"✓ Disabled: {name}")
    
    def set_steps(self, step_names):
        """Enable only specific steps (disable all others)."""
        self.enabled_steps = set(step_names)
        print(f"✓ Pipeline set to run only: {', '.join(step_names)}")
    
    def list_steps(self):
        """Print all available steps and their status."""
       
        for step in self.steps:
            status = "✓ ENABLED" if step['name'] in self.enabled_steps else "✗ DISABLED"
            # print(f"  {status:12} - {step['name']}")
    
    def get_result(self, step_name, key=None, default=None):
        """Safely retrieve results from a previous step."""
        if step_name not in self.results:
            return default
        
        result = self.results[step_name]
        if key is None:
            return result
        
        if isinstance(result, dict):
            return result.get(key, default)
        
        return default
    
    def run(self, **kwargs):
        """Execute the pipeline, running only enabled steps in order."""
      
        
        for step in self.steps:
            step_name = step['name']
            
            if step_name not in self.enabled_steps:
                print(f"\n⊘ Skipping: {step_name}")
                continue
         
            
            try:
                result = step['func'](self.results, self, **kwargs)
                self.results[step_name] = result
                print(f"✓ Completed: {step_name}")
            except KeyError as e:
                print(f"✗ Error in {step_name}: Missing dependency {e}")
                if self.skip_missing_deps:
                    print(f"  Skipping this step due to missing results from a disabled step.")
                    continue
                else:
                    raise
            except Exception as e:
                print(f"✗ Error in {step_name}: {e}")
                raise
        
        return self.results


def step_phase1_load_data(results, pipeline, **kwargs):
    """Step 1.1: Load clustering results and prepare data."""
    df_clustered = kwargs.get('df_clustered')
    medoids = kwargs.get('medoids')
    X_standardized = kwargs.get('X_standardized')
    X_processed_df = kwargs.get('X_processed_df')
    metric_cols = kwargs.get('metric_cols')
    param_cols = kwargs.get('param_cols')
    cluster_labels = kwargs.get('cluster_labels')
    n_clusters = kwargs.get('n_clusters')
    small_clusters = kwargs.get('small_clusters', set())
    # data_folder = kwargs.get('data_folder', 'data')
    # csv_dir = kwargs.get('csv_dir', os.path.join(data_folder, 'csv'))
    # export_csv = kwargs.get('export_csv', True)
    
    return {
        'df_clustered': df_clustered,
        'medoids': medoids,
        'X_standardized': X_standardized,
        'X_processed_df': X_processed_df,
        'metric_cols': metric_cols,
        'param_cols': param_cols,
        'cluster_labels': cluster_labels,
        'n_clusters': n_clusters,
        'small_clusters': small_clusters,
        # 'data_folder': data_folder,
        # 'csv_dir': csv_dir,
        # 'export_csv': export_csv
    }


def step_phase1_feature_selection(results, pipeline, **kwargs):
    """
    Step 1.2-1.3: PHASE 1 - Feature Selection on Original Data
    
    - Multi-step feature selection (correlation, SHAP-based)
    - Track removed features
    """
    load_result = pipeline.get_result('step_phase1_load_data', default={})
    
    df_clustered = load_result.get('df_clustered')
    X_standardized = load_result.get('X_standardized')
    metric_cols = load_result.get('metric_cols')
    cluster_labels = load_result.get('cluster_labels')
    n_clusters = load_result.get('n_clusters')
    small_clusters = load_result.get('small_clusters', set())
    # data_folder = load_result.get('data_folder', 'data')
    # csv_dir = load_result.get('csv_dir', os.path.join(data_folder, 'csv'))
    # export_csv = load_result.get('export_csv', True)

    correlation_threshold = kwargs.get('correlation_threshold', 0.75)
    n_iterations = kwargs.get('n_iterations', None)
    ablation_mode = kwargs.get('ablation_mode', 'full')
    
    if any(v is None for v in [X_standardized, metric_cols, cluster_labels]):
        raise KeyError("step_phase1_load_data: Required data not loaded")
    
   
    
    all_results = []
    correlation_analysis_per_cluster = {}  # Store correlation analysis for removed features
    removed_features_analysis_per_cluster = {}  # Store detailed removed features analysis
    
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        n_cluster = np.sum(cluster_mask)
        
      
        
        if cluster_id in small_clusters:
            print(f" [SMALL - SKIPPED]")
            continue
        
        
        
        # Create binary classification: this cluster vs. others
        y_binary = (cluster_labels == cluster_id).astype(int)
        
    
        X_train = X_standardized
        y_train = y_binary
    
        # Feature selection (Steps 1-3 from paper) - Skip if ablation mode is no_iterative_filter
        if ablation_mode == 'no_iterative_filter':
            print(f"\n  ⚠ ABLATION: Skipping iterative feature selection")
            print(f"  Selecting top {n_iterations if n_iterations else 'all'} features by SHAP importance (no correlation removal)")
            
            
        else:
            selected_features, selection_history, removed_features_analysis = feature_selection_shap_iterative(
                X_train, y_train, metric_cols, n_iterations, correlation_threshold
            )
        
        print(f"\n✓ Final selected features ({len(selected_features)}): {selected_features}")
        
        # Store removed features analysis for later use in comprehensive insights
        removed_features_analysis_per_cluster[cluster_id] = removed_features_analysis
        
        # Store results
        cluster_result = {
            'cluster_id': cluster_id,
            'n_samples': n_cluster,
            'n_features_selected': len(selected_features),
            'selected_features': ','.join(selected_features),
        }
        all_results.append(cluster_result)
        
        # Save selection history with correlation data already embedded
        selection_history_expanded = []
        for entry in selection_history:
            for removed_feat, relation_measure in entry['removed_details']:
                selection_history_expanded.append({
                    'iteration': entry['iteration'],
                    'selected_feature': entry['selected_feature'],
                    'selected_shap_importance': entry['shap_importance'],
                    'removed_feature': removed_feat,
                    'relationship_measure': relation_measure
                })
        
        if selection_history_expanded:
            selection_df = pd.DataFrame(selection_history_expanded)
            # if export_csv:
            #     selection_df.to_csv(os.path.join(csv_dir, f'cluster_{cluster_id}_selection_history.csv'), index=False)
            #     print(f"✓ Saved cluster_{cluster_id}_selection_history.csv ({len(selection_df)} removed features tracked)")
        
        # Extract correlation analysis from selection history (removed features were correlated to selected by design)
      
        
        correlation_results = []
        for entry in selection_history:
            selected_feat = entry['selected_feature']
            for removed_feat, relation_measure in entry['removed_details']:
                if removed_feat != selected_feat:  # Skip the selected feature itself
                    correlation_results.append({
                        'removed_feature': removed_feat,
                        'selected_feature': selected_feat,
                        'relationship_strength': relation_measure
                    })
        
        if correlation_results:
            # Store correlation analysis for JSON output
            correlation_analysis_per_cluster[cluster_id] = correlation_results
            
            print(f"\n✓ Extracted {len(correlation_results)} removed-to-selected feature correlations from feature selection")
            print("\nTop Correlations (Removed vs Selected):")
            sorted_corr = sorted(correlation_results, key=lambda x: x['relationship_strength'], reverse=True)
            for idx, corr in enumerate(sorted_corr[:10], 1):
                print(f"  {corr['removed_feature']} (removed) ↔ {corr['selected_feature']} (selected)")
                print(f"    Relationship Strength: {corr['relationship_strength']:.4f}\n")
        else:
            print(f"\n⚠ No correlation analysis available")
            correlation_analysis_per_cluster[cluster_id] = []
    
    # Create summary results dataframe
    if all_results:
        results_summary = pd.DataFrame(all_results)
        # if export_csv:
        #     results_file = os.path.join(csv_dir, 'workflows_classification_results.csv')
        #     results_summary.to_csv(results_file, index=False)
        #     print(f"\n✓ Saved classification results summary to: {results_file}")
    else:
        results_summary = pd.DataFrame()
    
    return {
        'results_summary': results_summary,
        'selected_features_per_cluster': {r['cluster_id']: r['selected_features'] for r in all_results},
        'correlation_analysis_per_cluster': correlation_analysis_per_cluster,
        'removed_features_analysis_per_cluster': removed_features_analysis_per_cluster
    }


# ============ PIPELINE STEPS - PHASE 1 CONTINUED: TRADE-OFF ANALYSIS ============

def step_phase1_tradeoff_analysis(results, pipeline, **kwargs):
    """
    Step 1.3: PHASE 1 - Trade-off Analysis (Selected vs Non-Selected Features)
    
    For each cluster:
    - Identify selected vs non-selected features
    - Compute relationship measures between feature pairs
    - Save trade-off analysis: selected features vs non-selected features
    - Track negative correlations as trade-offs
    """
    load_result = pipeline.get_result('step_phase1_load_data', default={})
    feature_result = pipeline.get_result('step_phase1_feature_selection', default={})
    
    X_standardized = load_result.get('X_standardized')
    metric_cols = load_result.get('metric_cols')
    cluster_labels = load_result.get('cluster_labels')
    n_clusters = load_result.get('n_clusters')
    small_clusters = load_result.get('small_clusters', set())
    # data_folder = load_result.get('data_folder', 'data')
    # csv_dir = load_result.get('csv_dir', os.path.join(data_folder, 'csv'))
    # export_csv = load_result.get('export_csv', True)
    # csv_dir = load_result.get('csv_dir', os.path.join(data_folder, 'csv'))
    # export_csv = load_result.get('export_csv', True)

    selected_features_dict = feature_result.get('selected_features_per_cluster', {})
    correlation_threshold = kwargs.get('correlation_threshold', 0.75)
    
    if any(v is None for v in [X_standardized, metric_cols, cluster_labels]):
        raise KeyError("step_phase1_load_data: Required data not loaded for trade-off analysis")
    
    
    tradeoff_analysis_per_cluster = {}
    
    for cluster_id in range(n_clusters):
        if cluster_id in small_clusters:
            print(f"\n⊘ Cluster {cluster_id}: SKIPPED (small cluster)")
            tradeoff_analysis_per_cluster[cluster_id] = []
            continue
        
        cluster_mask = cluster_labels == cluster_id
        
        # Get selected features for this cluster
        selected_feat_str = selected_features_dict.get(cluster_id, '')
        if not selected_feat_str:
            print(f"\n⊘ Cluster {cluster_id}: No features selected, skipping trade-off analysis")
            tradeoff_analysis_per_cluster[cluster_id] = []
            continue
        
        selected_features = [f.strip() for f in selected_feat_str.split(',') if f.strip()]
        
      
        
        # Extract cluster data
        X_cluster = X_standardized[cluster_mask]
        X_cluster_df = pd.DataFrame(X_cluster, columns=metric_cols)
        
        var_types = detect_variable_types(X_cluster_df)
        correlation_results = []
        
        print(f"Representative metrics (selected features): {len(selected_features)}")
        print(f"Analyzing pairs: between representative metrics only")
        
        # Analyze pairs: between selected features only (representative metrics)
        for i, feat1 in enumerate(selected_features):
            for feat2 in selected_features[i+1:]:  # Only analyze unique pairs
                try:
                    measure, measure_type = compute_relationship_measure(
                        X_cluster_df, feat1, feat2, var_types
                    )
                    
                    x = X_cluster_df[feat1].values
                    y = X_cluster_df[feat2].values
                    print(f"  {feat1} ↔ {feat2} | Measure: {measure:.4f} ({measure_type})")
                    # Only include negative relationships as trade-offs
                    if measure < 0:
                        correlation_results.append({
                            'metric_1': feat1,
                            'metric_2': feat2,
                            'relationship_type': measure_type,
                            'relationship_strength': measure,
                            'is_tradeoff': 1 if measure < - correlation_threshold else 0
                        })
                except Exception as e:
                    pass
        
        if correlation_results:
            tradeoff_df = pd.DataFrame(correlation_results)
            tradeoff_df = tradeoff_df.sort_values('relationship_strength', ascending=False)
            # if export_csv:
            #     tradeoff_df.to_csv(os.path.join(csv_dir, f'cluster_{cluster_id}_tradeoff_analysis.csv'), index=False)
            
            # Store trade-off data for JSON output
            tradeoff_analysis_per_cluster[cluster_id] = tradeoff_df.to_dict('records')
            
            print(f"\n✓ Found {len(tradeoff_df)} negative correlations (trade-offs) in Cluster {cluster_id}")
            # if export_csv:
            #     print(f"✓ Saved cluster_{cluster_id}_tradeoff_analysis.csv")
            
            print("\nTop Trade-off Relationships (Strongest Negative Correlations):")
            for idx, row in tradeoff_df.head(10).iterrows():
                print(f"  {row['metric_1']} ↔ {row['metric_2']}")
                print(f"    Relationship Strength: {row['relationship_strength']:.4f} ({row['relationship_type']})")
                print(f"    Trade-off: {row['is_tradeoff']}\n")
        else:
            print(f"\n⚠ No negative correlations found in Cluster {cluster_id}")
            tradeoff_df = pd.DataFrame()
            # if export_csv:
            #     tradeoff_df.to_csv(os.path.join(csv_dir, f'cluster_{cluster_id}_tradeoff_analysis.csv'), index=False)
            # Store empty list for this cluster
            tradeoff_analysis_per_cluster[cluster_id] = []
    
    return {
        'status': 'completed',
        'tradeoff_analysis_per_cluster': tradeoff_analysis_per_cluster
    }


# ============ PIPELINE STEPS - PHASE 1 CONTINUED: MODEL TRAINING & EVALUATION ============
def feature_selection_shap_iterative(X_train, y_train, feature_names, n_iterations, correlation_threshold):
    """
    Steps 1-3: Iterative Feature Selection with SHAP + Correlation Removal.
    
    Following paper methodology:
    "First, if some features are to be kept because of their relevance to the study, all other 
    features with a strong relationship to the former are removed. Secondly, a random forest 
    classifier is iteratively trained. At each iteration, the feature with the highest SHAP 
    value not previously visited is selected and all other highly related features are removed. 
    
    Returns:
    - final_selected: List of final selected features
    - removal_analysis: DataFrame with feature removal metrics
    - selection_history: List of selected features with their removed correlates
    
    Process:
    1. Train RF on all remaining features
    2. Get SHAP values
    3. Select highest SHAP feature (not previously selected)
    4. Find ALL features related to it (using multi-type correlation)
    5. Remove those related features from remaining set
    6. REPEAT until convergence or n_iterations reached
    """
    print("\n" + "="*80)
    print("STEPS 1-3: Iterative SHAP Selection + Correlation Removal")
    print("="*80)
    print("Following paper: Iterative RF + SHAP ranking + correlation-based removal per iteration")
    print(f"Correlation threshold for removal: {correlation_threshold}")
    
    if n_iterations is None:
        n_iterations = min(10, len(feature_names) // 2) 
    
    selected_features = []
    previously_selected = set()
    remaining_features = list(feature_names)
    iteration = 0
    
    # Track selection history for feature removal analysis
    selection_history = []
    
    # Track removed features with their correlations for later analysis
    removed_features_analysis = {}  # {removed_feature: {'max_relationship': float, 'related_to': selected_feature, 'all_relationships': [...]}}
    
    # ============================================================================
    # Iterative SHAP-based selection with correlation removal
    # ============================================================================
    print("\n" + "-"*80)
    print("Iterative SHAP-based Feature Selection + Correlation Removal")
    print("-"*80)
    
    while len(selected_features) < n_iterations and len(remaining_features) > 1:
        iteration += 1
        
        # print(f"\n  Iteration {iteration}:")
        # print(f"    Training RF on {len(remaining_features)} remaining features...")
        
        # Step 1: Train random forest on REMAINING features only
        X_temp = X_train[:, [feature_names.index(f) for f in remaining_features]]
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_temp, y_train)
        
        # Step 2: Get SHAP importance for current feature set
        try:
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X_temp)
        except Exception as e:
            print(f"    Warning: SHAP failed, using tree importance fallback")
            shap_importance = rf.feature_importances_
            importance_with_idx = [(i, shap_importance[i], remaining_features[i]) 
                                   for i in range(len(remaining_features))]
        else:
            # Handle SHAP output format
            if isinstance(shap_values, list):
                shap_vals = shap_values[1]
            else:
                shap_vals = shap_values
            
            if len(shap_vals.shape) == 3:
                shap_importance = np.abs(shap_vals).mean(axis=(0, 2))
            else:
                shap_importance = np.abs(shap_vals).mean(axis=0)
            
            # Create list of (index, importance, feature_name) tuples
            importance_with_idx = [(i, shap_importance[i], remaining_features[i]) 
                                   for i in range(len(remaining_features))]
        
        # Sort by importance, descending
        importance_with_idx.sort(key=lambda x: x[1], reverse=True)
        
        # Step 3: Find highest SHAP feature not previously selected
        most_important_feature = None
        most_important_shap = 0
        
        for idx, shap_val, feat in importance_with_idx:
            if feat not in previously_selected:
                most_important_feature = feat
                most_important_shap = shap_val
                break
        
        # If all remaining features were already selected, stop
        if most_important_feature is None:
            print(f"    All remaining features already selected. Stopping.")
            break
        
        selected_features.append(most_important_feature)
        previously_selected.add(most_important_feature)
        
        print(f"    Selected: '{most_important_feature}' (SHAP: {most_important_shap:.6f})")
        
        # Step 4 & 5: Find and remove highly related features using multi-type correlation
        print(f"    Removing features related to '{most_important_feature}'...")
        
        # Compute relationship matrix for remaining features
        X_remaining = X_train[:, [feature_names.index(f) for f in remaining_features]]
        relationship_matrix = compute_relationship_matrix(X_remaining, remaining_features)
        
        # Get relationships to the selected feature
        related_to_selected = relationship_matrix[most_important_feature]
        # Find features exceeding correlation threshold (excluding the selected feature itself)
        to_remove = [f for f in remaining_features 
                     if f != most_important_feature 
                     and related_to_selected.get(f, 0) > correlation_threshold]
        to_remove.append(most_important_feature)  # Also remove the selected feature from remaining

        # Track removed features with their relationships
        removed_with_relations = [(f, related_to_selected.get(f, 0)) for f in to_remove]
        removed_with_relations.sort(key=lambda x: x[1], reverse=True)
        
        # Build removed features analysis (excluding the selected feature itself)
        removed_features_only = [f for f in to_remove if f != most_important_feature]
        for removed_feat, relation_measure in removed_with_relations:
            if removed_feat != most_important_feature:  # Skip the selected feature
                if removed_feat not in removed_features_analysis:
                    removed_features_analysis[removed_feat] = {
                        'max_relationship': relation_measure,
                        'related_to': most_important_feature,
                        'all_relationships': []
                    }
                else:
                    # Update max_relationship if this one is stronger
                    if relation_measure > removed_features_analysis[removed_feat]['max_relationship']:
                        removed_features_analysis[removed_feat]['max_relationship'] = relation_measure
                        removed_features_analysis[removed_feat]['related_to'] = most_important_feature
                
                # Track all relationships for this removed feature
                removed_features_analysis[removed_feat]['all_relationships'].append({
                    'selected_feature': most_important_feature,
                    'relationship_measure': relation_measure
                })
        
        selection_history.append({
            'iteration': iteration,
            'selected_feature': most_important_feature,
            'shap_importance': most_important_shap,
            'features_removed': to_remove,
            'removed_count': len(to_remove),
            'removed_details': removed_with_relations  # Keep for detailed tracking
        })
        
        if to_remove:
            print(f"    Removing {len(to_remove)} related features (measure > {correlation_threshold}):")
            for removed_feat, relation in removed_with_relations[:5]:  # Show first 5
                print(f"      - {removed_feat} (relation: {relation:.4f})")
            if len(to_remove) > 5:
                print(f"      ... and {len(to_remove) - 5} more")
        
        # Step 6: Update remaining features for next iteration
        remaining_features = [f for f in remaining_features if f not in to_remove]
    
    print(f"\n✓ After iterative selection: {len(selected_features)} features selected")
    
    return selected_features, selection_history, removed_features_analysis


def step_phase1_model_training_and_evaluation(results, pipeline, **kwargs):
    """
    Step 1.3-1.4: PHASE 1 - Model Training & SHAP Interpretation
    
    For each cluster:
    1. Use selected features from step_phase1_feature_selection
    2. Train XGBoost classifier for cluster vs. others
    3. Hyperparameter grid search with cross-validation
    4. Evaluate with AUC, balanced accuracy, confusion matrix, precision, recall, F1
    5. Generate local & global SHAP values using TreeExplainer
    
    Requires XGBoost to be installed: pip install xgboost
    """
    try:
        import xgboost as xgb
        from sklearn.model_selection import GridSearchCV, cross_val_score
        from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, balanced_accuracy_score
    except ImportError:
        print("⚠ Warning: XGBoost not installed. Model training skipped.")
        print("  Install with: pip install xgboost")
        return {
            'status': 'skipped',
            'message': 'XGBoost not available',
            'model_results': []
        }
    
    load_result = pipeline.get_result('step_phase1_load_data', default={})
    feature_result = pipeline.get_result('step_phase1_feature_selection', default={})
    
    X_standardized = load_result.get('X_standardized')
    cluster_labels = load_result.get('cluster_labels')
    n_clusters = load_result.get('n_clusters')
    small_clusters = load_result.get('small_clusters', set())
    metric_cols = load_result.get('metric_cols')
    # data_folder = load_result.get('data_folder', 'data')
    # csv_dir = load_result.get('csv_dir', os.path.join(data_folder, 'csv'))
    # export_csv = load_result.get('export_csv', True)
    # csv_dir = load_result.get('csv_dir', os.path.join(data_folder, 'csv'))
    # export_csv = load_result.get('export_csv', True)

    selected_features_dict = feature_result.get('selected_features_per_cluster', {})
    
    if any(v is None for v in [X_standardized, cluster_labels, metric_cols]):
        raise KeyError("step_phase1_load_data: Required data not available for model training")
    
    print("\n" + "="*80)
    print("PHASE 1 STEP 3: MODEL TRAINING & EVALUATION")
    print("Training XGBoost classifiers per cluster with SHAP interpretation")
    print("="*80)
    
    all_model_results = []
    
    for cluster_id in range(n_clusters):
        if cluster_id in small_clusters:
            print(f"\n⊘ Cluster {cluster_id}: SKIPPED (small cluster)")
            continue
        
        cluster_mask = cluster_labels == cluster_id
        n_cluster = np.sum(cluster_mask)
        
        # Get selected features for this cluster
        selected_feat_str = selected_features_dict.get(cluster_id, '')
        if not selected_feat_str:
            print(f"\n⊘ Cluster {cluster_id}: No features selected, skipping model training")
            continue
        
        selected_features = [f.strip() for f in selected_feat_str.split(',') if f.strip()]
        
        print(f"\n{'='*80}")
        print(f"CLUSTER {cluster_id}: Training XGBoost Model")
        print(f"{'='*80}")
        print(f"Selected features ({len(selected_features)}): {selected_features[:5]}", end="")
        if len(selected_features) > 5:
            print(f" ... and {len(selected_features) - 5} more")
        else:
            print()
        
        # Create binary classification: this cluster vs. others
        y_binary = (cluster_labels == cluster_id).astype(int)
        
        # Check if we have enough samples in both classes
        unique_classes, class_counts = np.unique(y_binary, return_counts=True)
        min_class_count = class_counts.min()
        
        if min_class_count < 2:
            print(f"\n⊘ Cluster {cluster_id}: Skipping - minority class has only {min_class_count} sample(s)")
            print(f"   Class distribution: {dict(zip(unique_classes, class_counts))}")
            print(f"   This happens when almost all data is in one cluster (likely due to small cluster filtering)")
            continue
        
        print(f"y_binary distribution: {dict(zip(unique_classes, class_counts))}")
        
        # Get feature indices
        feature_indices = [metric_cols.index(f) for f in selected_features]
        X_selected = X_standardized[:, feature_indices]
        
        # Train-test split (80-20). Always stratify: the `min_class_count < 2`
        # check above already guarantees both classes have at least 2 members,
        # which is all sklearn's stratify needs for a single split. A plain
        # random split was used here for small minority classes, but with a
        # tiny dataset (a handful of runs) that let the 20% test slice land
        # entirely on one class by chance — roc_auc_score then crashes with
        # "Only one class present in y_true" a few lines down. Stratifying
        # guarantees both classes appear in the test set whenever they can.
        if min_class_count < 10:
            print(f"Warning: Minority class has only {min_class_count} samples; splitting with stratify to keep both classes in the test set")
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y_binary, test_size=0.2, random_state=42, stratify=y_binary
        )
        
        print(f"Train: {len(X_train)} | Test: {len(X_test)}")
        print(f"Class distribution (train): {np.bincount(y_train)}")
        print(f"Class distribution (test): {np.bincount(y_test)}")
        
        # ========== Hyperparameter Grid Search ==========
        param_grid = {
            'max_depth': [3, 5, 7],
            'n_estimators': [30, 60],
            'eta': [0.15, 0.25],
        }

        # GridSearchCV's default scoring here uses StratifiedKFold, which needs
        # at least `cv` members of EACH class in the training set — cv=5
        # crashes outright on a small cluster ("n_splits=5 cannot be greater
        # than the number of members in each class"). Cap folds to what the
        # data can actually support, and skip grid search entirely (fit one
        # reasonable default) when even 2-fold CV isn't possible.
        train_class_counts = np.bincount(y_train)
        min_train_class_count = int(train_class_counts[train_class_counts > 0].min())
        cv_folds = min(5, min_train_class_count)

        if cv_folds < 2:
            print(f"⊘ Cluster {cluster_id}: smallest training class has only {min_train_class_count} sample(s) — "
                  f"too few for cross-validation, fitting with default hyperparameters instead of grid search")
            best_params = {'max_depth': 5, 'n_estimators': 60, 'eta': 0.15}
            best_model = xgb.XGBClassifier(random_state=42, eval_metric='auc', verbosity=0, **best_params)
            best_model.fit(X_train, y_train)
            best_cv_score = None
        else:
            if cv_folds < 5:
                print(f"Warning: smallest training class has {min_train_class_count} sample(s); "
                      f"reducing grid-search CV folds from 5 to {cv_folds}")
            print(f"\nPerforming hyperparameter grid search with {cv_folds}-fold cross-validation...")
            xgb_base = xgb.XGBClassifier(random_state=42, eval_metric='auc', verbosity=0)
            grid_search = GridSearchCV(xgb_base, param_grid, cv=cv_folds, scoring='roc_auc', n_jobs=-1, verbose=0)
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_cv_score = grid_search.best_score_

        print(f"✓ Best CV AUC: {best_cv_score:.4f}" if best_cv_score is not None else "✓ Fit with default hyperparameters (no grid search)")
        print(f"  Best parameters: {best_params}")
        
        # ========== Model Evaluation ==========
        print(f"\nEvaluating on test set...")
        
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Belt-and-suspenders: stratification above should guarantee both
        # classes land in y_test, but fall back instead of crashing the whole
        # request if an edge case (e.g. a 1-sample test set) still slips
        # through. 0.0 matches this file's existing "no valid AUC" sentinel
        # (see the fallback model_evaluation dicts below) rather than NaN,
        # which downstream JSON serialization on the Java side can't handle.
        if len(np.unique(y_test)) < 2:
            print(f"⊘ Cluster {cluster_id}: test set has only one class present ({np.unique(y_test)}); AUC is undefined, reporting 0.0")
            auc = 0.0
        else:
            auc = roc_auc_score(y_test, y_pred_proba)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)

        # Ensure confusion matrix and report always account for both classes (0 and 1)
        # so that indexing cm[0, 1], cm[1, 0], cm[1, 1] and report['1'] is safe
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        report = classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            output_dict=True,
            zero_division=0,
        )
        
        # print(f"Test AUC: {auc:.4f}")
        # print(f"Balanced Accuracy: {balanced_acc:.4f}")
        # print(f"Confusion Matrix:\n{cm}")
        # print(f"Classification Report:")
        # print(f"  Precision (cluster): {report['1']['precision']:.4f}")
        # print(f"  Recall (cluster): {report['1']['recall']:.4f}")
        # print(f"  F1 (cluster): {report['1']['f1-score']:.4f}")
        
        # ========== SHAP Interpretability ==========
        print(f"\nGenerating SHAP values for interpretation...")
        
        high_shap_features = []
        try:
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_test)

            # Handle SHAP output format (binary classification)
            if isinstance(shap_values, list):
                shap_vals = shap_values[1]
            else:
                shap_vals = shap_values
            
            # Global feature importance (mean |SHAP|)
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            shap_importance = dict(zip(selected_features, mean_abs_shap))
            shap_importance_sorted = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)
            
            # top_percentile = 95  # Top 5% of features
            # importance_values = [imp for _, imp in shap_importance_sorted]
            # threshold = np.percentile(importance_values, top_percentile) if importance_values else 0
            # high_shap_features = [feat for feat, importance in shap_importance_sorted if importance >= threshold]
            
            # Select all features with SHAP value > 0.1
            high_shap_features = [feat for feat, importance in shap_importance_sorted if importance > 0.1]
            
            print(f"✓ SHAP features with importance > 0.1 ({len(high_shap_features)} features):")
            for feat, importance in shap_importance_sorted:
                if importance > 0.1:
                    print(f"    {feat}: {importance:.6f}")
            
        except Exception as e:
            print(f"⚠ SHAP analysis failed: {e}") 
            shap_importance_sorted = []
        
        # Store model results
        model_result = {
            'cluster_id': cluster_id,
            'n_samples': n_cluster,
            'n_features_used': len(selected_features),
            'selected_features': ','.join(selected_features),
            'test_auc': auc,
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score'],
            'balanced_accuracy': balanced_acc,
            'tn': int(cm[0, 0]),
            'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]),
            'tp': int(cm[1, 1]),
            'high_shap_features': ','.join(high_shap_features)
        }
        all_model_results.append(model_result)
        
       
    # Create summary results dataframe
    if all_model_results:
        models_summary = pd.DataFrame(all_model_results)
        # if export_csv:
        #     models_file = os.path.join(csv_dir, 'workflows_model_evaluation_summary.csv')
        #     models_summary.to_csv(models_file, index=False)
        #     print(f"\n✓ Saved model evaluation summary to: {models_file}")
    else:
        models_summary = pd.DataFrame()
    
    return {
        'status': 'completed',
        'model_results': all_model_results,
        'models_summary': models_summary
    }


def step_phase1_decision_tree_rules(results, pipeline, **kwargs):
    """
    Step 1.5: Using imodels built-in rule evaluation
    """
    try:
        from imodels import SkopeRulesClassifier
        from sklearn.preprocessing import LabelEncoder
        import numpy as np
        import pandas as pd
        import os
    except ImportError as e:
        print(f"⚠ Warning: {e}")
        return {'status': 'skipped', 'message': str(e)}
    
    load_result = pipeline.get_result('step_phase1_load_data', default={})
    
    cluster_labels = load_result.get('cluster_labels')
    n_clusters = load_result.get('n_clusters')
    small_clusters = load_result.get('small_clusters', set())
    df_clustered = load_result.get('df_clustered')
    param_cols = load_result.get('param_cols')
    # data_folder = load_result.get('data_folder', 'data')
    # csv_dir = load_result.get('csv_dir', os.path.join(data_folder, 'csv'))
    # export_csv = load_result.get('export_csv', True)
    
    if any(v is None for v in [cluster_labels, n_clusters, df_clustered, param_cols]):
        raise KeyError("Required data not available")
    
    # Sanitize column names
    hyperparameters = [col.replace(' ', '_') for col in param_cols]
    df_clustered.columns = [col.replace(' ', '_') for col in df_clustered.columns]
    
    print("\n" + "="*80)
    print("PHASE 1 STEP 5: CLUSTER RULES (Using Built-in Rule Evaluation)")
    print("="*80)
    
    # Prepare data
    X_hyperparam = df_clustered[hyperparameters].copy()
    
    label_encoders = {}
    categorical_mappings = {}
    
    for col in X_hyperparam.columns:
        if X_hyperparam[col].dtype == 'object':
            le = LabelEncoder()
            X_hyperparam[col] = le.fit_transform(X_hyperparam[col])
            label_encoders[col] = le
            categorical_mappings[col] = {i: val for i, val in enumerate(le.classes_)}
    
    X_array = X_hyperparam.values
    y_cluster = cluster_labels
    
    def decode_categorical_in_rule(rule_str, categorical_mappings, label_encoders):
        """Decode categorical features"""
        import re
        decoded = rule_str
        
        for col, mapping in categorical_mappings.items():
            if col not in label_encoders:
                continue
            
            le = label_encoders[col]
            all_classes = le.classes_
            
            patterns = [
                (rf'{col}\s*<=\s*([\d.]+)', '<='),
                (rf'{col}\s*>\s*([\d.]+)', '>'),
            ]
            
            for pattern, operator in patterns:
                matches = list(re.finditer(pattern, decoded))
                for match in matches:
                    threshold = float(match.group(1))
                    original_condition = match.group(0)
                    
                    if operator == '<=':
                        valid_indices = [i for i in range(len(all_classes)) if i <= threshold]
                    else:
                        valid_indices = [i for i in range(len(all_classes)) if i > threshold]
                    
                    valid_categories = [all_classes[i] for i in valid_indices]
                    
                    if len(valid_categories) == 1:
                        cat_val = valid_categories[0]
                        if pd.isna(cat_val):
                            replacement = f"{col} = None"
                        else:
                            replacement = f"{col} = '{cat_val}'"
                    else:
                        # Handle NaN values in categories
                        categories_parts = []
                        for c in valid_categories:
                            if pd.isna(c):
                                categories_parts.append('None')
                            else:
                                categories_parts.append(f"'{c}'")
                        categories_str = ', '.join(categories_parts)
                        replacement = f"{col} IN {{{categories_str}}}"
                    
                    decoded = decoded.replace(original_condition, replacement, 1)
        
        return decoded
    
    all_cluster_rules = {}
    cluster_rules_summary = []
    
    for cluster_id in range(n_clusters):
        if cluster_id in small_clusters:
            continue
        
        y_binary = (y_cluster == cluster_id).astype(int)
        n_cluster_samples = y_binary.sum()
        
        print(f"\n{'─'*80}")
        print(f"Cluster {cluster_id} ({n_cluster_samples} workflows)")
        print(f"{'─'*80}")
        
        clf = SkopeRulesClassifier(
            max_depth_duplication=3,
            n_estimators=30,
            precision_min=0.5,
            recall_min=0.1,
            max_depth=4,
            random_state=42
        )
        
        try:
            clf.fit(X_array, y_binary, feature_names=hyperparameters)
            
            rules_list = clf.rules_
            print(f"  ✓ Extracted {len(rules_list)} rules")
            
            if len(rules_list) == 0:
                continue
            
            # Get cluster-specific data
            cluster_mask = (y_cluster == cluster_id)
            X_cluster = X_array[cluster_mask]
            df_cluster = X_hyperparam[cluster_mask]
            
            cluster_rules = []
            for rule_obj in rules_list:
                rule_str = rule_obj.rule
                precision, recall, nb_samples_total = rule_obj.args
                
                # USE BUILT-IN: Query the rule on cluster data using pandas
                try:
                    # imodels internally uses df.query() - we do the same
                    satisfied_mask = df_cluster.query(rule_str).index
                    n_workflows_in_cluster = len(satisfied_mask)
                    
                    # Cluster-specific recall
                    cluster_recall = n_workflows_in_cluster / n_cluster_samples if n_cluster_samples > 0 else 0
                    
                except Exception as e:
                    print(f"    ⚠ Could not evaluate rule: {rule_str}")
                    continue
                
                decoded_rule = decode_categorical_in_rule(rule_str, categorical_mappings, label_encoders)
                
                if precision + cluster_recall > 0:
                    f1_score = 2 * (precision * cluster_recall) / (precision + cluster_recall)
                else:
                    f1_score = 0.0
                
                # Score based on cluster-specific counts
                # significance_weight = np.log1p(n_workflows_in_cluster)
                p_workflows_in_cluster = n_workflows_in_cluster / n_cluster_samples if n_cluster_samples > 0 else 0
                combined_score = f1_score * p_workflows_in_cluster
                
                cluster_rules.append({
                    'rule': decoded_rule,
                    'precision': precision,
                    'recall': cluster_recall,
                    'f1_score': f1_score,
                    'n_workflows_in_cluster': int(n_workflows_in_cluster),
                    'combined_score': combined_score
                })
            
            cluster_rules = sorted(cluster_rules, key=lambda x: x['combined_score'], reverse=True)
            all_cluster_rules[cluster_id] = cluster_rules
            
            print(f"\n  Top rules (by F1 × cluster coverage):")
            for idx, rule_info in enumerate(cluster_rules[:3], 1):
                # print(f"\n  Rule {idx}: {rule_info['rule']}")
                # print(f"           F1: {rule_info['f1_score']:.3f} | "
                #       f"Precision: {rule_info['precision']:.3f} | "
                #       f"Recall: {rule_info['recall']:.3f}")
                # print(f"           Covers {rule_info['n_workflows_in_cluster']}/{n_cluster_samples} workflows in this cluster")
                
                cluster_rules_summary.append({
                    'cluster_id': cluster_id,
                    'rule_number': idx,
                    'rule': rule_info['rule'],
                    'precision': rule_info['precision'],
                    'recall': rule_info['recall'],
                    'f1_score': rule_info['f1_score'],
                    'n_workflows_in_cluster': rule_info['n_workflows_in_cluster'],
                    'cluster_size': n_cluster_samples,
                    'combined_score': rule_info['combined_score']
                })
        
        except Exception as e:
            print(f"  ⚠ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if cluster_rules_summary:
        rules_df = pd.DataFrame(cluster_rules_summary)
        # if export_csv:
        #     rules_file = os.path.join(csv_dir, 'cluster_decision_rules.csv')
        #     rules_df.to_csv(rules_file, index=False)
        #     print(f"\n✓ Saved to: {rules_file}")
    else:
        rules_df = pd.DataFrame()
    
    print("\n" + "="*80)
    print("BEST RULE PER CLUSTER")
    print("="*80)
    for cluster_id in sorted(all_cluster_rules.keys()):
        rules = all_cluster_rules[cluster_id]
        if rules:
            main_rule = rules[0]
            n_total = df_clustered[df_clustered['cluster'] == cluster_id].shape[0]
            print(f"\nCluster {cluster_id}: {main_rule['rule']}")
            print(f"  [F1: {main_rule['f1_score']:.3f}, covers {main_rule['n_workflows_in_cluster']}/{n_total} workflows]")
    
    return {
        'status': 'completed',
        'cluster_rules': all_cluster_rules,
        'rules_summary': rules_df
    }



def generate_cluster_names_with_ollama(cluster_insights):

    """
    Call a remote Ollama model to assign short names to clusters.
    Args:
        cluster_insights (dict): Dictionary of cluster insights.
    Returns:
        dict: Parsed JSON object with cluster names, or None on failure.
    """
    import requests
    import json

    # Helper to convert numpy types to native Python types
    def to_serializable(obj):
        import numpy as np
        if isinstance(obj, dict):
            return {to_serializable(k): to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_serializable(i) for i in obj]
        elif isinstance(obj, tuple):
            return tuple(to_serializable(i) for i in obj)
        elif isinstance(obj, np.generic):
            return obj.item()
        else:
            return obj

    # Reduce cluster_insights to only metadata, high_shap_features, and distinct_features
    minimal_insights = {}
    for cid, info in cluster_insights.items():
        minimal_insights[cid] = {
            'metadata': info.get('metadata', {}),
            'high_shap_features': info.get('high_shap_features', {}),
            'distinct_features': info.get('distinct_features', {})
        }
    cluster_insights_serializable = to_serializable(minimal_insights)

    prompt = (
        "You are a data scientist helping to label clusters of ML workflows.\n\n"
        "You receive a JSON object called `clusterInsights`. Each key is a cluster id\n"
        "and each value contains:\n"
        "- `metadata`: cluster size and share\n"
        "- `high_shap_features` and `distinct_features`: which metrics distinguish the cluster\n\n"
        "TASK:\n"
        "1. For each cluster, assign a short, human-readable name of 3–4 words.\n"
        "2. The name should reflect the most distinctive metrics and high shap metrics\n"
        "     (e.g., “High Precision, Low Recall”, “Well-Calibrated Fair Models”).\n"
        "3. Return ONLY a compact JSON object of the form:\n\n"
        "{\n"
        "    \"0\": {\"cluster_name\": \"<3-4 word name>\"},\n"
        "    \"1\": {\"cluster_name\": \"<3-4 word name>\"},\n"
        "    ...\n"
        "}\n\n"
        "STRICT OUTPUT RULES:\n"
        "- Respond with a single JSON object only.\n"
        "- Keys MUST be the cluster ids present in clusterInsights (e.g. \"0\", \"1\", ...).\n"
        "- Each value MUST be an object with exactly one key: \"cluster_name\".\n"
        "- Do NOT include any other keys like \"text\", \"message\", \"analysis\", etc.\n"
        "- Do NOT add explanations, narration, markdown, or questions.\n\n"
        "Here is the `clusterInsights` JSON: \n\"\"\"\n"
        f"{json.dumps(cluster_insights_serializable, ensure_ascii=False)}\n\"\"\""
    )

    print(prompt)

    url = "http://gatepc73.cs.uoi.gr:27880/llm_api/api/generate"
    import os
    api_key = os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        raise RuntimeError("OLLAMA_API_KEY environment variable not set. Please set it before running.")
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key
    }
    data = {
        "model": "granite4:3b",
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=60)
        response.raise_for_status()
        result = response.json()
        # Ollama returns the actual JSON as a string in the 'response' field
        if isinstance(result, dict) and 'response' in result:
            try:
                cluster_names = json.loads(result['response'])
            except Exception:
                return None
        elif isinstance(result, dict) and 'text' in result:
            try:
                cluster_names = json.loads(result['text'])
            except Exception:
                return None
        else:
            cluster_names = result
        if not isinstance(cluster_names, dict):
            return None
        cluster_ids = set(str(k) for k in cluster_insights.keys())
        if not cluster_ids.issubset(set(cluster_names.keys())):
            return None
        return cluster_names
    except Exception as e:
        print(f"[Ollama] Error generating cluster names: {e}")
        return None
    
def step_phase1_comprehensive_cluster_insights(results, pipeline, **kwargs):
    """
    Step 1.6: Generate Comprehensive Cluster Statistics JSON
    
    Aggregates all important statistics from previous steps into a single
    comprehensive JSON file for each cluster containing:
    - Cluster metadata (size, samples)
    - Feature selection results (n_selected, selected_features)
    - Model evaluation metrics (AUC, F1, balanced accuracy, precision, recall)
    - High SHAP features (features with importance > 0.1)
    - Trade-off analysis (negative correlations between selected and non-selected)
    - Hyperparameter patterns (dominant values and percentages)
    - Decision tree rules (interpretable hyperparameter combinations)
    """
    import json
    
    load_result = pipeline.get_result('step_phase1_load_data', default={})
    feature_result = pipeline.get_result('step_phase1_feature_selection', default={})
    model_result = pipeline.get_result('step_phase1_model_training', default={})
    rules_result = pipeline.get_result('step_phase1_decision_tree_rules', default={})
    tradeoff_result = pipeline.get_result('step_phase1_tradeoff_analysis', default={})
    
    df_clustered = load_result.get('df_clustered')
    medoids = load_result.get('medoids')
    cluster_labels = load_result.get('cluster_labels')
    n_clusters = load_result.get('n_clusters')
    small_clusters = load_result.get('small_clusters', set())
    metric_cols = load_result.get('metric_cols')
    param_cols = load_result.get('param_cols')
    # data_folder = load_result.get('data_folder', 'data')
    X_standardized = load_result.get('X_standardized')
    correlation_threshold = kwargs.get('correlation_threshold', 0.75)

    results_summary = feature_result.get('results_summary', pd.DataFrame())
    models_summary = model_result.get('models_summary', pd.DataFrame())
    rules_summary = rules_result.get('rules_summary', pd.DataFrame())
    
    if df_clustered is None:
        raise KeyError("step_phase1_load_data: Required data not available")
    
    print("\n" + "="*80)
    print("PHASE 1 STEP 6: GENERATING COMPREHENSIVE CLUSTER STATISTICS")
    print("="*80)
    
    # Use loaded hyperparameters
    hyperparameters = param_cols
    
    cluster_insights_dict = {}
    
    for cluster_id in range(n_clusters):
        if cluster_id in small_clusters:
            continue
        
        cluster_mask = cluster_labels == cluster_id
        cluster_df = df_clustered[cluster_mask].copy()
        n_cluster = len(cluster_df)
        
        # Get medoid ID for this cluster
        medoid_id = None
        medoid_index = None
        if medoids is not None and not medoids.empty:
            cluster_medoids = medoids[medoids['cluster_id'] == cluster_id]
            if not cluster_medoids.empty:
                medoid_id = cluster_medoids.iloc[0]['workflow_id'] if 'workflow_id' in cluster_medoids.columns else None
                medoid_index = int(cluster_medoids.iloc[0]['medoid_index']) if 'medoid_index' in cluster_medoids.columns else None
        
        cluster_insights = {
            'cluster_id': cluster_id,
            'metadata': {
                'n_workflows': n_cluster,
                'percentage_of_total': round((n_cluster / len(df_clustered)) * 100, 2),
                'medoid_workflow_id': medoid_id,
                'medoid_index': medoid_index,
            },
            'feature_selection': {},
            'model_evaluation': {},
            'high_shap_features': [],
            'correlation_analysis': {},
            'trade_off_analysis': {},
            'hyperparameter_patterns': {},
            'decision_tree_rules': []
        }
        
        # ===== FEATURE SELECTION INSIGHTS =====
        if not results_summary.empty:
            cluster_fs = results_summary[results_summary['cluster_id'] == cluster_id]
            if not cluster_fs.empty:
                row = cluster_fs.iloc[0]
                selected_feats = row['selected_features'].split(',') if isinstance(row['selected_features'], str) else []
                cluster_insights['feature_selection'] = {
                    'n_features_selected': row['n_features_selected'],
                    'selected_features': selected_feats,
                    'n_metrics_total': len(metric_cols)
                }
                
                # Compute feature statistics comparing cluster vs others
                # This will be reused for high_shap_features to avoid redundant computation
                feature_stats = {}
                other_mask = cluster_labels != cluster_id
                
                for feat in selected_feats:
                    if feat in metric_cols:
                        feat_idx = metric_cols.index(feat)
                        
                        # Get values for this cluster
                        cluster_values = X_standardized[cluster_mask, feat_idx]
                        cluster_mean = float(np.mean(cluster_values))
                        cluster_std = float(np.std(cluster_values))
                        
                        # Get values for all other clusters
                        other_values = X_standardized[other_mask, feat_idx]
                        other_mean = float(np.mean(other_values))
                        other_std = float(np.std(other_values))
                        
                        # Compute global stats for z-score
                        all_values = X_standardized[:, feat_idx]
                        global_mean = float(np.mean(all_values))
                        global_std = float(np.std(all_values))
                        
                        # Classify cluster's mean value
                        z_score = (cluster_mean - global_mean) / (global_std + 1e-6)
                        # Calculate how distinctive this feature is for the cluster
                        distinctiveness = (cluster_mean - other_mean) / (other_std + 1e-6)
                        use = z_score
                        threshold = 1.0
                        if use > threshold:
                            value_category = "high"  # > threshold std above global
                        elif use < -threshold:
                            value_category = "low"   # < threshold std below global
                        else:
                            value_category = "mid"
                        
                        
                        feature_stats[feat] = {
                            'cluster_mean': round(cluster_mean, 4),
                            'cluster_std': round(cluster_std, 4),
                            'other_clusters_mean': round(other_mean, 4),
                            'other_clusters_std': round(other_std, 4),
                            'value_category': value_category,
                            'distinctiveness_score': round(distinctiveness, 4),
                            'z-score': round(z_score, 4),
                        }
                
                cluster_insights['feature_selection']['feature_statistics'] = feature_stats
                
                # ===== DISTINCT FEATURES SECTION (High/Low only) =====
                # Filter feature_stats to keep only high and low features
                distinct_features = {
                    feat: stats for feat, stats in feature_stats.items()
                    if stats.get('value_category') in ['high', 'low']
                }

                if distinct_features:
                    # Align structure with high_shap_features: expose both the list of
                    # feature names and their statistics so downstream consumers can
                    # treat them uniformly.
                    cluster_insights['distinct_features'] = {
                        'n_distinct_features': len(distinct_features),
                        'features': list(distinct_features.keys()),
                        'feature_statistics': distinct_features
                    }
        
        # ===== CORRELATION ANALYSIS FOR REMOVED FEATURES =====
        # Get removed features analysis from feature selection step
        removed_features_per_cluster = feature_result.get('removed_features_analysis_per_cluster', {})
        cluster_removed_features = removed_features_per_cluster.get(cluster_id, {})
        
        if cluster_removed_features:
            cluster_insights['correlation_analysis'] = {
                'n_removed_features': len(cluster_removed_features),
                'removed_features': {
                    feat: {
                        'max_relationship': round(float(details['max_relationship']), 4),
                        'related_to': details['related_to'],
                        'all_relationships': details.get('all_relationships', [])
                    }
                    for feat, details in cluster_removed_features.items()
                }
            }
        
        # ===== MODEL EVALUATION INSIGHTS =====
        if not models_summary.empty:
            cluster_model = models_summary[models_summary['cluster_id'] == cluster_id]
            if not cluster_model.empty:
                row = cluster_model.iloc[0]
                high_shap = row['high_shap_features'].split(',') if isinstance(row['high_shap_features'], str) else []
                high_shap = [f.strip() for f in high_shap if f.strip()]
                
                # Calculate model quality score
                test_auc = float(row['test_auc'])
                f1 = float(row['f1_score'])
                precision = float(row['precision'])
                recall = float(row['recall'])
                balanced_acc = float(row['balanced_accuracy']) if 'balanced_accuracy' in row else None
                
                # Quality score: weighted combination of metrics
                # Prioritize AUC as main discriminator, then balanced F1
                model_quality_score = round((test_auc * 0.6 + f1 * 0.4), 4)
                
                # Generate quality interpretation
                if model_quality_score >= 0.9:
                    quality_level = "Excellent - Cluster is very well distinguished"
                elif model_quality_score >= 0.7:
                    quality_level = "Good - Cluster is well distinguished"
                elif model_quality_score >= 0.5:
                    quality_level = "Fair - Cluster has moderate distinction"
                else:
                    quality_level = "Poor - Cluster is not well distinguished"
                
                cluster_insights['model_evaluation'] = {
                    'test_auc': round(float(row['test_auc']), 4),
                    'balanced_accuracy': round(float(balanced_acc), 4) if balanced_acc is not None else None,
                    'precision': round(float(row['precision']), 4),
                    'recall': round(float(row['recall']), 4),
                    'f1_score': round(float(row['f1_score']), 4),
                    'model_quality_score': model_quality_score,
                    'quality_interpretation': quality_level,
                    'confusion_matrix': {
                        'true_negatives': int(row['tn']),
                        'false_positives': int(row['fp']),
                        'false_negatives': int(row['fn']),
                        'true_positives': int(row['tp'])
                    }
                }
                
                # Reuse already computed feature_stats for high SHAP features
                # High SHAP features are a subset of selected features
                shap_feature_stats = {feat: feature_stats[feat] for feat in high_shap if feat in feature_stats}
                
                cluster_insights['high_shap_features'] = {
                    'features': list(shap_feature_stats.keys()),
                    'feature_statistics': shap_feature_stats
                }
            else:
                # Cluster not found in model summary - use zero scores with worst quality level
                cluster_insights['model_evaluation'] = {
                    'test_auc': 0.0,
                    'balanced_accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'model_quality_score': 0.0,
                    'quality_interpretation': "Poor - Cluster model training was skipped or failed",
                    'confusion_matrix': {
                        'true_negatives': 0,
                        'false_positives': 0,
                        'false_negatives': 0,
                        'true_positives': 0
                    }
                }
                
                cluster_insights['high_shap_features'] = {
                    'features': [],
                    'feature_statistics': {}
                }
        else:
            # No models summary available - use zero scores with worst quality level
            cluster_insights['model_evaluation'] = {
                'test_auc': 0.0,
                'balanced_accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'model_quality_score': 0.0,
                'quality_interpretation': "Poor - Model evaluation data not available",
                'confusion_matrix': {
                    'true_negatives': 0,
                    'false_positives': 0,
                    'false_negatives': 0,
                    'true_positives': 0
                }
            }
            
            cluster_insights['high_shap_features'] = {
                'features': [],
                'feature_statistics': {}
            }
        
        # ===== TRADE-OFF ANALYSIS =====
        # Get trade-off data from trade-off analysis step
        tradeoff_data_per_cluster = tradeoff_result.get('tradeoff_analysis_per_cluster', {})
        cluster_tradeoffs = tradeoff_data_per_cluster.get(cluster_id, [])
        
        if cluster_tradeoffs:
            # Convert to list of dicts with rounded values
            strong_tradeoffs = []
            
            for tradeoff in cluster_tradeoffs:
                # Round numeric values
                if(tradeoff["is_tradeoff"] == 1):
                    strong_tradeoffs.append(tradeoff)
            
            cluster_insights['trade_off_analysis'] = {
                'n_total_tradeoffs': len(cluster_tradeoffs),
                'n_strong_tradeoffs': len(strong_tradeoffs),
                'strong_threshold': -correlation_threshold,
                'strong_tradeoffs': strong_tradeoffs
            }
        
        # ===== HYPERPARAMETER PATTERNS =====
        for hyperparam in hyperparameters:
            if hyperparam in cluster_df.columns:
                value_counts = cluster_df[hyperparam].value_counts()
                # Skip if no values
                if len(value_counts) == 0:
                    continue
                    
                dominant_value = value_counts.index[0]
                dominant_pct = round((value_counts.iloc[0] / n_cluster) * 100, 2)
                
                # Convert dominant_value to string, but handle NaN/None properly
                if pd.isna(dominant_value):
                    dominant_value_str = None
                else:
                    dominant_value_str = str(dominant_value)
                
                # Convert value distribution keys, handling NaN/None
                value_dist = {}
                for k, v in value_counts.items():
                    if pd.isna(k):
                        value_dist[None] = int(v)
                    else:
                        value_dist[str(k)] = int(v)
                
                cluster_insights['hyperparameter_patterns'][hyperparam] = {
                    'dominant_value': dominant_value_str,
                    'dominant_percentage': float(dominant_pct),
                    'unique_values': int(value_counts.shape[0]),
                    'value_distribution': value_dist
                }
        
        # ===== DECISION TREE RULES WITH SCORES =====
        if not rules_summary.empty:
            cluster_rules_filtered = rules_summary[rules_summary['cluster_id'] == cluster_id]
            if not cluster_rules_filtered.empty:
                # Include rules with their scores
                rules_with_scores = []
                for idx, row in cluster_rules_filtered.iterrows():
                    rules_with_scores.append({
                        'rule': row['rule'],
                        'f1_score': float(row['f1_score']),
                        'precision': float(row['precision']),
                        'recall': float(row['recall']),
                        'n_workflows_in_cluster': int(row['n_workflows_in_cluster']),
                        'combined_score': float(row['combined_score']) if 'combined_score' in row and pd.notna(row['combined_score']) else None
                    })
                cluster_insights['decision_tree_rules'] = rules_with_scores
        
        cluster_insights_dict[str(cluster_id)] = cluster_insights
    
    # ===== OPTIONAL: CALL OLLAMA TO NAME CLUSTERS =====
    use_ollama = kwargs.get('use_ollama', False)
    cluster_names = None
    if use_ollama:
        cluster_names = generate_cluster_names_with_ollama(
            cluster_insights_dict,
           
        )
        if cluster_names is not None:
            print("✓ Retrieved cluster names from Ollama")
            # Pretty-print cluster names
            if isinstance(cluster_names, dict):
                print("\nCLUSTER NAMES FROM OLLAMA:")
                for cid, info in cluster_names.items():
                    if isinstance(info, dict):
                        if "cluster_name" in info:
                            name = info["cluster_name"]
                        elif "name" in info:
                            # Backwards compatibility if model still returns the old schema
                            name = info["name"]
                        else:
                            name = str(info)
                    else:
                        name = str(info)
                    print(f"  Cluster {cid}: {name}")
            else:
                print("\nCluster names (raw):", cluster_names)
    # Attach cluster_name into each cluster's metadata, with fallback 'no_name'
    for cid_str, insights in cluster_insights_dict.items():
        cluster_name_value = "no_name"
        if isinstance(cluster_names, dict):
            info = cluster_names.get(cid_str)
            if isinstance(info, dict):
                if "cluster_name" in info:
                    cluster_name_value = str(info["cluster_name"])
                elif "name" in info:
                    # Backwards compatibility: accept 'name' if present
                    cluster_name_value = str(info["name"])
                elif info:
                    cluster_name_value = str(info)
        insights.setdefault("metadata", {})
        insights["metadata"]["cluster_name"] = cluster_name_value
    print("\n✓ Generated comprehensive insights for all clusters.")

    # ===== SAVE COMPREHENSIVE JSON =====
    # json_file = os.path.join(data_folder, 'clusters_comprehensive_insights.json')
    # with open(json_file, 'w') as f:
    #     json.dump(cluster_insights_dict, f, indent=2, cls=NumpyEncoder)
    
    # print(f"\n✓ Saved comprehensive cluster statistics to: {json_file}")
    # print(f"  Clusters included: {list(cluster_insights_dict.keys())}")
    # print("clusterssssssss", cluster_insights_dict)
    
    # Print summary
    print("\n" + "="*80)
    print("COMPREHENSIVE CLUSTER STATISTICS SUMMARY")
    print("="*80)
    for cluster_id, insights in cluster_insights_dict.items():
        # print(f"\nCluster {cluster_id}:")
        # print(f"  Workflows: {insights['metadata']['n_workflows']} ({insights['metadata']['percentage_of_total']}%)")
        if insights['feature_selection']:
            n_selected = insights['feature_selection']['n_features_selected']
            # print(f"  Selected Features: {n_selected}")
            # Show representative features with their categories
            if 'feature_statistics' in insights['feature_selection']:
                # print(f"    Representative features (low/mid/high):")
                feat_stats = insights['feature_selection']['feature_statistics']
                for feat, stats in list(feat_stats.items())[:3]:
                    category = stats['value_category']
                    dist_score = stats['distinctiveness_score']
                    # print(f"      - {feat}: {category.upper()} (distinctiveness: {dist_score:.2f})")
        if insights['correlation_analysis'] and 'n_removed_features' in insights['correlation_analysis']:
            print(f"  Correlation Analysis: {insights['correlation_analysis']['n_removed_features']} removed features analyzed")
        if insights['model_evaluation']:
            print(f"  Quality: {insights['model_evaluation']['quality_interpretation']}")
        if insights['high_shap_features']:
            features_list = insights['high_shap_features'] if isinstance(insights['high_shap_features'], list) else insights['high_shap_features'].get('features', [])
            n_features = len(features_list)
            print(f"  High SHAP Features: {n_features}")
            # Show SHAP features with their categories
            if isinstance(insights['high_shap_features'], dict) and 'feature_statistics' in insights['high_shap_features']:
                print(f"    Key discriminative features (low/mid/high):")
                shap_stats = insights['high_shap_features']['feature_statistics']
                for feat, stats in list(shap_stats.items())[:3]:
                    category = stats['value_category']
                    dist_score = stats['distinctiveness_score']
                    print(f"      - {feat}: {category.upper()} (distinctiveness: {dist_score:.2f})")
        if insights['trade_off_analysis'] and 'n_strong_tradeoffs' in insights['trade_off_analysis']:
            print(f"  Strong Trade-offs: {insights['trade_off_analysis']['n_strong_tradeoffs']}")
        if insights["decision_tree_rules"]:
            print(f"  Decision Tree Rules: {len(insights['decision_tree_rules'])}")
            print(f"    Top Rule: {insights['decision_tree_rules'][0]['rule']}")
    
    return {
        'status': 'completed',
        'cluster_insights': cluster_insights_dict,
        'cluster_names_from_ollama': cluster_names,
        # 'json_file': json_file
    }



def build_default_insights_pipeline():
    """Build the default insights pipeline with all steps."""
    pipeline = InsightsPipeline()
    
    # PHASE 1: Feature Selection & Model Training on Original Data
    pipeline.add_step('step_phase1_load_data', step_phase1_load_data, enabled=True)
    pipeline.add_step('step_phase1_feature_selection', step_phase1_feature_selection, enabled=True)
    pipeline.add_step('step_phase1_tradeoff_analysis', step_phase1_tradeoff_analysis, enabled=True)
    pipeline.add_step('step_phase1_model_training', step_phase1_model_training_and_evaluation, enabled=True)
    pipeline.add_step('step_phase1_decision_tree_rules', step_phase1_decision_tree_rules, enabled=True)
    pipeline.add_step('step_phase1_comprehensive_cluster_insights', step_phase1_comprehensive_cluster_insights, enabled=True)
    
    return pipeline
