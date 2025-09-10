from enum import Enum
from typing import Optional, Union, Literal

import numpy as np
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field, validator
from scipy import stats


class SelectionMethod(str, Enum):
    """Enumeration of available k-selection methods for Hill estimator."""
    GOODNESS_OF_FIT = "gof"
    AMSE_PROXY = "amse"


class HillCurveResult(BaseModel):
    """Results from computing the Hill estimator curve across multiple k values.
    
    Attributes:
        k_grid: Array of k values (number of upper order statistics used).
        hill_gamma: Array of Hill estimator values γ̂_k for each k.
        log_threshold: Array of log-transformed thresholds for each k.
        sorted_data: The input data sorted in ascending order.
        log_data: Log-transformed sorted data.
    """
    k_grid: np.ndarray = Field(..., description="Grid of k values")
    hill_gamma: np.ndarray = Field(..., description="Hill gamma estimates")
    log_threshold: np.ndarray = Field(..., description="Log thresholds")
    sorted_data: np.ndarray = Field(..., description="Sorted input data")
    log_data: np.ndarray = Field(..., description="Log-transformed data")
    
    class Config:
        arbitrary_types_allowed = True


class SelectionResult(BaseModel):
    """Results from k-selection method for Hill estimator.
    
    Attributes:
        k: Optimal number of upper order statistics.
        gamma: Hill estimator γ̂ at optimal k.
        alpha: Tail index α̂ = 1/γ̂ at optimal k.
        selection_criterion: Value of the selection criterion at optimal k.
    """
    k: int = Field(..., description="Optimal k value")
    gamma: float = Field(..., description="Hill gamma estimate at optimal k")
    alpha: float = Field(..., description="Tail index alpha = 1/gamma")
    selection_criterion: float = Field(..., description="Selection criterion value")
    
    class Config:
        arbitrary_types_allowed = True


class GOFSelectionResult(SelectionResult):
    """Results from Goodness-of-Fit k-selection method.
    
    Additional Attributes:
        ks_stat: Kolmogorov-Smirnov test statistic at optimal k.
        ks_stats: Array of KS statistics for all k values.
    """
    ks_stat: float = Field(..., description="KS test statistic at optimal k")
    ks_stats: np.ndarray = Field(..., description="KS statistics for all k")
    
    @validator('selection_criterion')
    def set_selection_criterion(cls, v, values):
        return values.get('ks_stat', v)


class AMSESelectionResult(SelectionResult):
    """Results from AMSE-proxy k-selection method.
    
    Additional Attributes:
        amse_proxy: AMSE proxy value at optimal k.
        scores: Array of AMSE proxy scores for all k values.
        lambda_param: Lambda parameter used for bias-variance tradeoff.
    """
    amse_proxy: float = Field(..., description="AMSE proxy value at optimal k")
    scores: np.ndarray = Field(..., description="AMSE proxy scores for all k")
    lambda_param: float = Field(..., description="Lambda parameter used")
    
    @validator('selection_criterion')
    def set_selection_criterion(cls, v, values):
        return values.get('amse_proxy', v)


class HillEstimatorResult(BaseModel):
    """Complete results from Hill estimator analysis.
    
    Attributes:
        hill_curve: Results from Hill curve computation.
        selection_result: Results from the chosen k-selection method.
        method_used: The selection method that was used.
    """
    hill_curve: HillCurveResult = Field(..., description="Hill curve results")
    selection_result: Union[GOFSelectionResult, AMSESelectionResult] = Field(
        ..., description="K-selection results"
    )
    method_used: SelectionMethod = Field(..., description="Selection method used")
    
    class Config:
        arbitrary_types_allowed = True


def _validate_and_clean_data(x: Union[np.ndarray, list]) -> np.ndarray:
    """Validate and clean input data for Hill estimator.
    
    Converts input to 1-dimensional numpy array, removes NaN values,
    and validates that all values are positive.
    
    Args:
        x: Input data array or list.
        
    Returns:
        Cleaned 1-dimensional numpy array with NaN values removed.
        
    Raises:
        ValueError: If input is not 1-dimensional, contains no finite values,
                   or contains non-positive values after cleaning.
    """
    # Convert to numpy array
    x_array = np.asarray(x, dtype=float)
    
    # Ensure 1-dimensional
    if x_array.ndim != 1:
        raise ValueError(f"Input must be 1-dimensional, got shape {x_array.shape}")
    
    # Remove NaN values
    x_clean = x_array[np.isfinite(x_array)]
    
    # Check if any data remains
    if len(x_clean) == 0:
        raise ValueError("No finite values found in input data")
    
    # Check for positive values
    if np.any(x_clean <= 0):
        raise ValueError("All observations must be positive for Hill estimation")
    
    return x_clean


def compute_hill_curve(
    x: Union[np.ndarray, list], 
    k_min: int = 10, 
    k_max: Optional[int] = None
) -> HillCurveResult:
    """Compute the Hill estimator curve for a range of k values.
    
    The Hill estimator for the tail index parameter γ at order k is given by:
    
    $$\\hat{\\gamma}_k = \\frac{1}{k} \\sum_{i=1}^k \\log X_{(n-i+1)} - \\log X_{(n-k)}$$
    
    where X_{(1)} ≤ ... ≤ X_{(n)} are the order statistics.
    
    Args:
        x: Positive observations for extreme value analysis. Must be 1-dimensional.
           NaN values are automatically removed.
        k_min: Minimum number of upper order statistics to use. Must be ≥ 1.
        k_max: Maximum number of upper order statistics to use. Defaults to n-1
               where n is the sample size after cleaning.
    
    Returns:
        HillCurveResult containing:
            - k_grid: Array of k values used
            - hill_gamma: Hill estimator values γ̂_k for each k
            - log_threshold: Log-transformed thresholds log(X_{(n-k)}) for each k
            - sorted_data: Input data sorted in ascending order
            - log_data: Log-transformed sorted data
    
    Raises:
        ValueError: If input validation fails, insufficient data points, or
                   invalid k_min/k_max values.
        
    Note:
        The Hill estimator assumes a Pareto-type tail: P(X > x) ~ L(x) x^{-1/γ}
        where L is slowly varying and γ > 0 is the tail index parameter.
    """
    # Validate and clean input data
    x_clean = _validate_and_clean_data(x)
    n = len(x_clean)
    
    # Validate minimum sample size
    if n < 3:
        raise ValueError("Need at least 3 observations after cleaning.")
    
    # Set default k_max and validate parameters
    if k_max is None:
        k_max = n - 1
    k_max = int(min(k_max, n - 1))
    k_min = int(max(k_min, 1))
    
    if k_min >= k_max:
        raise ValueError("k_min must be < k_max and < n-1.")

    # Sort data and compute log-transform
    xs = np.sort(x_clean)
    logs = np.log(xs)
    
    # Compute Hill estimator for k = 1, ..., k_max
    # Using efficient vectorized computation of cumulative sums
    top_logs = logs[-k_max:]
    s_k = np.cumsum(top_logs[::-1])  # Cumulative sum from largest to smallest
    log_u = logs[n - 1 - np.arange(k_max)]  # Log thresholds
    ks = np.arange(1, k_max + 1, dtype=float)
    gamma_all = s_k / ks - log_u

    # Filter to requested k range
    mask = (ks >= k_min)
    k_grid = ks[mask].astype(int)
    gamma_grid = gamma_all[mask]
    log_u_grid = log_u[mask]

    return HillCurveResult(
        k_grid=k_grid,
        hill_gamma=gamma_grid,
        log_threshold=log_u_grid,
        sorted_data=xs,
        log_data=logs
    )

def select_k_gof(hill_result: HillCurveResult) -> GOFSelectionResult:
    """Select optimal k using Goodness-of-Fit to exponential distribution.
    
    This method is based on the fact that for a Pareto distribution with tail
    index γ, the normalized log-exceedances follow an exponential distribution:
    
    $$Z_i = \\frac{Y_i}{\\bar{Y}} \\sim \\text{Exp}(1)$$
    
    where $Y_i = \\log X_{(n-i+1)} - \\log X_{(n-k)}$ are log-exceedances and
    $\\bar{Y}$ is their sample mean.
    
    Args:
        hill_result: Results from compute_hill_curve containing Hill curve data.
    
    Returns:
        GOFSelectionResult containing:
            - k: Optimal number of upper order statistics
            - gamma: Hill estimator γ̂ at optimal k  
            - alpha: Tail index α̂ = 1/γ̂ at optimal k
            - ks_stat: Kolmogorov-Smirnov test statistic at optimal k
            - ks_stats: KS statistics for all k values tested
            - selection_criterion: Same as ks_stat
            
    Note:
        The optimal k minimizes the KS test statistic for exponentiality of
        the normalized log-exceedances, indicating best fit to the asymptotic
        distribution under the Pareto model.
    """
    k_grid = hill_result.k_grid
    gamma_grid = hill_result.hill_gamma
    logs = hill_result.log_data
    n = len(logs)
    
    # Compute KS test statistic for each k
    ks_stats = np.empty_like(k_grid, dtype=float)
    for j, k in enumerate(k_grid):
        # Extract log-exceedances: Y_i = log(X_{n-i+1}) - log(X_{n-k})
        y = logs[n - np.arange(1, k + 1)] - logs[n - k]
        # Normalize: Z_i = Y_i / mean(Y)
        z = y / y.mean()
        # Test for exponential(1) distribution
        ks_stats[j] = stats.kstest(z, 'expon').statistic
    
    # Select k that minimizes KS statistic
    idx_optimal = np.argmin(ks_stats)
    k_optimal = k_grid[idx_optimal]
    gamma_optimal = gamma_grid[idx_optimal]
    alpha_optimal = 1.0 / gamma_optimal
    
    return GOFSelectionResult(
        k=int(k_optimal),
        gamma=float(gamma_optimal),
        alpha=float(alpha_optimal),
        ks_stat=float(ks_stats[idx_optimal]),
        ks_stats=ks_stats,
        selection_criterion=float(ks_stats[idx_optimal])
    )

def select_k_amse(
    hill_result: HillCurveResult, 
    lam: float = 2.0
) -> AMSESelectionResult:
    """Select optimal k using Asymptotic Mean Squared Error (AMSE) proxy method.
    
    This method balances bias and variance by minimizing a proxy for AMSE:
    
    $$\\text{AMSE-proxy}(k) = \\frac{\\hat{\\gamma}_k^2}{k} + \\lambda (\\hat{\\gamma}_k - \\hat{\\gamma}_{k/2})^2$$
    
    where the first term approximates variance and the second term approximates 
    squared bias, with λ controlling the bias-variance tradeoff.
    
    Args:
        hill_result: Results from compute_hill_curve containing Hill curve data.
        lam: Lambda parameter controlling bias-variance tradeoff. Higher values
             give more weight to bias reduction. Defaults to 2.0.
    
    Returns:
        AMSESelectionResult containing:
            - k: Optimal number of upper order statistics
            - gamma: Hill estimator γ̂ at optimal k
            - alpha: Tail index α̂ = 1/γ̂ at optimal k
            - amse_proxy: AMSE proxy value at optimal k
            - scores: AMSE proxy scores for all k values
            - lambda_param: Lambda parameter used
            - selection_criterion: Same as amse_proxy
            
    Raises:
        RuntimeError: If AMSE proxy cannot be evaluated for any k value.
        
    Note:
        The method requires k/2 to also be in the k_grid for bias estimation,
        which may limit the effective range of k values that can be evaluated.
    """
    k_grid = hill_result.k_grid
    gamma_grid = hill_result.hill_gamma
    k_min = k_grid[0]
    
    # Create lookup map for gamma values
    gamma_map = {int(k): float(g) for k, g in zip(k_grid, gamma_grid)}
    scores = np.full_like(k_grid, np.nan, dtype=float)
    
    # Compute AMSE proxy for each k
    for j, k in enumerate(k_grid):
        k2 = k // 2
        # Skip if k/2 is not available or too small
        if k2 < k_min or (k2 not in gamma_map):
            continue
            
        gk = gamma_map[k]
        g2 = gamma_map[k2]
        
        # Variance proxy: γ²/k
        var_term = (gk ** 2) / k
        # Bias² proxy: (γ_k - γ_{k/2})²
        bias2 = (gk - g2) ** 2
        # Combined AMSE proxy
        scores[j] = var_term + lam * bias2
    
    # Check if any scores are computable
    finite = np.isfinite(scores)
    if not np.any(finite):
        raise RuntimeError(
            "AMSE-proxy could not be evaluated on the k-grid; "
            "try smaller k_min or denser k_grid."
        )
    
    # Select k that minimizes AMSE proxy
    idx_optimal = np.nanargmin(scores)
    k_optimal = k_grid[idx_optimal]
    gamma_optimal = gamma_grid[idx_optimal]
    alpha_optimal = 1.0 / gamma_optimal
    
    return AMSESelectionResult(
        k=int(k_optimal),
        gamma=float(gamma_optimal),
        alpha=float(alpha_optimal),
        amse_proxy=float(scores[idx_optimal]),
        scores=scores,
        lambda_param=lam,
        selection_criterion=float(scores[idx_optimal])
    )

def hill_estimator(
    x: Union[np.ndarray, list],
    method: SelectionMethod = SelectionMethod.GOODNESS_OF_FIT,
    k_min: int = 10,
    k_max: Optional[int] = None,
    lam: float = 2.0
) -> HillEstimatorResult:
    """Compute Hill estimator with automatic k-selection.
    
    This is the main function for Hill tail index estimation. It computes the
    Hill estimator curve over a range of k values and selects the optimal k
    using the specified method.
    
    Args:
        x: Positive observations for extreme value analysis. Must be 1-dimensional.
           NaN values are automatically removed.
        method: Selection method to use for choosing optimal k.
                Options: 'gof' (Goodness-of-Fit) or 'amse' (AMSE-proxy).
        k_min: Minimum number of upper order statistics to use. Must be ≥ 1.
        k_max: Maximum number of upper order statistics to use. Defaults to n-1
               where n is the sample size after cleaning.
        lam: Lambda parameter for AMSE method controlling bias-variance tradeoff.
             Only used when method='amse'. Defaults to 2.0.
    
    Returns:
        HillEstimatorResult containing:
            - hill_curve: Complete Hill curve results
            - selection_result: Results from the chosen k-selection method
            - method_used: The selection method that was applied
    
    Raises:
        ValueError: If input validation fails or invalid parameters.
        RuntimeError: If k-selection method fails to find optimal k.
        
    Example:
        >>> import numpy as np
        >>> # Generate Pareto data with tail index 0.5 (alpha = 2)
        >>> np.random.seed(42)
        >>> data = np.random.pareto(0.5, 1000) + 1
        >>> 
        >>> # Estimate using Goodness-of-Fit method
        >>> result = hill_estimator(data, method='gof')
        >>> print(f"Estimated alpha: {result.selection_result.alpha:.3f}")
        >>> 
        >>> # Estimate using AMSE method with custom lambda
        >>> result_amse = hill_estimator(data, method='amse', lam=1.5)
        >>> print(f"Estimated alpha (AMSE): {result_amse.selection_result.alpha:.3f}")
    """
    # Compute Hill curve
    hill_curve = compute_hill_curve(x, k_min, k_max)
    
    # Apply selected k-selection method
    if method == SelectionMethod.GOODNESS_OF_FIT:
        selection_result = select_k_gof(hill_curve)
    elif method == SelectionMethod.AMSE_PROXY:
        selection_result = select_k_amse(hill_curve, lam)
    else:
        raise ValueError(f"Unknown selection method: {method}")
    
    return HillEstimatorResult(
        hill_curve=hill_curve,
        selection_result=selection_result,
        method_used=method
    )


# Deprecated function for backward compatibility
def hill_estimator_with_selection(x, k_min=10, k_max=None, lam=2.0):
    """Deprecated: Use hill_estimator() instead.
    
    Compute Hill estimator with both k selection methods.
    This function is deprecated and maintained only for backward compatibility.
    Use hill_estimator() with method parameter instead.
    """
    import warnings
    warnings.warn(
        "hill_estimator_with_selection() is deprecated. "
        "Use hill_estimator() with method parameter instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    hill_result = compute_hill_curve(x, k_min, k_max)
    method_A = select_k_gof(hill_result)
    method_B = select_k_amse(hill_result, lam)
    
    return {
        "k_grid": hill_result.k_grid,
        "hill_gamma": hill_result.hill_gamma,
        "log_threshold": hill_result.log_threshold,
        "sorted_data": hill_result.sorted_data,
        "log_data": hill_result.log_data,
        "method_A": {
            "k": method_A.k,
            "gamma": method_A.gamma,
            "alpha": method_A.alpha,
            "ks_stat": method_A.ks_stat,
            "ks_stats": method_A.ks_stats
        },
        "method_B": {
            "k": method_B.k,
            "gamma": method_B.gamma,
            "alpha": method_B.alpha,
            "amse_proxy": method_B.amse_proxy,
            "scores": method_B.scores
        }
    }


def plot_log_survival(
    x: Union[np.ndarray, list],
    result: HillEstimatorResult,
    title: str = "Log Survival Function",
    ax: Optional[plt.Axes] = None,
    show_empirical: bool = True,
    show_fit: bool = True,
    empirical_kwargs: Optional[dict] = None,
    fit_kwargs: Optional[dict] = None
) -> plt.Axes:
    """Plot log survival function with Hill estimator tail fit.
    
    Creates a log-log plot of the empirical survival function P(X > x) vs x,
    along with the fitted tail line based on the Hill estimator results.
    
    The tail fit is based on the power law relationship:
    $$P(X > x) \\approx \\frac{k}{n} \\left(\\frac{x}{u}\\right)^{-\\alpha}$$
    
    where u is the threshold, k is the number of exceedances, n is sample size,
    and α is the tail index.
    
    Args:
        x: Original data used for Hill estimation. Must be 1-dimensional.
           NaN values are automatically handled.
        result: Results from hill_estimator containing selection results.
        title: Plot title. Defaults to "Log Survival Function".
        ax: Matplotlib axes to plot on. If None, creates new figure.
        show_empirical: Whether to show empirical survival function points.
        show_fit: Whether to show fitted tail line.
        empirical_kwargs: Additional arguments for empirical survival plot.
        fit_kwargs: Additional arguments for fitted line plot.
        
    Returns:
        The matplotlib axes object containing the plot.
        
    Example:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> 
        >>> # Generate Pareto data
        >>> np.random.seed(42)
        >>> data = np.random.pareto(0.5, 1000) + 1
        >>> 
        >>> # Estimate Hill parameters
        >>> result = hill_estimator(data, method='gof')
        >>> 
        >>> # Plot log survival function
        >>> fig, ax = plt.subplots(figsize=(10, 6))
        >>> plot_log_survival(data, result, "Pareto Data Tail Analysis", ax=ax)
        >>> plt.show()
        
    Note:
        The function automatically handles data cleaning and uses the same
        preprocessing as the Hill estimator. The fitted line is extrapolated
        beyond the threshold for visualization purposes.
    """
    # Validate and clean data using the same function as Hill estimator
    x_clean = _validate_and_clean_data(x)
    
    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set default plot arguments
    if empirical_kwargs is None:
        empirical_kwargs = {'alpha': 0.6, 's': 1, 'color': 'blue', 'label': 'Empirical'}
    if fit_kwargs is None:
        fit_kwargs = {'linewidth': 2, 'alpha': 0.8}
    
    # Sort data
    sorted_data = np.sort(x_clean)
    n = len(sorted_data)
    
    if show_empirical:
        # Empirical survival function P(X > x) = (number of values > x) / n
        survival_probs = (np.arange(n, 0, -1)) / n
        
        # Remove zeros for log transformation
        mask = (sorted_data > 0) & (survival_probs > 0)
        log_x = np.log(sorted_data[mask])
        log_survival = np.log(survival_probs[mask])
        
        # Plot empirical survival
        ax.scatter(log_x, log_survival, **empirical_kwargs)
    
    if show_fit:
        # Extract parameters from Hill estimator result
        selection_result = result.selection_result
        k = selection_result.k
        alpha = selection_result.alpha
        
        # Get threshold from Hill curve results
        hill_curve = result.hill_curve
        sorted_orig = hill_curve.sorted_data
        n_orig = len(sorted_orig)
        
        # Threshold is the (n-k)th order statistic
        x_thresh = sorted_orig[n_orig - k]
        log_x_thresh = np.log(x_thresh)
        
        # Survival probability at threshold
        survival_thresh = k / n_orig
        log_survival_thresh = np.log(survival_thresh)
        
        # Create extended x range for fitted line
        log_x_max = np.log(sorted_data[-1])
        log_x_range = np.linspace(log_x_thresh, log_x_max + 0.5, 100)
        
        # Fitted line: log P(X > x) = log(k/n) - α*(log(x) - log(x_thresh))
        log_survival_fit = log_survival_thresh - alpha * (log_x_range - log_x_thresh)
        
        # Determine method-specific styling
        method_name = result.method_used.value
        if method_name == "gof":
            color = 'red'
            method_label = 'GOF'
        else:
            color = 'green'
            method_label = 'AMSE'
        
        # Plot fitted line
        fit_label = f'{method_label}: α={alpha:.3f}, k={k}'
        ax.plot(log_x_range, log_survival_fit, color=color, 
                label=fit_label, **fit_kwargs)
        
        # Mark threshold point
        ax.scatter([log_x_thresh], [log_survival_thresh], 
                   color=color, s=100, marker='o', zorder=5,
                   label=f'Threshold: x={x_thresh:.3f}')
    
    # Formatting
    ax.set_xlabel('log(x)')
    ax.set_ylabel('log P(X > x)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_hill_diagnostics(
    result: HillEstimatorResult,
    title_prefix: str = "",
    figsize: tuple = (15, 10)
) -> plt.Figure:
    """Plot comprehensive Hill estimator diagnostics.
    
    Creates a multi-panel figure showing:
    1. Hill estimator curve (γ vs k)
    2. Tail index curve (α vs k)  
    3. Selection criterion (KS stat or AMSE proxy vs k)
    4. Log survival function with fit
    
    Args:
        result: Results from hill_estimator containing all diagnostic information.
        title_prefix: Prefix for subplot titles. Defaults to empty string.
        figsize: Figure size as (width, height). Defaults to (15, 10).
        
    Returns:
        The matplotlib figure object containing all diagnostic plots.
        
    Example:
        >>> # Generate and analyze data
        >>> data = np.random.pareto(0.5, 1000) + 1
        >>> result = hill_estimator(data, method='gof')
        >>> 
        >>> # Create diagnostic plots
        >>> fig = plot_hill_diagnostics(result, "Pareto Analysis")
        >>> plt.show()
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # Extract data
    hill_curve = result.hill_curve
    selection_result = result.selection_result
    method_name = result.method_used.value
    
    k_grid = hill_curve.k_grid
    gamma_grid = hill_curve.hill_gamma
    optimal_k = selection_result.k
    
    # Determine method-specific colors and labels
    if method_name == "gof":
        color = 'red'
        method_label = 'GOF'
        criterion_data = selection_result.ks_stats
        criterion_label = 'KS Statistic'
    else:
        color = 'green'
        method_label = 'AMSE'
        criterion_data = selection_result.scores
        criterion_label = 'AMSE Proxy Score'
    
    # 1. Hill estimator curve
    ax1.plot(k_grid, gamma_grid, 'b-', alpha=0.7, linewidth=2)
    ax1.axvline(optimal_k, color=color, linestyle='--', 
                label=f'{method_label}: k={optimal_k}')
    ax1.set_xlabel('k')
    ax1.set_ylabel('γ (Hill estimator)')
    ax1.set_title(f'{title_prefix} Hill Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Tail index curve
    alpha_grid = 1.0 / gamma_grid
    ax2.plot(k_grid, alpha_grid, 'b-', alpha=0.7, linewidth=2)
    ax2.axvline(optimal_k, color=color, linestyle='--', 
                label=f'{method_label}: α={selection_result.alpha:.3f}')
    ax2.set_xlabel('k')
    ax2.set_ylabel('α (Tail index)')
    ax2.set_title(f'{title_prefix} Tail Index (α = 1/γ)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Selection criterion
    if method_name == "gof":
        ax3.plot(k_grid, criterion_data, color, linewidth=2)
    else:
        # For AMSE, only plot finite values
        finite_mask = np.isfinite(criterion_data)
        ax3.plot(k_grid[finite_mask], criterion_data[finite_mask], 
                 color, linewidth=2)
    
    ax3.axvline(optimal_k, color=color, linestyle='--', alpha=0.8,
                label=f'Optimal k={optimal_k}')
    ax3.set_xlabel('k')
    ax3.set_ylabel(criterion_label)
    ax3.set_title(f'{title_prefix} {criterion_label}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Log survival function - requires original data
    # This subplot will be empty as we don't have original data in result
    # User should call plot_log_survival separately with original data
    ax4.text(0.5, 0.5, f'Use plot_log_survival(x, result)\nto display survival function',
             transform=ax4.transAxes, ha='center', va='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    ax4.set_title(f'{title_prefix} Log Survival Function')
    ax4.set_xlabel('log(x)')
    ax4.set_ylabel('log P(X > x)')
    
    plt.tight_layout()
    return fig

