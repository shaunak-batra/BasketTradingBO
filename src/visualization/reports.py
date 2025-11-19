"""
Module: Report Generator

Generate comprehensive HTML reports for backtest results.

Functions
---------
generate_backtest_report
    Generate complete HTML report with metrics and plots

Author: Quantitative Research Team
Created: 2025-01-18
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.backtesting.backtester import BacktestResult
from src.utils.logger import StructuredLogger
from src.visualization.plots import (
    plot_drawdown,
    plot_portfolio_performance,
    plot_spread_zscore,
)


def generate_backtest_report(
    result: BacktestResult,
    spread: pd.Series,
    zscore: pd.Series,
    parameters: Dict,
    output_dir: str = "results/reports",
    use_timestamp: bool = True,
) -> str:
    """
    Generate comprehensive HTML backtest report.

    Parameters
    ----------
    result : BacktestResult
        Backtest results
    spread : pd.Series
        Spread time series
    zscore : pd.Series
        Z-score time series
    parameters : Dict
        Backtest parameters
    output_dir : str
        Directory to save report
    use_timestamp : bool
        If True, creates timestamped folders (default). If False, uses output_dir directly.

    Returns
    -------
    str
        Path to generated report

    Examples
    --------
    >>> report_path = generate_backtest_report(result, spread, zscore, params)
    """
    logger = StructuredLogger(__name__)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate timestamp if needed
    if use_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"backtest_report_{timestamp}.html"
        report_path = output_path / report_name
        plot_dir = output_path / f"plots_{timestamp}"
        plot_dir.mkdir(parents=True, exist_ok=True)
        img_src_prefix = f"plots_{timestamp}"  # For HTML img src paths
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = "backtest_report.html"
        report_path = output_path.parent / report_name
        plot_dir = output_path  # Use the provided output_dir as plot directory
        img_src_prefix = "plots"  # For HTML img src paths

    performance_plot = str(plot_dir / "performance.png")
    spread_plot = str(plot_dir / "spread_zscore.png")
    drawdown_plot = str(plot_dir / "drawdown.png")

    plot_portfolio_performance(
        result.portfolio_value,
        result.returns,
        result.signals,
        save_path=performance_plot,
    )

    plot_spread_zscore(
        spread,
        zscore,
        entry_threshold=parameters.get("entry_threshold", 2.0),
        exit_threshold=parameters.get("exit_threshold", 0.5),
        signals=result.signals,
        save_path=spread_plot,
    )

    plot_drawdown(result.portfolio_value, save_path=drawdown_plot)

    # Generate HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Backtest Report - {timestamp}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: auto;
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2E86AB;
                border-bottom: 3px solid #2E86AB;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #06A77D;
                margin-top: 30px;
                border-bottom: 2px solid #06A77D;
                padding-bottom: 5px;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 20px;
                margin: 20px 0;
            }}
            .metric-box {{
                background: #f9f9f9;
                padding: 15px;
                border-radius: 8px;
                border-left: 4px solid #2E86AB;
            }}
            .metric-label {{
                font-size: 12px;
                color: #666;
                text-transform: uppercase;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #333;
                margin-top: 5px;
            }}
            .positive {{
                color: #06A77D;
            }}
            .negative {{
                color: #D62828;
            }}
            img {{
                max-width: 100%;
                height: auto;
                margin: 20px 0;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #2E86AB;
                color: white;
            }}
            .footer {{
                margin-top: 40px;
                text-align: center;
                color: #666;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Basket Trading Backtest Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

            <h2>Performance Metrics</h2>
            <div class="metrics-grid">
                <div class="metric-box">
                    <div class="metric-label">Total Return</div>
                    <div class="metric-value {'positive' if result.metrics.total_return > 0 else 'negative'}">
                        {result.metrics.total_return*100:.2f}%
                    </div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Annualized Return</div>
                    <div class="metric-value {'positive' if result.metrics.annualized_return > 0 else 'negative'}">
                        {result.metrics.annualized_return*100:.2f}%
                    </div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Sharpe Ratio</div>
                    <div class="metric-value">
                        {result.metrics.sharpe_ratio:.2f}
                    </div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Sortino Ratio</div>
                    <div class="metric-value">
                        {result.metrics.sortino_ratio:.2f}
                    </div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-value negative">
                        {result.metrics.max_drawdown*100:.2f}%
                    </div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Calmar Ratio</div>
                    <div class="metric-value">
                        {result.metrics.calmar_ratio:.2f}
                    </div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Win Rate</div>
                    <div class="metric-value">
                        {result.metrics.win_rate*100:.2f}%
                    </div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Profit Factor</div>
                    <div class="metric-value">
                        {result.metrics.profit_factor:.2f}
                    </div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Number of Trades</div>
                    <div class="metric-value">
                        {result.metrics.num_trades}
                    </div>
                </div>
            </div>

            <h2>Strategy Parameters</h2>
            <table>
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                </tr>
                {"".join([f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in parameters.items()])}
            </table>

            <h2>Portfolio Performance</h2>
            <img src="{img_src_prefix}/performance.png" alt="Portfolio Performance">

            <h2>Spread and Z-Score</h2>
            <img src="{img_src_prefix}/spread_zscore.png" alt="Spread Z-Score">

            <h2>Drawdown Analysis</h2>
            <img src="{img_src_prefix}/drawdown.png" alt="Drawdown">

            <h2>Transaction Costs</h2>
            <div class="metrics-grid">
                <div class="metric-box">
                    <div class="metric-label">Total Transaction Costs</div>
                    <div class="metric-value negative">
                        ${result.transaction_costs.sum():,.2f}
                    </div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Average Cost per Trade</div>
                    <div class="metric-value">
                        ${result.transaction_costs.sum() / len(result.trades) if len(result.trades) > 0 else 0:,.2f}
                    </div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Cost as % of Returns</div>
                    <div class="metric-value">
                        {(result.transaction_costs.sum() / abs(result.portfolio_value.iloc[-1] - result.portfolio_value.iloc[0]) * 100) if result.portfolio_value.iloc[-1] != result.portfolio_value.iloc[0] else 0:.2f}%
                    </div>
                </div>
            </div>

            <div class="footer">
                <p>Generated by Basket Trading with Bayesian Optimization System - {datetime.now().year}</p>
            </div>
        </div>
    </body>
    </html>
    """

    # Write HTML file
    with open(report_path, "w") as f:
        f.write(html_content)

    logger.info("Backtest report generated", path=str(report_path))

    return str(report_path)
