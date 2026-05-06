"""Backtest entry point.

Replays the live strategy modules against historical Dukascopy XAUUSD
data with realistic Vantage spread (20-40 points) and slippage (1-3
points). Walk-forward across train (2020-2023) / test (2024-2025) splits
per spec Section 11.2; pass criteria: profit factor > 1.3, max drawdown
< 20 %, win rate > 55 %, ≥ 200 trades.
"""
