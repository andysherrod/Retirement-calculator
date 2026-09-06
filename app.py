from flask import Flask, render_template, request, jsonify
import os
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.ticker
import base64
import io
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@dataclass
class InvestmentStream:
    """Represents a single investment stream with tax implications"""
    name: str
    annual_contribution: float
    tax_treatment: str  # 'traditional', 'roth', 'taxable'
    contribution_limit: float = float('inf')
    employer_match_rate: float = 0.0
    employer_match_limit: float = 0.0

class EnhancedRetirementCalculator:
    def __init__(self, age_current, age_retire, life_expectancy, monthly_benefit_income,
                 portfolio_total, yearly_investment, years_contributing, current_monthly_budget,
                 inflation_rate, expected_return, account_type, tax_rate, retirement_budget_ratio,
                 retirement_phases, healthcare_costs,
                 portfolio_volatility=0.16, investment_streams=None,
                 random_seed: Optional[int] = None, num_simulations: int = 10000):

        # Basic validations
        if age_retire <= age_current:
            raise ValueError("age_retire must be greater than age_current")
        self.age_current = age_current
        self.age_retire = age_retire
        self.life_expectancy = life_expectancy
        self.monthly_benefit_income = monthly_benefit_income
        self.portfolio_total = portfolio_total
        self.yearly_investment = yearly_investment
        self.years_contributing = years_contributing
        self.current_monthly_budget = current_monthly_budget
        self.inflation_rate = inflation_rate
        self.expected_return = expected_return
        self.account_type = account_type
        self.tax_rate = tax_rate
        self.retirement_budget_ratio = retirement_budget_ratio
        self.retirement_phases = retirement_phases
        self.healthcare_costs = healthcare_costs

        self.portfolio_volatility = portfolio_volatility

        # Randomness and simulation config
        self.rng = np.random.default_rng(random_seed)
        self.num_simulations = int(num_simulations)

        # Multiple investment streams
        self.investment_streams = investment_streams or []

    def generate_portfolio_returns(self, num_simulations: int, num_years: int) -> np.ndarray:
        """Generate annual portfolio returns directly from portfolio assumptions."""
        return self.rng.normal(
            self.expected_return,
            self.portfolio_volatility,
            size=(num_simulations, num_years)
        )

    def process_investment_streams(self, year: int) -> Dict[str, float]:
        """Process multiple investment streams with tax implications"""
        total_contributions = 0
        tax_deductions = 0
        roth_contributions = 0
        taxable_contributions = 0

        for stream in self.investment_streams:
            # Apply contribution limits
            actual_contribution = min(stream.annual_contribution, stream.contribution_limit)

            # Add employer match if applicable
            employer_match = min(actual_contribution * stream.employer_match_rate,
                                 stream.employer_match_limit)

            total_contribution = actual_contribution + employer_match
            total_contributions += total_contribution

            # Track tax implications
            if stream.tax_treatment == 'traditional':
                tax_deductions += actual_contribution
            elif stream.tax_treatment == 'roth':
                roth_contributions += actual_contribution
            else:  # taxable
                taxable_contributions += actual_contribution

        # If no streams defined, use the original yearly investment
        if not self.investment_streams:
            total_contributions = self.yearly_investment

        return {
            'total_contributions': total_contributions,
            'tax_deductions': tax_deductions,
            'roth_contributions': roth_contributions,
            'taxable_contributions': taxable_contributions
        }

    def calculate(self):
        retirement_years = self.life_expectancy - self.age_retire
        years_until_retirement = self.age_retire - self.age_current
        years_contributing = min(self.years_contributing, years_until_retirement)
        buckets = self.initial_bucket_balances()

        pre_retirement_df = pd.DataFrame(index=range(years_until_retirement + 1))
        pre_retirement_df.loc[0, 'Age'] = self.age_current
        for bucket in ('roth', 'traditional', 'taxable'):
            pre_retirement_df.loc[0, f'{bucket.title()}_Balance'] = buckets[bucket]
        pre_retirement_df.loc[0, 'Taxable_Basis'] = buckets['taxable_basis']
        pre_retirement_df.loc[0, 'Portfolio'] = sum(buckets[key] for key in ('roth', 'traditional', 'taxable'))
        pre_retirement_df.loc[0, 'Yearly_Contribution'] = 0

        # Process investment streams during accumulation phase
        for year in range(1, years_until_retirement + 1):
            pre_retirement_df.loc[year, 'Age'] = self.age_current + year

            if year <= years_contributing:
                if self.investment_streams:
                    stream_data = self.calculate_tax_efficient_contributions(year)
                    contribution = stream_data['total_contributions']
                    pre_retirement_df.loc[year, 'Tax_Savings'] = stream_data['tax_savings']
                else:
                    contribution = self.yearly_investment
                    pre_retirement_df.loc[year, 'Tax_Savings'] = 0
            else:
                contribution = 0
                pre_retirement_df.loc[year, 'Tax_Savings'] = 0

            pre_retirement_df.loc[year, 'Yearly_Contribution'] = contribution
            yearly_buckets = self.contribution_buckets(
                stream_data if year <= years_contributing and self.investment_streams else None
            )

            portfolio_return = self.expected_return
            for bucket in ('roth', 'traditional', 'taxable'):
                buckets[bucket] = (buckets[bucket] + yearly_buckets[bucket]) * (1 + portfolio_return)
            buckets['taxable_basis'] += yearly_buckets['taxable']
            for bucket in ('roth', 'traditional', 'taxable'):
                pre_retirement_df.loc[year, f'{bucket.title()}_Balance'] = buckets[bucket]
            pre_retirement_df.loc[year, 'Taxable_Basis'] = buckets['taxable_basis']
            pre_retirement_df.loc[year, 'Portfolio'] = sum(buckets[key] for key in ('roth', 'traditional', 'taxable'))

        future_portfolio = pre_retirement_df.loc[years_until_retirement, 'Portfolio']
        retirement_start_buckets = buckets.copy()

        # Run Monte Carlo for pre-retirement (accumulation) phase
        pre_retirement_contributions = pre_retirement_df.loc[1:years_until_retirement, 'Yearly_Contribution'].values
        pre_retirement_median_path = self.monte_carlo_pre_retirement_vectorized(
            self.portfolio_total, years_until_retirement, pre_retirement_contributions
        )

        # Retirement phase calculations remain similar but with enhanced Monte Carlo
        df = pd.DataFrame(index=range(retirement_years + 1))
        df.loc[0, 'Age'] = self.age_retire
        df.loc[0, 'Portfolio'] = future_portfolio
        for bucket in ('roth', 'traditional', 'taxable'):
            df.loc[0, f'{bucket.title()}_Balance'] = buckets[bucket]
        df.loc[0, 'Taxable_Basis'] = buckets['taxable_basis']
        df.loc[0, 'Annual_Benefit'] = self.monthly_benefit_income * 12

        initial_retirement_budget = (self.current_monthly_budget * 12 *
                                     self.retirement_budget_ratio *
                                     (1 + self.inflation_rate) ** years_until_retirement)
        df.loc[0, 'Annual_Budget'] = initial_retirement_budget
        df.loc[0, 'Healthcare_Costs'] = (self.healthcare_costs * 12 *
                                         (1 + self.inflation_rate) ** years_until_retirement)

        phase_adjustments = {
            'Constant': [1.0] * retirement_years,
            'Early Active': [1.2] * min(10, retirement_years) + [0.9] * max(0, retirement_years - 10),
            'Late Increase': [0.9] * min(20, retirement_years) + [1.3] * max(0, retirement_years - 20)
        }
        if self.retirement_phases not in phase_adjustments:
            raise ValueError(f"Invalid retirement_phases: {self.retirement_phases}. Valid options: {list(phase_adjustments.keys())}")
        spending_adjustments = phase_adjustments[self.retirement_phases]

        for year in range(1, retirement_years + 1):
            df.loc[year, 'Age'] = self.age_retire + year
            df.loc[year, 'Annual_Benefit'] = df.loc[year - 1, 'Annual_Benefit'] * (1 + self.inflation_rate)

            phase_factor = spending_adjustments[year - 1] if year - 1 < len(spending_adjustments) else spending_adjustments[-1]
            df.loc[year, 'Annual_Budget'] = df.loc[0, 'Annual_Budget'] * (1 + self.inflation_rate) ** year * phase_factor
            df.loc[year, 'Healthcare_Costs'] = df.loc[year - 1, 'Healthcare_Costs'] * (1 + self.inflation_rate + 0.02)

            total_expenses = df.loc[year, 'Annual_Budget'] + df.loc[year, 'Healthcare_Costs']
            cash_needed = max(0, total_expenses - df.loc[year, 'Annual_Benefit'])
            withdrawal_data = self.withdraw_from_buckets(buckets, cash_needed)
            df.loc[year, 'Withdrawal'] = withdrawal_data['withdrawal']
            df.loc[year, 'Tax_Paid'] = withdrawal_data['tax_paid']
            for bucket in ('taxable', 'traditional', 'roth'):
                df.loc[year, f'{bucket.title()}_Withdrawal'] = withdrawal_data[f'{bucket}_withdrawal']
            df.loc[year, 'Withdrawal_Rate'] = (
                df.loc[year, 'Withdrawal'] / df.loc[year - 1, 'Portfolio']
                if df.loc[year - 1, 'Portfolio'] > 0 else 0
            )

            for bucket in ('roth', 'traditional', 'taxable'):
                buckets[bucket] = max(0, buckets[bucket] * (1 + self.expected_return))
                df.loc[year, f'{bucket.title()}_Balance'] = buckets[bucket]
            df.loc[year, 'Taxable_Basis'] = buckets['taxable_basis']
            df.loc[year, 'Portfolio'] = sum(buckets[key] for key in ('roth', 'traditional', 'taxable'))

            if df.loc[year, 'Portfolio'] <= 0 and year < retirement_years:
                df.loc[year:, 'Portfolio'] = 0
                break

        # Enhanced Monte Carlo with vectorized operations
        success_probability, ending_values, median_path = self.monte_carlo_simulation_vectorized(
            future_portfolio, retirement_years, years_until_retirement,
            initial_retirement_budget, spending_adjustments, retirement_start_buckets
        )

        return {
            'pre_retirement_df': pre_retirement_df,
            'retirement_df': df,
            'success_probability': success_probability,
            'future_portfolio': future_portfolio,
            'years_until_retirement': years_until_retirement,
            'retirement_years': retirement_years,
            'initial_retirement_budget': initial_retirement_budget,
            'ending_values': ending_values,  # New: for probability distribution
            'median_path': median_path,  # New: median Monte Carlo path for retirement years
            'pre_retirement_median_path': pre_retirement_median_path  # New: median path for accumulation
        }

    def monte_carlo_simulation_vectorized(self, future_portfolio, retirement_years, years_until_retirement,
                                        initial_retirement_budget, spending_adjustments, initial_buckets=None):
        """Vectorized Monte Carlo simulation using NumPy for massive speed improvements"""
        # Use configured number of simulations
        num_simulations = int(self.num_simulations)

        # Generate all random returns at once using vectorization
        portfolio_returns = self.generate_portfolio_returns(num_simulations, retirement_years)

        # Track each tax bucket independently while preserving aggregate output.
        initial_buckets = initial_buckets or self.initial_bucket_balances()
        roth = np.full(num_simulations, initial_buckets['roth'], dtype=np.float64)
        traditional = np.full(num_simulations, initial_buckets['traditional'], dtype=np.float64)
        taxable = np.full(num_simulations, initial_buckets['taxable'], dtype=np.float64)
        taxable_basis = np.full(num_simulations, initial_buckets['taxable_basis'], dtype=np.float64)
        portfolios = np.zeros((num_simulations, retirement_years + 1), dtype=np.float64)
        portfolios[:, 0] = future_portfolio

        # Vectorized simulation across all scenarios
        for year in range(retirement_years):
            # Calculate expenses for this year (vectorized across all simulations)
            annual_benefit = self.monthly_benefit_income * 12 * (1 + self.inflation_rate) ** (year + years_until_retirement)
            phase_factor = spending_adjustments[year] if year < len(spending_adjustments) else spending_adjustments[-1]
            annual_budget = initial_retirement_budget * (1 + self.inflation_rate) ** year * phase_factor
            healthcare_cost = self.healthcare_costs * 12 * (1 + self.inflation_rate + 0.02) ** (year + years_until_retirement)
            total_expenses = annual_budget + healthcare_cost

            remaining = np.full(num_simulations, max(0, total_expenses - annual_benefit), dtype=np.float64)
            taxable_ratio = np.divide(taxable_basis, taxable, out=np.ones_like(taxable), where=taxable > 0)
            taxable_tax_rate = self.tax_rate * (1 - taxable_ratio)
            taxable_withdrawal = np.minimum(
                taxable,
                np.divide(remaining, 1 - taxable_tax_rate, out=np.zeros_like(remaining), where=taxable_tax_rate < 1)
            )
            taxable_tax = taxable_withdrawal * taxable_tax_rate
            taxable -= taxable_withdrawal
            taxable_basis = np.maximum(0, taxable_basis - taxable_withdrawal * taxable_ratio)
            remaining -= taxable_withdrawal - taxable_tax

            traditional_withdrawal = np.minimum(traditional, remaining / (1 - self.tax_rate))
            traditional -= traditional_withdrawal
            remaining -= traditional_withdrawal * (1 - self.tax_rate)

            roth_withdrawal = np.minimum(roth, np.maximum(0, remaining))
            roth -= roth_withdrawal

            # Grow each remaining bucket after withdrawals.
            returns = 1 + portfolio_returns[:, year]
            roth = np.maximum(0, roth * returns)
            traditional = np.maximum(0, traditional * returns)
            taxable = np.maximum(0, taxable * returns)
            portfolios[:, year + 1] = roth + traditional + taxable

            # Set depleted portfolios to 0 for remaining years
            depleted = portfolios[:, year + 1] <= 0
            portfolios[depleted, year + 1:] = 0

        # Calculate success rate and ending values
        ending_values = portfolios[:, -1]
        success_rate = (ending_values > 0).mean() * 100
        
        # Calculate median path for visualization
        median_path = np.median(portfolios, axis=0)

        return success_rate, ending_values, median_path

    def monte_carlo_pre_retirement_vectorized(self, initial_portfolio, years_until_retirement, 
                                              yearly_contributions):
        """Monte Carlo simulation for accumulation phase - returns median path"""
        num_simulations = int(self.num_simulations)
        
        # Generate correlated returns for accumulation phase
        portfolio_returns = self.generate_portfolio_returns(num_simulations, years_until_retirement)
        
        # Initialize arrays for vectorized calculations
        portfolios = np.full((num_simulations, years_until_retirement + 1), initial_portfolio, dtype=np.float64)
        
        # Vectorized simulation across all scenarios
        for year in range(years_until_retirement):
            # Determine contribution for this year
            if year < len(yearly_contributions):
                contribution = yearly_contributions[year]
            else:
                contribution = 0
            
            # Apply returns with contribution
            current_portfolios = portfolios[:, year]
            after_contribution = current_portfolios + contribution
            returns = after_contribution * portfolio_returns[:, year]
            portfolios[:, year + 1] = after_contribution + returns
        
        # Calculate median path for visualization
        median_path = np.median(portfolios, axis=0)
        
        return median_path

    def calculate_tax_efficient_contributions(self, year: int) -> Dict[str, float]:
        """Calculate tax-efficient allocation across multiple investment streams"""
        remaining_capacity = {}
        total_contributions = 0
        tax_savings = 0

        # Sort streams by tax efficiency (Roth limits first, then traditional, then taxable)
        priority = {'roth': 0, 'traditional': 1, 'taxable': 2}
        sorted_streams = sorted(self.investment_streams, key=lambda s: priority.get(s.tax_treatment, 99))

        for stream in sorted_streams:
            available_contribution = min(stream.annual_contribution, stream.contribution_limit)

            # Add employer match
            employer_match = min(available_contribution * stream.employer_match_rate,
                                 stream.employer_match_limit)

            total_contribution = available_contribution + employer_match
            total_contributions += total_contribution

            # Calculate tax implications
            if stream.tax_treatment == 'traditional':
                tax_savings += available_contribution * self.tax_rate

            remaining_capacity[stream.name] = {
                'contributed': available_contribution,
                'employer_match': employer_match,
                'tax_treatment': stream.tax_treatment,
                'tax_savings': available_contribution * self.tax_rate if stream.tax_treatment == 'traditional' else 0
            }

        return {
            'total_contributions': total_contributions,
            'tax_savings': tax_savings,
            'stream_details': remaining_capacity
        }

    def initial_bucket_balances(self) -> Dict[str, float]:
        """Allocate the current aggregate balance across tax buckets."""
        if not self.investment_streams:
            treatment = {
                'Roth IRA/401k': 'roth',
                'Traditional IRA/401k': 'traditional',
                'Taxable Account': 'taxable'
            }.get(self.account_type, 'traditional')
            return {
                'roth': self.portfolio_total if treatment == 'roth' else 0.0,
                'traditional': self.portfolio_total if treatment == 'traditional' else 0.0,
                'taxable': self.portfolio_total if treatment == 'taxable' else 0.0,
                'taxable_basis': self.portfolio_total if treatment == 'taxable' else 0.0
            }

        stream_totals = {'roth': 0.0, 'traditional': 0.0, 'taxable': 0.0}
        for stream in self.investment_streams:
            contribution = min(stream.annual_contribution, stream.contribution_limit)
            match = min(contribution * stream.employer_match_rate, stream.employer_match_limit)
            stream_totals[stream.tax_treatment] = stream_totals.get(stream.tax_treatment, 0.0) + contribution + match

        total = sum(stream_totals.values())
        if total <= 0:
            stream_totals['traditional'] = self.portfolio_total
            total = self.portfolio_total
        return {
            'roth': self.portfolio_total * stream_totals['roth'] / total,
            'traditional': self.portfolio_total * stream_totals['traditional'] / total,
            'taxable': self.portfolio_total * stream_totals['taxable'] / total,
            'taxable_basis': self.portfolio_total * stream_totals['taxable'] / total
        }

    def withdraw_from_buckets(self, buckets: Dict[str, float], cash_needed: float) -> Dict[str, float]:
        """Withdraw cash in taxable, traditional, Roth order and apply taxes."""
        remaining = max(0.0, cash_needed)
        total_withdrawal = 0.0
        tax_paid = 0.0
        withdrawals = {'taxable': 0.0, 'traditional': 0.0, 'roth': 0.0}

        taxable = buckets['taxable']
        if remaining > 0 and taxable > 0:
            basis_ratio = min(1.0, buckets['taxable_basis'] / taxable) if taxable else 1.0
            gain_ratio = 1.0 - basis_ratio
            tax_rate = self.tax_rate * gain_ratio
            gross = min(taxable, remaining / (1.0 - tax_rate) if tax_rate < 1 else taxable)
            tax = gross * tax_rate
            buckets['taxable'] -= gross
            buckets['taxable_basis'] = max(0.0, buckets['taxable_basis'] - gross * basis_ratio)
            withdrawals['taxable'] = gross
            total_withdrawal += gross
            tax_paid += tax
            remaining -= gross - tax

        if remaining > 0 and buckets['traditional'] > 0:
            gross = min(buckets['traditional'], remaining / (1.0 - self.tax_rate))
            buckets['traditional'] -= gross
            withdrawals['traditional'] = gross
            total_withdrawal += gross
            tax_paid += gross * self.tax_rate
            remaining -= gross * (1.0 - self.tax_rate)

        if remaining > 0 and buckets['roth'] > 0:
            gross = min(buckets['roth'], remaining)
            buckets['roth'] -= gross
            withdrawals['roth'] = gross
            total_withdrawal += gross
            remaining -= gross

        return {
            'withdrawal': total_withdrawal,
            'tax_paid': tax_paid,
            'unfunded': max(0.0, remaining),
            **{f'{key}_withdrawal': value for key, value in withdrawals.items()}
        }

    def contribution_buckets(self, stream_data: Optional[Dict[str, object]] = None) -> Dict[str, float]:
        """Return the current year's contributions grouped by tax treatment."""
        contributions = {'roth': 0.0, 'traditional': 0.0, 'taxable': 0.0}
        if not self.investment_streams:
            treatment = {
                'Roth IRA/401k': 'roth',
                'Traditional IRA/401k': 'traditional',
                'Taxable Account': 'taxable'
            }.get(self.account_type, 'traditional')
            contributions[treatment] = self.yearly_investment
            return contributions

        for detail in (stream_data or {}).get('stream_details', {}).values():
            treatment = detail['tax_treatment']
            contributions[treatment] = contributions.get(treatment, 0.0) + detail['contributed'] + detail['employer_match']
        return contributions

    def create_enhanced_charts(self, results):
        """Create enhanced charts including probability distribution - THREAD SAFE OO API"""
        pre_retirement_df = results['pre_retirement_df']
        df = results['retirement_df']
        ending_values = results['ending_values']
        median_path = results.get('median_path')
        pre_retirement_median_path = results.get('pre_retirement_median_path')

        # Create figure using OO API (thread-safe)
        fig = Figure(figsize=(20, 16))
        
        # Layout: 3 rows, 2 columns
        ax1 = fig.add_subplot(3, 2, 1)
        ax2 = fig.add_subplot(3, 2, 2)
        ax3 = fig.add_subplot(3, 2, 3)
        ax4 = fig.add_subplot(3, 2, 4)
        ax5 = fig.add_subplot(3, 2, 5)
        ax6 = fig.add_subplot(3, 2, 6)

        # Chart 1: Pre-Retirement Portfolio Growth (Median Monte Carlo Path)
        if pre_retirement_median_path is not None:
            pre_retirement_ages = self.age_current + np.arange(len(pre_retirement_median_path))
            ax1.plot(pre_retirement_ages, pre_retirement_median_path, 'purple', linewidth=3, label='Median Path (Monte Carlo)', linestyle='-')
            ax1.fill_between(pre_retirement_ages, 0, pre_retirement_median_path, alpha=0.2, color='purple')
        ax1.plot(pre_retirement_df['Age'], pre_retirement_df['Portfolio'], 'g--', linewidth=2, label='Deterministic Path', alpha=0.6)
        ax1.bar(pre_retirement_df['Age'], pre_retirement_df['Yearly_Contribution'],
                color='blue', alpha=0.4, label='Annual Contributions', width=0.8)
        ax1.set_title('Pre-Retirement Portfolio Growth: Median vs Deterministic', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Age')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

        # Chart 2: Retirement Portfolio Value (Median Monte Carlo Path)
        if median_path is not None:
            retirement_ages = self.age_retire + np.arange(len(median_path))
            ax2.plot(retirement_ages, median_path, 'purple', linewidth=3, label='Median Path (Monte Carlo)', linestyle='-')
            ax2.fill_between(retirement_ages, 0, median_path, alpha=0.2, color='purple')
        ax2.plot(df['Age'], df['Portfolio'], 'b--', linewidth=2, label='Deterministic Path', alpha=0.6)
        ax2.fill_between(df['Age'], 0, df['Portfolio'], alpha=0.2, color='blue')
        ax2.set_title('Retirement Portfolio: Median vs Deterministic', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Age')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

        # Chart 3: Income vs Expenses
        width = 0.35
        ages = df['Age'].values
        ax3.bar(ages - width/2, df['Annual_Benefit'], width, label='Benefit Income', color='green', alpha=0.7)
        ax3.bar(ages + width/2, df['Withdrawal'], width, label='Portfolio Withdrawals', color='red', alpha=0.7)
        ax3.plot(ages, df['Annual_Budget'], 'k--', label='Living Expenses', linewidth=2)
        ax3.plot(ages, df['Healthcare_Costs'], 'r--', label='Healthcare Costs', linewidth=2)
        ax3.set_title('Retirement Income Sources vs Expenses', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Age')
        ax3.set_ylabel('Amount ($)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

        # Chart 4: Portfolio Performance
        ax4.plot(df['Age'], df['Portfolio'], color='teal', linewidth=3, label='Portfolio Balance')
        ax4.fill_between(df['Age'], 0, df['Portfolio'], alpha=0.2, color='teal')
        ax4.set_title('Portfolio Performance Over Time', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Age')
        ax4.set_ylabel('Value ($)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

        # Chart 5: Probability Distribution of Ending Values
        successful = ending_values[ending_values > 0]
        failed_count = int(np.sum(ending_values == 0))

        if successful.size == 0:
            ax5.text(0.5, 0.5, 'No successful scenarios', ha='center', va='center', transform=ax5.transAxes, fontsize=12)
        else:
            ax5.hist(successful, bins=50, alpha=0.7, color='green', density=True, label='Successful Scenarios')
            ax5.axvline(np.median(successful), color='orange', linestyle='--', linewidth=2, label='Median (Successful)')

        if failed_count > 0:
            ax5.text(0.95, 0.95, f'Failed scenarios: {failed_count}/{len(ending_values)}', ha='right', va='top', transform=ax5.transAxes, fontsize=10, color='red')

        ax5.set_title('Distribution of Portfolio Values at End of Retirement', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Final Portfolio Value ($)')
        ax5.set_ylabel('Probability Density')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f'${x/1000000:.1f}M'))

        # Chart 6: Withdrawal Rate Analysis
        withdrawal_rates = df['Withdrawal_Rate'] * 100
        ax6.plot(df['Age'], withdrawal_rates, 'r-', linewidth=3, label='Actual Withdrawal Rate')
        ax6.axhline(4, color='green', linestyle='--', label='4% Rule', linewidth=2)
        ax6.axhline(3, color='blue', linestyle='--', label='3% Conservative', linewidth=2)
        ax6.fill_between(df['Age'], 0, withdrawal_rates, alpha=0.3, color='red')
        ax6.set_title('Portfolio Withdrawal Rate Over Time', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Age')
        ax6.set_ylabel('Withdrawal Rate (%)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(0, max(15, withdrawal_rates.max() * 1.1))

        # Use OO API for saving (thread-safe)
        fig.tight_layout()
        img = io.BytesIO()
        canvas = FigureCanvas(fig)
        canvas.print_png(img)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        return plot_url

    # Keep the original create_charts method for compatibility
    def create_charts(self, results):
        return self.create_enhanced_charts(results)


@app.route('/')
def index():
    return render_template('calculator.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    try:
        # Parse JSON safely (don't raise on bad/missing Content-Type)
        data = request.get_json(silent=True)

        # Basic request validation
        if not isinstance(data, dict):
            raise ValueError('Invalid or missing JSON payload')

        # Required fields (yearly_investment is optional now)
        required_fields = ['age_current', 'age_retire', 'life_expectancy', 'monthly_benefit_income',
                           'portfolio_total', 'years_contributing', 'current_monthly_budget',
                           'inflation_rate', 'expected_return', 'account_type', 'tax_rate',
                           'retirement_budget_ratio', 'retirement_phases', 'healthcare_costs']
        missing = [f for f in required_fields if f not in data or str(data.get(f)).strip() == '']
        if missing:
            raise ValueError(f"Missing required field(s): {', '.join(missing)}")

        # Process investment streams if provided
        investment_streams = []
        if 'investment_streams' in data:
            for stream_data in data['investment_streams']:
                stream = InvestmentStream(
                    name=stream_data['name'],
                    annual_contribution=float(stream_data['annual_contribution']),
                    tax_treatment=stream_data['tax_treatment'],
                    contribution_limit=float(stream_data.get('contribution_limit', float('inf'))),
                    employer_match_rate=float(stream_data.get('employer_match_rate', 0)),
                    employer_match_limit=float(stream_data.get('employer_match_limit', 0))
                )
                investment_streams.append(stream)

        calculator = EnhancedRetirementCalculator(
            age_current=int(data['age_current']),
            age_retire=int(data['age_retire']),
            life_expectancy=int(data['life_expectancy']),
            monthly_benefit_income=float(data['monthly_benefit_income']),
            portfolio_total=float(data['portfolio_total']),
            yearly_investment=float(data.get('yearly_investment', 0)),
            years_contributing=int(data['years_contributing']),
            current_monthly_budget=float(data['current_monthly_budget']),
            inflation_rate=float(data['inflation_rate']) / 100,
            expected_return=float(data['expected_return']) / 100,
            account_type=data['account_type'],
            tax_rate=float(data['tax_rate']) / 100,
            retirement_budget_ratio=float(data['retirement_budget_ratio']) / 100,
            retirement_phases=data['retirement_phases'],
            healthcare_costs=float(data['healthcare_costs']),
            portfolio_volatility=float(data.get('portfolio_volatility', 16)) / 100,
            random_seed=int(data.get('random_seed')) if data.get('random_seed') is not None else None,
            num_simulations=int(data.get('num_simulations', 10000)),
            investment_streams=investment_streams
        )

        results = calculator.calculate()
        chart_data = calculator.create_enhanced_charts(results)

        # Serialize per-year tables for the Details tab
        pre_table = []
        pre_df = results.get('pre_retirement_df')
        if pre_df is not None:
            pre_df = pre_df.reset_index(drop=True)
            for i, row in pre_df.iterrows():
                beginning = float(pre_df.loc[i-1, 'Portfolio']) if i > 0 else float(row['Portfolio'])
                contribution = float(row['Yearly_Contribution']) if 'Yearly_Contribution' in pre_df.columns and not pd.isna(row.get('Yearly_Contribution', 0)) else 0.0
                tax_savings = float(row['Tax_Savings']) if 'Tax_Savings' in pre_df.columns and not pd.isna(row.get('Tax_Savings', 0)) else 0.0
                ending = float(row['Portfolio'])
                pre_table.append({
                    'Year': int(i),
                    'Age': int(row['Age']),
                    'Beginning_Portfolio': beginning,
                    'Yearly_Contribution': contribution,
                    'Tax_Savings': tax_savings,
                    'Ending_Portfolio': ending,
                    'Roth_Balance': float(row.get('Roth_Balance', 0)),
                    'Traditional_Balance': float(row.get('Traditional_Balance', 0)),
                    'Taxable_Balance': float(row.get('Taxable_Balance', 0)),
                    'Taxable_Basis': float(row.get('Taxable_Basis', 0))
                })

        ret_table = []
        ret_df = results.get('retirement_df')
        if ret_df is not None:
            ret_df = ret_df.reset_index(drop=True)
            for i, row in ret_df.iterrows():
                if pd.isna(row.get('Age')):
                    continue
                beginning = float(ret_df.loc[i-1, 'Portfolio']) if i > 0 else float(row['Portfolio'])
                withdrawal = float(row['Withdrawal']) if 'Withdrawal' in ret_df.columns and not pd.isna(row.get('Withdrawal', 0)) else 0.0
                ending = float(row['Portfolio'])
                investment_return = ending - (beginning - withdrawal)
                ret_table.append({
                    'Year': int(i),
                    'Age': int(row['Age']),
                    'Beginning_Portfolio': beginning,
                    'Withdrawal': withdrawal,
                    'Investment_Return': investment_return,
                    'Ending_Portfolio': ending,
                    'Tax_Paid': float(row.get('Tax_Paid', 0)) if 'Tax_Paid' in ret_df.columns and not pd.isna(row.get('Tax_Paid', 0)) else 0.0,
                    'Roth_Balance': float(row.get('Roth_Balance', 0)),
                    'Traditional_Balance': float(row.get('Traditional_Balance', 0)),
                    'Taxable_Balance': float(row.get('Taxable_Balance', 0)),
                    'Taxable_Basis': float(row.get('Taxable_Basis', 0)),
                    'Roth_Withdrawal': float(row.get('Roth_Withdrawal', 0)) if 'Roth_Withdrawal' in ret_df.columns and not pd.isna(row.get('Roth_Withdrawal', 0)) else 0.0,
                    'Traditional_Withdrawal': float(row.get('Traditional_Withdrawal', 0)) if 'Traditional_Withdrawal' in ret_df.columns and not pd.isna(row.get('Traditional_Withdrawal', 0)) else 0.0,
                    'Taxable_Withdrawal': float(row.get('Taxable_Withdrawal', 0)) if 'Taxable_Withdrawal' in ret_df.columns and not pd.isna(row.get('Taxable_Withdrawal', 0)) else 0.0,
                    'Annual_Budget': float(row['Annual_Budget']) if 'Annual_Budget' in ret_df.columns and not pd.isna(row.get('Annual_Budget', 0)) else 0.0,
                    'Healthcare_Costs': float(row['Healthcare_Costs']) if 'Healthcare_Costs' in ret_df.columns and not pd.isna(row.get('Healthcare_Costs', 0)) else 0.0,
                    'Withdrawal_Rate': float(row['Withdrawal_Rate']) if 'Withdrawal_Rate' in ret_df.columns and not pd.isna(row.get('Withdrawal_Rate', 0)) else 0.0
                })

        # Build a simplified, single combined table for the Details tab so the frontend can render one table
        combined_table = []
        # Build continuous year index and avoid duplicating the retirement-start row
        year_counter = 0
        for row in pre_table:
            investment = float(row['Beginning_Portfolio'])
            contributions = float(row.get('Yearly_Contribution', 0.0))
            ending = float(row['Ending_Portfolio'])
            interest = ending - (investment + contributions)
            combined_table.append({
                'Year': year_counter,
                'Age': row['Age'],
                'Investment_Amount': investment,
                'Contributions': contributions,
                'Interest_Earned': interest,
                'Withdrawals': 0.0,
                'Taxable_Withdrawal': 0.0,
                'Traditional_Withdrawal': 0.0,
                'Roth_Withdrawal': 0.0,
                'Ending_Balance': ending
            })
            year_counter += 1

        # Decide start index for retirement rows (skip the initial retirement row if it duplicates the last pre row)
        ret_start = 0
        if pre_table and ret_table:
            if ret_table[0]['Age'] == pre_table[-1]['Age']:
                ret_start = 1

        for idx in range(ret_start, len(ret_table)):
            row = ret_table[idx]
            investment = float(row['Beginning_Portfolio'])
            withdrawals = float(row.get('Withdrawal', 0.0))
            ending = float(row['Ending_Portfolio'])
            interest = float(row.get('Investment_Return', ending - (investment - withdrawals)))
            combined_table.append({
                'Year': year_counter,
                'Age': row['Age'],
                'Investment_Amount': investment,
                'Contributions': 0.0,
                'Interest_Earned': interest,
                'Withdrawals': withdrawals,
                'Taxable_Withdrawal': float(row.get('Taxable_Withdrawal', 0.0)),
                'Traditional_Withdrawal': float(row.get('Traditional_Withdrawal', 0.0)),
                'Roth_Withdrawal': float(row.get('Roth_Withdrawal', 0.0)),
                'Ending_Balance': ending
            })
            year_counter += 1


        df = results['retirement_df']
        ending_values = results['ending_values']
        final_portfolio = df['Portfolio'].iloc[-1]
        portfolio_survives = bool(pd.notna(final_portfolio) and final_portfolio > 0)
        depleted_rows = df.loc[df['Portfolio'].fillna(0) <= 0, 'Age']
        depletion_age = None if portfolio_survives else int(depleted_rows.min()) if not depleted_rows.empty else None

        # Additional statistics from Monte Carlo
        successful_endings = ending_values[ending_values > 0]
        percentiles = {
            '10th': np.percentile(successful_endings, 10) if len(successful_endings) > 0 else 0,
            '50th': np.percentile(successful_endings, 50) if len(successful_endings) > 0 else 0,
            '90th': np.percentile(successful_endings, 90) if len(successful_endings) > 0 else 0
        }

        return jsonify({
            'success': True,
            'results': {
                'future_portfolio': f"${results['future_portfolio']:,.2f}",
                'years_until_retirement': results['years_until_retirement'],
                'retirement_years': results['retirement_years'],
                'initial_monthly_budget': f"${results['initial_retirement_budget'] / 12:,.2f}",
                'portfolio_survives': portfolio_survives,
                'final_portfolio': f"${final_portfolio:,.2f}" if portfolio_survives else None,
                'depletion_age': depletion_age,
                'success_probability': f"{results['success_probability']:.1f}%",
                'chart': chart_data,
                # New results
                'percentiles': {
                    '10th_percentile': f"${percentiles['10th']:,.0f}",
                    '50th_percentile': f"${percentiles['50th']:,.0f}",
                    '90th_percentile': f"${percentiles['90th']:,.0f}"
                },
                'details': {
                    'pre_retirement': pre_table,
                    'retirement': ret_table,
                    'combined': combined_table
                }
            }
        })

    except ValueError as e:
        logger.error("Validation error during calculation: %s", e)
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.exception("Unhandled exception during calculation")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    debug_flag = os.getenv('FLASK_DEBUG', '0') == '1'
    app.run(debug=debug_flag)
