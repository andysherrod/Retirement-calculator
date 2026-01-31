from flask import Flask, render_template, request, jsonify
import os
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
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
                 # New parameters for asset allocation
                 stock_allocation=0.7, bond_allocation=0.3, stock_volatility=0.16,
                 bond_volatility=0.05, correlation=-0.1, investment_streams=None,
                 random_seed: Optional[int] = None, num_simulations: int = 10000):

        # Basic validations
        if age_retire <= age_current:
            raise ValueError("age_retire must be greater than age_current")
        if stock_allocation < 0 or bond_allocation < 0 or abs(stock_allocation + bond_allocation - 1.0) > 1e-6:
            raise ValueError("stock_allocation and bond_allocation must be non-negative and sum to 1.0")

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

        # New asset allocation parameters
        self.stock_allocation = stock_allocation
        self.bond_allocation = bond_allocation
        self.stock_volatility = stock_volatility
        self.bond_volatility = bond_volatility
        self.correlation = correlation

        # Randomness and simulation config
        self.rng = np.random.default_rng(random_seed)
        self.num_simulations = int(num_simulations)

        # Multiple investment streams
        self.investment_streams = investment_streams or []

    def generate_correlated_returns(self, num_simulations: int, num_years: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate correlated stock and bond returns using NumPy vectorization"""
        cov = [[self.stock_volatility**2, self.correlation * self.stock_volatility * self.bond_volatility],
               [self.correlation * self.stock_volatility * self.bond_volatility, self.bond_volatility**2]]

        # Draw samples using the instance RNG
        random_returns = self.rng.multivariate_normal(
            [self.expected_return, 0.04],  # Stock and bond expected returns
            cov,
            size=(num_simulations, num_years)
        )

        stock_returns = random_returns[:, :, 0]
        bond_returns = random_returns[:, :, 1]

        return stock_returns, bond_returns

    def calculate_portfolio_returns(self, stock_returns: np.ndarray, bond_returns: np.ndarray) -> np.ndarray:
        """Calculate portfolio returns based on asset allocation"""
        return self.stock_allocation * stock_returns + self.bond_allocation * bond_returns

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

        pre_retirement_df = pd.DataFrame(index=range(years_until_retirement + 1))
        pre_retirement_df.loc[0, 'Age'] = self.age_current
        pre_retirement_df.loc[0, 'Portfolio'] = self.portfolio_total
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
            previous_portfolio = pre_retirement_df.loc[year - 1, 'Portfolio']

            # Use asset allocation for return calculation
            portfolio_return = (self.stock_allocation * self.expected_return +
                               self.bond_allocation * 0.04)  # Assume 4% bond return
            investment_return = (previous_portfolio + contribution) * portfolio_return
            pre_retirement_df.loc[year, 'Portfolio'] = previous_portfolio + contribution + investment_return

        future_portfolio = pre_retirement_df.loc[years_until_retirement, 'Portfolio']

        # Retirement phase calculations remain similar but with enhanced Monte Carlo
        df = pd.DataFrame(index=range(retirement_years + 1))
        df.loc[0, 'Age'] = self.age_retire
        df.loc[0, 'Portfolio'] = future_portfolio
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
            if self.account_type == 'Roth IRA/401k':
                withdrawal_needed = total_expenses - df.loc[year, 'Annual_Benefit']
            else:
                withdrawal_needed = (total_expenses - df.loc[year, 'Annual_Benefit']) / (1 - self.tax_rate)

            df.loc[year, 'Withdrawal'] = max(0, withdrawal_needed)
            df.loc[year, 'Withdrawal_Rate'] = (
                df.loc[year, 'Withdrawal'] / df.loc[year - 1, 'Portfolio']
                if df.loc[year - 1, 'Portfolio'] > 0 else 0
            )

            starting_portfolio = df.loc[year - 1, 'Portfolio']
            investment_returns = (starting_portfolio - df.loc[year, 'Withdrawal'] / 2) * self.expected_return
            df.loc[year, 'Portfolio'] = max(0, starting_portfolio - df.loc[year, 'Withdrawal'] + investment_returns)

            if df.loc[year, 'Portfolio'] <= 0 and year < retirement_years:
                df.loc[year:, 'Portfolio'] = 0
                break

        # Enhanced Monte Carlo with vectorized operations
        success_probability, ending_values = self.monte_carlo_simulation_vectorized(
            future_portfolio, retirement_years, years_until_retirement,
            initial_retirement_budget, spending_adjustments
        )

        return {
            'pre_retirement_df': pre_retirement_df,
            'retirement_df': df,
            'success_probability': success_probability,
            'future_portfolio': future_portfolio,
            'years_until_retirement': years_until_retirement,
            'retirement_years': retirement_years,
            'initial_retirement_budget': initial_retirement_budget,
            'ending_values': ending_values  # New: for probability distribution
        }

    def monte_carlo_simulation_vectorized(self, future_portfolio, retirement_years, years_until_retirement,
                                        initial_retirement_budget, spending_adjustments):
        """Vectorized Monte Carlo simulation using NumPy for massive speed improvements"""
        # Use configured number of simulations
        num_simulations = int(self.num_simulations)

        # Generate all random returns at once using vectorization
        stock_returns, bond_returns = self.generate_correlated_returns(num_simulations, retirement_years)
        portfolio_returns = self.calculate_portfolio_returns(stock_returns, bond_returns)

        # Initialize arrays for vectorized calculations
        portfolios = np.full((num_simulations, retirement_years + 1), future_portfolio, dtype=np.float64)

        # Vectorized simulation across all scenarios
        for year in range(retirement_years):
            # Calculate expenses for this year (vectorized across all simulations)
            annual_benefit = self.monthly_benefit_income * 12 * (1 + self.inflation_rate) ** (year + years_until_retirement)
            phase_factor = spending_adjustments[year] if year < len(spending_adjustments) else spending_adjustments[-1]
            annual_budget = initial_retirement_budget * (1 + self.inflation_rate) ** year * phase_factor
            healthcare_cost = self.healthcare_costs * 12 * (1 + self.inflation_rate + 0.02) ** (year + years_until_retirement)
            total_expenses = annual_budget + healthcare_cost

            # Calculate withdrawals (vectorized)
            if self.account_type == 'Roth IRA/401k':
                withdrawals = np.maximum(0, total_expenses - annual_benefit)
            else:
                withdrawals = np.maximum(0, (total_expenses - annual_benefit) / (1 - self.tax_rate))

            # Update portfolios (vectorized across all simulations)
            current_portfolios = portfolios[:, year]
            after_withdrawal = current_portfolios - withdrawals
            returns = after_withdrawal * portfolio_returns[:, year]
            portfolios[:, year + 1] = np.maximum(0, after_withdrawal + returns)

            # Set depleted portfolios to 0 for remaining years
            depleted = portfolios[:, year + 1] <= 0
            portfolios[depleted, year + 1:] = 0

        # Calculate success rate and ending values
        ending_values = portfolios[:, -1]
        success_rate = (ending_values > 0).mean() * 100

        return success_rate, ending_values

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

    def create_enhanced_charts(self, results):
        """Create enhanced charts including probability distribution"""
        pre_retirement_df = results['pre_retirement_df']
        df = results['retirement_df']
        ending_values = results['ending_values']

        # Create a larger figure with more subplots
        fig = plt.figure(figsize=(20, 16))

        # Layout: 3 rows, 2 columns
        ax1 = plt.subplot(3, 2, 1)
        ax2 = plt.subplot(3, 2, 2)
        ax3 = plt.subplot(3, 2, 3)
        ax4 = plt.subplot(3, 2, 4)
        ax5 = plt.subplot(3, 2, 5)
        ax6 = plt.subplot(3, 2, 6)

        # Chart 1: Pre-Retirement Portfolio Growth
        ax1.plot(pre_retirement_df['Age'], pre_retirement_df['Portfolio'], 'g-', linewidth=3, label='Portfolio Value')
        ax1.bar(pre_retirement_df['Age'], pre_retirement_df['Yearly_Contribution'],
                color='blue', alpha=0.4, label='Annual Contributions', width=0.8)
        ax1.set_title('Pre-Retirement Portfolio Growth', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Age')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

        # Chart 2: Retirement Portfolio Value
        ax2.plot(df['Age'], df['Portfolio'], 'b-', linewidth=3)
        ax2.fill_between(df['Age'], 0, df['Portfolio'], alpha=0.3, color='blue')
        ax2.set_title('Retirement Portfolio Depletion', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Age')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

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
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

        # Chart 4: Asset Allocation Performance
        years = np.arange(len(df))
        stock_component = df['Portfolio'] * self.stock_allocation
        bond_component = df['Portfolio'] * self.bond_allocation
        ax4.fill_between(df['Age'], 0, stock_component, alpha=0.6, color='red', label=f'Stocks ({self.stock_allocation*100:.0f}%)')
        ax4.fill_between(df['Age'], stock_component, stock_component + bond_component, alpha=0.6, color='blue', label=f'Bonds ({self.bond_allocation*100:.0f}%)')
        ax4.set_title('Portfolio Asset Allocation Over Time', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Age')
        ax4.set_ylabel('Value ($)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

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
        ax5.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000000:.1f}M'))

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

        plt.tight_layout()
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=150, bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

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
            # New parameters
            stock_allocation=float(data.get('stock_allocation', 70)) / 100,
            bond_allocation=float(data.get('bond_allocation', 30)) / 100,
            stock_volatility=float(data.get('stock_volatility', 16)) / 100,
            bond_volatility=float(data.get('bond_volatility', 5)) / 100,
            correlation=float(data.get('correlation', -10)) / 100,
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
                    'Ending_Portfolio': ending
                })

        ret_table = []
        ret_df = results.get('retirement_df')
        if ret_df is not None:
            ret_df = ret_df.reset_index(drop=True)
            for i, row in ret_df.iterrows():
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
                    'Annual_Budget': float(row['Annual_Budget']) if 'Annual_Budget' in ret_df.columns and not pd.isna(row.get('Annual_Budget', 0)) else 0.0,
                    'Healthcare_Costs': float(row['Healthcare_Costs']) if 'Healthcare_Costs' in ret_df.columns and not pd.isna(row.get('Healthcare_Costs', 0)) else 0.0,
                    'Withdrawal_Rate': float(row['Withdrawal_Rate']) if 'Withdrawal_Rate' in ret_df.columns and not pd.isna(row.get('Withdrawal_Rate', 0)) else 0.0
                })

        df = results['retirement_df']
        ending_values = results['ending_values']
        final_portfolio = df['Portfolio'].iloc[-1]
        portfolio_survives = bool(final_portfolio > 0)
        depletion_age = None if portfolio_survives else int(df.loc[df['Portfolio'] <= 0, 'Age'].min())

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
                'asset_allocation': {
                    'stocks': f"{calculator.stock_allocation*100:.0f}%",
                    'bonds': f"{calculator.bond_allocation*100:.0f}%"
                },
                'details': {
                    'pre_retirement': pre_table,
                    'retirement': ret_table
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
