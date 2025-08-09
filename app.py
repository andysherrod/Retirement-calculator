from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import base64
import io
import json

app = Flask(__name__)

class EnhancedRetirementCalculator:
    def __init__(self, age_current, age_retire, life_expectancy, monthly_benefit_income,
                 portfolio_total, yearly_investment, years_contributing, current_monthly_budget,
                 inflation_rate, expected_return, account_type, tax_rate, retirement_budget_ratio,
                 retirement_phases, healthcare_costs):
        
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

    def calculate(self):
        retirement_years = self.life_expectancy - self.age_retire
        years_until_retirement = self.age_retire - self.age_current
        years_contributing = min(self.years_contributing, years_until_retirement)

        pre_retirement_df = pd.DataFrame(index=range(years_until_retirement + 1))
        pre_retirement_df.loc[0, 'Age'] = self.age_current
        pre_retirement_df.loc[0, 'Portfolio'] = self.portfolio_total
        pre_retirement_df.loc[0, 'Yearly_Contribution'] = self.yearly_investment if years_contributing > 0 else 0

        for year in range(1, years_until_retirement + 1):
            pre_retirement_df.loc[year, 'Age'] = self.age_current + year
            contribution = self.yearly_investment if year <= years_contributing else 0
            pre_retirement_df.loc[year, 'Yearly_Contribution'] = contribution
            previous_portfolio = pre_retirement_df.loc[year - 1, 'Portfolio']
            investment_return = (previous_portfolio + contribution) * self.expected_return
            pre_retirement_df.loc[year, 'Portfolio'] = previous_portfolio + contribution + investment_return

        future_portfolio = pre_retirement_df.loc[years_until_retirement, 'Portfolio']

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

        success_probability = self.monte_carlo_simulation(
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
            'initial_retirement_budget': initial_retirement_budget
        }

    def monte_carlo_simulation(self, future_portfolio, retirement_years, years_until_retirement,
                               initial_retirement_budget, spending_adjustments):
        num_simulations = 1000
        success_count = 0

        historical_returns = np.random.normal(self.expected_return, 0.12, (num_simulations, retirement_years))

        for sim in range(num_simulations):
            sim_portfolio = future_portfolio
            for year in range(retirement_years):
                if sim_portfolio <= 0:
                    break

                annual_benefit = self.monthly_benefit_income * 12 * (1 + self.inflation_rate) ** (year + years_until_retirement)
                phase_factor = spending_adjustments[year] if year < len(spending_adjustments) else spending_adjustments[-1]
                annual_budget = initial_retirement_budget * (1 + self.inflation_rate) ** year * phase_factor
                healthcare_cost = self.healthcare_costs * 12 * (1 + self.inflation_rate + 0.02) ** (year + years_until_retirement)
                total_expenses = annual_budget + healthcare_cost

                if self.account_type == 'Roth IRA/401k':
                    withdrawal = max(0, total_expenses - annual_benefit)
                else:
                    withdrawal = max(0, (total_expenses - annual_benefit) / (1 - self.tax_rate))

                sim_portfolio = sim_portfolio - withdrawal + (sim_portfolio - withdrawal / 2) * historical_returns[sim, year]

            if sim_portfolio > 0:
                success_count += 1

        return success_count / num_simulations * 100

    def create_charts(self, results):
        pre_retirement_df = results['pre_retirement_df']
        df = results['retirement_df']

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        ax1.plot(pre_retirement_df['Age'], pre_retirement_df['Portfolio'], 'g-', linewidth=2)
        ax1.bar(pre_retirement_df['Age'], pre_retirement_df['Yearly_Contribution'],
                color='blue', alpha=0.3, label='Annual Contributions')
        ax1.set_title('Pre-Retirement Portfolio Growth')
        ax1.set_xlabel('Age')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True)
        ax1.legend()
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

        ax2.plot(df['Age'], df['Portfolio'], 'b-', linewidth=2)
        ax2.set_title('Retirement Portfolio Value by Age')
        ax2.set_xlabel('Age')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.grid(True)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

        ax3.bar(df['Age'], df['Annual_Benefit'], label='Benefit Income', color='green', alpha=0.6)
        ax3.bar(df['Age'], df['Withdrawal'], bottom=df['Annual_Benefit'],
                label='Portfolio Withdrawals', color='red', alpha=0.6)
        ax3.plot(df['Age'], df['Annual_Budget'], 'k--', label='Living Expenses', linewidth=2)
        ax3.plot(df['Age'], df['Healthcare_Costs'], 'r--', label='Healthcare Costs', linewidth=2)
        ax3.set_title('Retirement Income and Expenses by Age')
        ax3.set_xlabel('Age')
        ax3.set_ylabel('Amount ($)')
        ax3.grid(True)
        ax3.legend()
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

        ax4.plot(df['Age'], df['Annual_Budget'], label='Yearly Budget', color='blue', linewidth=2)
        ax4.plot(df['Age'], df['Annual_Benefit'], label='Benefit Income', color='green', linewidth=2)
        ax4.plot(df['Age'], df['Portfolio'] * self.expected_return, label='Investment Income',
                 color='orange', linewidth=2)
        ax4.set_title('Yearly Budget vs Income Sources')
        ax4.set_xlabel('Age')
        ax4.set_ylabel('Amount ($)')
        ax4.grid(True)
        ax4.legend()
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

        plt.tight_layout()
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=150, bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return plot_url


@app.route('/')
def index():
    return render_template('calculator.html')


@app.route('/calculate', methods=['POST'])
def calculate():
    try:
        data = request.get_json()

        calculator = EnhancedRetirementCalculator(
            age_current=int(data['age_current']),
            age_retire=int(data['age_retire']),
            life_expectancy=int(data['life_expectancy']),
            monthly_benefit_income=float(data['monthly_benefit_income']),
            portfolio_total=float(data['portfolio_total']),
            yearly_investment=float(data['yearly_investment']),
            years_contributing=int(data['years_contributing']),
            current_monthly_budget=float(data['current_monthly_budget']),
            inflation_rate=float(data['inflation_rate']) / 100,
            expected_return=float(data['expected_return']) / 100,
            account_type=data['account_type'],
            tax_rate=float(data['tax_rate']) / 100,
            retirement_budget_ratio=float(data['retirement_budget_ratio']) / 100,
            retirement_phases=data['retirement_phases'],
            healthcare_costs=float(data['healthcare_costs'])
        )

        results = calculator.calculate()
        chart_data = calculator.create_charts(results)

        df = results['retirement_df']
        final_portfolio = df['Portfolio'].iloc[-1]
        portfolio_survives = bool(final_portfolio > 0)
        depletion_age = None if portfolio_survives else int(df.loc[df['Portfolio'] <= 0, 'Age'].min())

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
                'chart': chart_data
            }
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
