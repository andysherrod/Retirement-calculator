from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.ticker
import base64
import io
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

TAX_CONFIG = {
    'single': {
        'standard_deduction': 14600,
        'brackets': [
            (11600, 0.10),
            (47150, 0.12),
            (100525, 0.22),
            (191950, 0.24),
            (243725, 0.32),
            (609350, 0.35),
            (float('inf'), 0.37),
        ],
    },
    'married': {
        'standard_deduction': 29200,
        'brackets': [
            (23200, 0.10),
            (94300, 0.12),
            (201050, 0.22),
            (383900, 0.24),
            (487450, 0.32),
            (731200, 0.35),
            (float('inf'), 0.37),
        ],
    },
}

@dataclass
class InvestmentStream:
    name: str
    annual_contribution: float
    tax_treatment: str


def calculate_tax_liability(annual_income, filing_status='married'):
    config = TAX_CONFIG.get(filing_status, TAX_CONFIG['married'])
    standard_deduction = config['standard_deduction']
    taxable_income = max(0.0, annual_income - standard_deduction)
    if annual_income <= 0:
        return 0.0, 0.0

    total_tax = 0.0
    prev_limit = 0.0
    for limit, rate in config['brackets']:
        if taxable_income <= prev_limit:
            break
        taxed_amt = min(taxable_income, limit) - prev_limit
        total_tax += taxed_amt * rate
        prev_limit = limit

    effective_rate = total_tax / annual_income
    return total_tax, effective_rate


def calculate_effective_tax(target_budget, filing_status='married'):
    _, effective_rate = calculate_tax_liability(target_budget, filing_status)
    return effective_rate

class EnhancedRetirementCalculator:
    def __init__(self, age_current, age_retire, trad_start, roth_start, tax_start, 
                 investment_streams, target_budget, monthly_benefits, expected_return, 
                 inflation_rate, effective_tax_rate):
        self.age_current = age_current
        self.age_retire = age_retire
        self.life_expectancy = 90
        
        self.trad_start = trad_start
        self.roth_start = roth_start
        self.tax_start = tax_start
        self.total_start = trad_start + roth_start + tax_start
        
        self.investment_streams = investment_streams
        self.target_budget = target_budget
        self.monthly_benefits = monthly_benefits
        self.expected_return = expected_return
        self.inflation_rate = inflation_rate
        self.effective_tax_rate = effective_tax_rate
        
        self.portfolio_volatility = 0.12 
        self.rng = np.random.default_rng(None)
        self.num_simulations = 10000

    def generate_returns(self, num_simulations, num_years):
        return self.rng.normal(self.expected_return, self.portfolio_volatility, size=(num_simulations, num_years))

    def calculate(self):
        years_to_retire = self.age_retire - self.age_current
        retirement_years = self.life_expectancy - self.age_retire
        
        trad_contr = sum(s.annual_contribution for s in self.investment_streams if s.tax_treatment == 'traditional')
        roth_contr = sum(s.annual_contribution for s in self.investment_streams if s.tax_treatment == 'roth')
        tax_contr = sum(s.annual_contribution for s in self.investment_streams if s.tax_treatment == 'taxable')
        total_contr = trad_contr + roth_contr + tax_contr
        
        # Calculate the total out-of-pocket money invested over their lifetime
        total_invested_out_of_pocket = self.total_start + (total_contr * years_to_retire)

        # --- 1. Deterministic Path (For Table) ---
        combined_table = []
        cur_trad, cur_roth, cur_tax = self.trad_start, self.roth_start, self.tax_start

        # Accumulation
        for year in range(years_to_retire):
            beginning = cur_trad + cur_roth + cur_tax
            
            int_trad = (cur_trad + trad_contr) * self.expected_return
            int_roth = (cur_roth + roth_contr) * self.expected_return
            int_tax = (cur_tax + tax_contr) * self.expected_return
            
            cur_trad += trad_contr + int_trad
            cur_roth += roth_contr + int_roth
            cur_tax += tax_contr + int_tax
            
            combined_table.append({
                'Year': year,
                'Age': self.age_current + year,
                'Investment_Amount': beginning,
                'Contributions': total_contr,
                'Interest_Earned': int_trad + int_roth + int_tax,
                'Withdrawals': 0.0,
                'Ending_Balance': cur_trad + cur_roth + cur_tax
            })

        future_total = cur_trad + cur_roth + cur_tax
        
        # Initial Withdrawal Rate Calculation
        first_year_budget = self.target_budget * ((1 + self.inflation_rate) ** years_to_retire)
        first_year_benefit = self.monthly_benefits * 12 * ((1 + self.inflation_rate) ** years_to_retire)
        initial_net_withdrawal = max(0, first_year_budget - first_year_benefit)
        
        initial_gross_withdrawal = 0
        temp_tax, temp_trad, temp_roth = cur_tax, cur_trad, cur_roth
        temp_needed = initial_net_withdrawal
        
        draw_tax = min(temp_tax, temp_needed)
        temp_needed -= draw_tax
        initial_gross_withdrawal += draw_tax
        
        trad_gross_needed = temp_needed / (1 - self.effective_tax_rate) if self.effective_tax_rate < 1 else 0
        draw_trad = min(temp_trad, trad_gross_needed)
        temp_needed -= draw_trad * (1 - self.effective_tax_rate)
        initial_gross_withdrawal += draw_trad
        
        draw_roth = min(temp_roth, temp_needed)
        initial_gross_withdrawal += draw_roth
        
        initial_withdrawal_rate = (initial_gross_withdrawal / future_total * 100) if future_total > 0 else float('inf')

        # Drawdown
        for year in range(retirement_years):
            beginning = cur_trad + cur_roth + cur_tax
            annual_benefit = self.monthly_benefits * 12 * ((1 + self.inflation_rate) ** (year + years_to_retire))
            annual_budget = self.target_budget * ((1 + self.inflation_rate) ** (year + years_to_retire))
            net_needed = max(0, annual_budget - annual_benefit)
            
            gross_withdrawal = 0

            draw_tax = min(cur_tax, net_needed)
            cur_tax -= draw_tax
            net_needed -= draw_tax
            gross_withdrawal += draw_tax

            draw_roth = min(cur_roth, net_needed)
            cur_roth -= draw_roth
            net_needed -= draw_roth
            gross_withdrawal += draw_roth

            trad_gross_needed = net_needed / (1 - self.effective_tax_rate) if self.effective_tax_rate < 1 else 0
            draw_trad = min(cur_trad, trad_gross_needed)
            cur_trad -= draw_trad
            net_needed -= draw_trad * (1 - self.effective_tax_rate)
            gross_withdrawal += draw_trad

            int_trad = cur_trad * self.expected_return
            int_roth = cur_roth * self.expected_return
            int_tax = cur_tax * self.expected_return

            cur_roth += int_roth
            cur_tax += int_tax

            tax_paid = draw_trad * self.effective_tax_rate

            combined_table.append({
                'Year': years_to_retire + year,
                'Age': self.age_retire + year,
                'Investment_Amount': beginning,
                'Contributions': 0.0,
                'Interest_Earned': int_trad + int_roth + int_tax,
                'Withdrawals': gross_withdrawal,
                'withdrawal_breakdown': {
                    'traditional': draw_trad,
                    'roth': draw_roth,
                    'taxable': draw_tax,
                    'tax_paid': tax_paid
                },
                'Ending_Balance': cur_trad + cur_roth + cur_tax
            })

        # --- 2. Monte Carlo Simulation ---
        pre_returns = self.generate_returns(self.num_simulations, years_to_retire)
        
        port_trad = np.full((self.num_simulations, years_to_retire + 1), self.trad_start, dtype=np.float64)
        port_roth = np.full((self.num_simulations, years_to_retire + 1), self.roth_start, dtype=np.float64)
        port_tax = np.full((self.num_simulations, years_to_retire + 1), self.tax_start, dtype=np.float64)
        
        for yr in range(years_to_retire):
            ret_multiplier = 1 + pre_returns[:, yr]
            port_trad[:, yr+1] = (port_trad[:, yr] + trad_contr) * ret_multiplier
            port_roth[:, yr+1] = (port_roth[:, yr] + roth_contr) * ret_multiplier
            port_tax[:, yr+1] = (port_tax[:, yr] + tax_contr) * ret_multiplier
            
        pre_total = port_trad + port_roth + port_tax
        pre_retirement_median_path = np.median(pre_total, axis=0)

        ret_returns = self.generate_returns(self.num_simulations, retirement_years)
        
        ret_trad = np.full((self.num_simulations, retirement_years + 1), 0.0, dtype=np.float64)
        ret_roth = np.full((self.num_simulations, retirement_years + 1), 0.0, dtype=np.float64)
        ret_tax = np.full((self.num_simulations, retirement_years + 1), 0.0, dtype=np.float64)
        
        ret_trad[:, 0] = port_trad[:, -1]
        ret_roth[:, 0] = port_roth[:, -1]
        ret_tax[:, 0] = port_tax[:, -1]
        
        for yr in range(retirement_years):
            annual_benefit = self.monthly_benefits * 12 * ((1 + self.inflation_rate) ** (yr + years_to_retire))
            annual_budget = self.target_budget * ((1 + self.inflation_rate) ** (yr + years_to_retire))
            net_needed = np.full(self.num_simulations, max(0, annual_budget - annual_benefit))
            
            draw_tax = np.minimum(ret_tax[:, yr], net_needed)
            ret_tax[:, yr] -= draw_tax
            net_needed -= draw_tax

            draw_roth = np.minimum(ret_roth[:, yr], net_needed)
            ret_roth[:, yr] -= draw_roth
            net_needed -= draw_roth

            trad_gross_needed = net_needed / (1 - self.effective_tax_rate)
            draw_trad = np.minimum(ret_trad[:, yr], trad_gross_needed)
            ret_trad[:, yr] -= draw_trad
            net_needed -= draw_trad * (1 - self.effective_tax_rate)

            ret_multiplier = 1 + ret_returns[:, yr]
            ret_trad[:, yr+1] = ret_trad[:, yr] * ret_multiplier
            ret_roth[:, yr+1] = ret_roth[:, yr] * ret_multiplier
            ret_tax[:, yr+1] = ret_tax[:, yr] * ret_multiplier

        ret_total = ret_trad + ret_roth + ret_tax
        success_rate = (ret_total[:, -1] > 0).mean() * 100
        retirement_median_path = np.median(ret_total, axis=0)

        return {
            'success_rate': success_rate,
            'pre_retirement_path': pre_retirement_median_path,
            'retirement_path': retirement_median_path,
            'combined_table': combined_table,
            'initial_withdrawal_rate': initial_withdrawal_rate,
            'total_invested': total_invested_out_of_pocket
        }

    def generate_chart(self, results):
        fig = Figure(figsize=(10, 10))
        
        ax1 = fig.add_subplot(2, 1, 1)
        pre_ages = self.age_current + np.arange(len(results['pre_retirement_path']))
        ax1.plot(pre_ages, results['pre_retirement_path'], 'b-', linewidth=3, label='Accumulation Phase')
        
        ret_ages = self.age_retire + np.arange(len(results['retirement_path']))
        ax1.plot(ret_ages, results['retirement_path'], 'g-', linewidth=3, label='Retirement Phase')
        
        ax1.set_title('Median Total Portfolio Projection (10,000 Monte Carlo Simulations)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Age')
        ax1.set_ylabel('Total Portfolio Value ($)')
        ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2 = fig.add_subplot(2, 1, 2)
        table = results['combined_table']
        ret_table = [r for r in table if r['Age'] >= self.age_retire]
        
        if ret_table:
            ages = [r['Age'] for r in ret_table]
            withdrawals = [r['Withdrawals'] for r in ret_table]
            
            years_until_ret = self.age_retire - self.age_current
            benefits = [self.monthly_benefits * 12 * ((1 + self.inflation_rate)**(yr + years_until_ret)) for yr in range(len(ret_table))]
            budgets = [self.target_budget * ((1 + self.inflation_rate)**(yr + years_until_ret)) for yr in range(len(ret_table))]
            
            ax2.bar(ages, benefits, label='Fixed Benefits (Pension/SSN)', color='#8ecae6')
            ax2.bar(ages, withdrawals, bottom=benefits, label='Gross Portfolio Withdrawals (Includes Taxes)', color='#219ebc')
            ax2.plot(ages, budgets, 'r--', linewidth=2, label='Target Net Inflation-Adjusted Budget')
            
            ax2.set_title('Funding Your Retirement: Income vs Target Budget', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Age')
            ax2.set_ylabel('Annual Amount ($)')
            ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        fig.tight_layout(pad=3.0)
        img = io.BytesIO()
        canvas = FigureCanvas(fig)
        canvas.print_png(img)
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode()

@app.route('/')
def index():
    return render_template('calculator.html')

@app.route('/calculate_gap', methods=['POST'])
def calculate_gap():
    data = request.get_json(silent=True)
    name = data.get('name', 'User')
    age_current = int(data.get('age_current', 35))
    age_retire = int(data.get('age_retire', 65))
    monthly_budget = float(data.get('monthly_budget', 5000))
    monthly_benefits = float(data.get('monthly_benefits', 0))
    filing_status = data.get('filing_status', 'married')
    current_annual_income = float(data.get('current_annual_income', data.get('annual_income', monthly_budget * 12)))

    trad_start = float(data.get('portfolio_traditional', 0))
    roth_start = float(data.get('portfolio_roth', 0))
    tax_start = float(data.get('portfolio_taxable', 0))
    total_start = trad_start + roth_start + tax_start

    expected_return = float(data.get('expected_return', 7.0)) / 100
    inflation_rate = float(data.get('inflation_rate', 2.5)) / 100

    years_to_retire = max(0, age_retire - age_current)

    target_income_today = monthly_budget * 12
    current_effective_tax_rate = calculate_effective_tax(current_annual_income, filing_status)
    target_income_future = target_income_today * ((1 + inflation_rate) ** years_to_retire)
    projected_retirement_tax_rate = calculate_effective_tax(target_income_future, filing_status)

    future_portfolio = total_start * ((1 + expected_return) ** years_to_retire)
    blended_withdrawal_rate = 0.04
    projected_net_from_portfolio = (future_portfolio * blended_withdrawal_rate) * (1 - (projected_retirement_tax_rate * (trad_start / max(total_start, 1))))
    projected_income_future = projected_net_from_portfolio + (monthly_benefits * 12)

    shortfall = target_income_future - projected_income_future
    is_on_track = shortfall <= 0

    required_monthly_investment = 0
    if not is_on_track and years_to_retire > 0:
        target_additional_portfolio = shortfall / blended_withdrawal_rate

        rng = np.random.default_rng(None)
        portfolio_returns = rng.normal(expected_return, 0.12, size=(10000, years_to_retire))
        current_bals = np.zeros(10000)
        for yr in range(years_to_retire):
            current_bals = (current_bals + 1.0) * (1 + portfolio_returns[:, yr])

        median_multiplier = np.median(current_bals)
        if median_multiplier > 0:
            required_yearly = target_additional_portfolio / median_multiplier
            required_monthly_investment = float(required_yearly / 12)

    return jsonify({
        'success': True,
        'data': {
            'name': name,
            'expected_return': expected_return * 100,
            'inflation_rate': inflation_rate * 100,
            'current_effective_tax_rate': current_effective_tax_rate * 100,
            'projected_retirement_tax_rate': projected_retirement_tax_rate * 100,
            'effective_tax_rate': current_effective_tax_rate * 100,
            'target_income_future': target_income_future,
            'projected_income_future': projected_income_future,
            'shortfall': max(0, shortfall),
            'surplus': abs(shortfall) if shortfall < 0 else 0,
            'is_on_track': is_on_track,
            'required_monthly_investment': required_monthly_investment
        }
    })

@app.route('/calculate_advanced', methods=['POST'])
def calculate_advanced():
    try:
        data = request.get_json()

        investment_streams = []
        if 'investment_streams' in data:
            for stream_data in data['investment_streams']:
                stream = InvestmentStream(
                    name=stream_data['name'],
                    annual_contribution=float(stream_data['annual_contribution']),
                    tax_treatment=stream_data['tax_treatment']
                )
                investment_streams.append(stream)

        target_budget_today = float(data['monthly_budget']) * 12
        filing_status = data.get('filing_status', 'married')
        current_annual_income = float(data.get('current_annual_income', data.get('annual_income', target_budget_today)))
        current_effective_tax_rate = calculate_effective_tax(current_annual_income, filing_status)
        projected_retirement_tax_rate = calculate_effective_tax(target_budget_today * ((1 + float(data['inflation_rate']) / 100) ** max(0, int(data['age_retire']) - int(data['age_current']))), filing_status)

        calculator = EnhancedRetirementCalculator(
            age_current=int(data['age_current']),
            age_retire=int(data['age_retire']),
            trad_start=float(data.get('portfolio_traditional', 0)),
            roth_start=float(data.get('portfolio_roth', 0)),
            tax_start=float(data.get('portfolio_taxable', 0)),
            investment_streams=investment_streams,
            target_budget=target_budget_today,
            monthly_benefits=float(data['monthly_benefits']),
            expected_return=float(data['expected_return']) / 100,
            inflation_rate=float(data['inflation_rate']) / 100,
            effective_tax_rate=current_effective_tax_rate
        )

        results = calculator.calculate()
        chart = calculator.generate_chart(results)

        return jsonify({
            'success': True,
            'results': {
                'success_probability': f"{results['success_rate']:.1f}%",
                'chart': chart,
                'combined_table': results['combined_table'],
                'initial_withdrawal_rate': results['initial_withdrawal_rate'],
                'current_effective_tax_rate': current_effective_tax_rate * 100,
                'projected_retirement_tax_rate': projected_retirement_tax_rate * 100,
                'effective_tax_rate': current_effective_tax_rate * 100,
                'total_invested': results['total_invested']
            }
        })
    except Exception as e:
        logger.exception("Error in advanced calc")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
