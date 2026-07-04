from flask import Flask, render_template, request, jsonify
import pandas as pd
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

@dataclass
class InvestmentStream:
    name: str
    annual_contribution: float
    tax_treatment: str

class EnhancedRetirementCalculator:
    def __init__(self, age_current, age_retire, portfolio_total, investment_streams, 
                 stock_allocation, bond_allocation, target_budget, monthly_benefits,
                 expected_return, inflation_rate):
        self.age_current = age_current
        self.age_retire = age_retire
        self.life_expectancy = 90
        self.portfolio_total = portfolio_total
        self.investment_streams = investment_streams
        self.stock_allocation = stock_allocation / 100
        self.bond_allocation = bond_allocation / 100
        self.target_budget = target_budget
        self.monthly_benefits = monthly_benefits
        self.expected_return = expected_return
        self.inflation_rate = inflation_rate
        self.stock_volatility = 0.16
        self.bond_volatility = 0.05
        self.correlation = -0.1
        self.rng = np.random.default_rng(None)
        self.num_simulations = 10000

    def generate_correlated_returns(self, num_simulations, num_years):
        cov = [[self.stock_volatility**2, self.correlation * self.stock_volatility * self.bond_volatility],
               [self.correlation * self.stock_volatility * self.bond_volatility, self.bond_volatility**2]]
        random_returns = self.rng.multivariate_normal([self.expected_return, 0.04], cov, size=(num_simulations, num_years))
        return random_returns[:, :, 0], random_returns[:, :, 1]

    def calculate(self):
        years_until_retirement = self.age_retire - self.age_current
        retirement_years = self.life_expectancy - self.age_retire
        
        total_yearly_investment = sum(stream.annual_contribution for stream in self.investment_streams)

        # 1. Deterministic Path (For Table)
        combined_table = []
        current_port = self.portfolio_total
        portfolio_expected_return = (self.stock_allocation * self.expected_return) + (self.bond_allocation * 0.04)

        for year in range(years_until_retirement):
            beginning = current_port
            contribution = total_yearly_investment
            interest = (beginning + contribution) * portfolio_expected_return
            current_port = beginning + contribution + interest
            
            combined_table.append({
                'Year': year,
                'Age': self.age_current + year,
                'Investment_Amount': beginning,
                'Contributions': contribution,
                'Interest_Earned': interest,
                'Withdrawals': 0.0,
                'Ending_Balance': current_port
            })

        future_portfolio_deterministic = current_port

        for year in range(retirement_years):
            beginning = current_port
            annual_benefit = self.monthly_benefits * 12 * ((1 + self.inflation_rate) ** (year + years_until_retirement))
            annual_budget = self.target_budget * ((1 + self.inflation_rate) ** (year + years_until_retirement))
            withdrawal = max(0, annual_budget - annual_benefit)
            
            if beginning - withdrawal < 0:
                withdrawal = beginning
                interest = 0
                current_port = 0
            else:
                interest = (beginning - withdrawal) * portfolio_expected_return
                current_port = beginning - withdrawal + interest

            combined_table.append({
                'Year': years_until_retirement + year,
                'Age': self.age_retire + year,
                'Investment_Amount': beginning,
                'Contributions': 0.0,
                'Interest_Earned': interest,
                'Withdrawals': withdrawal,
                'Ending_Balance': current_port
            })

        # 2. Monte Carlo Simulation
        stock_ret, bond_ret = self.generate_correlated_returns(self.num_simulations, years_until_retirement)
        portfolio_returns = self.stock_allocation * stock_ret + self.bond_allocation * bond_ret
        
        portfolios = np.full((self.num_simulations, years_until_retirement + 1), self.portfolio_total, dtype=np.float64)
        for year in range(years_until_retirement):
            portfolios[:, year + 1] = (portfolios[:, year] + total_yearly_investment) * (1 + portfolio_returns[:, year])
        
        future_portfolios = portfolios[:, -1]
        pre_retirement_median_path = np.median(portfolios, axis=0)

        stock_ret_ret, bond_ret_ret = self.generate_correlated_returns(self.num_simulations, retirement_years)
        portfolio_returns_ret = self.stock_allocation * stock_ret_ret + self.bond_allocation * bond_ret_ret
        
        ret_portfolios = np.full((self.num_simulations, retirement_years + 1), 0.0, dtype=np.float64)
        ret_portfolios[:, 0] = future_portfolios
        
        for year in range(retirement_years):
            annual_benefit = self.monthly_benefits * 12 * ((1 + self.inflation_rate) ** (year + years_until_retirement))
            annual_budget = self.target_budget * ((1 + self.inflation_rate) ** (year + years_until_retirement))
            withdrawals = np.maximum(0, annual_budget - annual_benefit)
            
            after_withdrawal = ret_portfolios[:, year] - withdrawals
            ret_portfolios[:, year + 1] = np.maximum(0, after_withdrawal * (1 + portfolio_returns_ret[:, year]))
            depleted = ret_portfolios[:, year + 1] <= 0
            ret_portfolios[depleted, year + 1:] = 0

        success_rate = (ret_portfolios[:, -1] > 0).mean() * 100
        retirement_median_path = np.median(ret_portfolios, axis=0)

        return {
            'success_rate': success_rate,
            'pre_retirement_path': pre_retirement_median_path,
            'retirement_path': retirement_median_path,
            'combined_table': combined_table
        }

    def generate_chart(self, results):
        fig = Figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1)
        
        pre_ages = self.age_current + np.arange(len(results['pre_retirement_path']))
        ax.plot(pre_ages, results['pre_retirement_path'], 'b-', linewidth=3, label='Accumulation Phase')
        
        ret_ages = self.age_retire + np.arange(len(results['retirement_path']))
        ax.plot(ret_ages, results['retirement_path'], 'g-', linewidth=3, label='Retirement Phase')
        
        ax.set_title('Median Portfolio Projection (Monte Carlo)', fontsize=14)
        ax.set_xlabel('Age')
        ax.set_ylabel('Portfolio Value ($)')
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        fig.tight_layout()
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
    portfolio_total = float(data.get('portfolio_total', 50000))
    monthly_benefits = float(data.get('monthly_benefits', 0))
    
    expected_return = float(data.get('expected_return', 7.0)) / 100
    inflation_rate = float(data.get('inflation_rate', 2.5)) / 100

    withdrawal_rate = 0.04
    years_to_retire = max(0, age_retire - age_current)

    # 1. Target Income Calculation based on Monthly Budget
    target_income_today = monthly_budget * 12
    target_income_future = target_income_today * ((1 + inflation_rate) ** years_to_retire)

    # 2. Projected Income
    future_portfolio = portfolio_total * ((1 + expected_return) ** years_to_retire)
    portfolio_income = future_portfolio * withdrawal_rate
    projected_income_future = portfolio_income + (monthly_benefits * 12)

    shortfall = target_income_future - projected_income_future
    is_on_track = shortfall <= 0

    # 3. Monte Carlo: Required Monthly Investment
    required_monthly_investment = 0
    if not is_on_track and years_to_retire > 0:
        target_additional_portfolio = shortfall / withdrawal_rate
        
        rng = np.random.default_rng(None)
        num_sims = 10000
        stock_vol, bond_vol, corr = 0.16, 0.05, -0.1
        cov = [[stock_vol**2, corr * stock_vol * bond_vol],
               [corr * stock_vol * bond_vol, bond_vol**2]]
        
        # Generate returns for default 70/30 allocation to calculate the multiplier
        random_returns = rng.multivariate_normal([expected_return, 0.04], cov, size=(num_sims, years_to_retire))
        portfolio_returns = 0.7 * random_returns[:, :, 0] + 0.3 * random_returns[:, :, 1]
        
                # Find the FV Multiplier of a $1 yearly investment
        current_bals = np.zeros(num_sims)
        for yr in range(years_to_retire):
            current_bals = (current_bals + 1.0) * (1 + portfolio_returns[:, yr])
        
        median_multiplier = np.median(current_bals)
        if median_multiplier > 0:
            required_yearly = target_additional_portfolio / median_multiplier
            # FIX: Wrap this in float() to make it JSON serializable
            required_monthly_investment = float(required_yearly / 12) 

    return jsonify({
        'success': True,
        'data': {
            'name': name,
            'expected_return': expected_return * 100,
            'inflation_rate': inflation_rate * 100,
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

        # Updated to translate monthly budget into the target annual budget
        target_budget_today = float(data['monthly_budget']) * 12

        calculator = EnhancedRetirementCalculator(
            age_current=int(data['age_current']),
            age_retire=int(data['age_retire']),
            portfolio_total=float(data['portfolio_total']),
            investment_streams=investment_streams,
            stock_allocation=float(data['stock_allocation']),
            bond_allocation=float(data['bond_allocation']),
            target_budget=target_budget_today,
            monthly_benefits=float(data['monthly_benefits']),
            expected_return=float(data['expected_return']) / 100,
            inflation_rate=float(data['inflation_rate']) / 100
        )

        results = calculator.calculate()
        chart = calculator.generate_chart(results)

        return jsonify({
            'success': True,
            'results': {
                'success_probability': f"{results['success_rate']:.1f}%",
                'chart': chart,
                'combined_table': results['combined_table']
            }
        })
    except Exception as e:
        logger.exception("Error in advanced calc")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
