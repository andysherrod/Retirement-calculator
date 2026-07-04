from flask import Flask, render_template, request, jsonify
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/')
def index():
    # We will build this HTML file next
    return render_template('calculator.html')

@app.route('/calculate_gap', methods=['POST'])
def calculate_gap():
    try:
        data = request.get_json(silent=True)
        if not data:
            raise ValueError('Invalid or missing JSON payload')

        # 1. Extract Simple Inputs
        name = data.get('name', 'User')
        age_current = int(data.get('age_current', 35))
        age_retire = int(data.get('age_retire', 65))
        yearly_income = float(data.get('yearly_income', 100000))
        portfolio_total = float(data.get('portfolio_total', 50000))
        monthly_benefits = float(data.get('monthly_benefits', 0))

        # 2. Financial Assumptions (These can be hardcoded for Phase 1 or passed via hidden HTML inputs)
        inflation_rate = float(data.get('inflation_rate', 2.5)) / 100
        expected_return = float(data.get('expected_return', 7.0)) / 100
        replacement_ratio = float(data.get('replacement_ratio', 80.0)) / 100
        withdrawal_rate = float(data.get('withdrawal_rate', 4.0)) / 100

        # 3. Time Horizon
        years_to_retire = max(0, age_retire - age_current)

        # 4. Calculate Target Retirement Income
        # How much they need to live on, adjusted for inflation over time
        target_income_today = yearly_income * replacement_ratio
        target_income_future = target_income_today * ((1 + inflation_rate) ** years_to_retire)

        # 5. Calculate Projected Retirement Income
        # Grow their current portfolio, then apply the 4% Safe Withdrawal Rate rule
        future_portfolio = portfolio_total * ((1 + expected_return) ** years_to_retire)
        portfolio_income = future_portfolio * withdrawal_rate
        
        # Add their annualized fixed benefits (SSN, Pension)
        annual_benefits = monthly_benefits * 12
        projected_income_future = portfolio_income + annual_benefits

        # 6. Calculate The Gap
        shortfall = target_income_future - projected_income_future
        is_on_track = shortfall <= 0

        # 7. Format Response
        return jsonify({
            'success': True,
            'data': {
                'name': name,
                'years_to_retire': years_to_retire,
                'target_income_future': target_income_future,
                'projected_income_future': projected_income_future,
                'future_portfolio': future_portfolio,
                'shortfall': max(0, shortfall),
                'surplus': abs(shortfall) if shortfall < 0 else 0,
                'is_on_track': is_on_track
            }
        })

    except ValueError as e:
        logger.error("Validation error: %s", e)
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.exception("Internal server error")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
