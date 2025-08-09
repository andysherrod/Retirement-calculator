from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
CORS(app)

class EnhancedRetirementCalculator:
    def __init__(self, starting_balance, annual_contribution, retirement_budget_ratio,
                 retirement_age, current_age, life_expectancy, expected_return,
                 inflation_rate, healthcare_cost, monthly_benefit, stock_allocation=0.6):
        self.starting_balance = starting_balance
        self.annual_contribution = annual_contribution
        self.retirement_budget_ratio = retirement_budget_ratio
        self.retirement_age = retirement_age
        self.current_age = current_age
        self.life_expectancy = life_expectancy
        self.expected_return = expected_return
        self.inflation_rate = inflation_rate
        self.healthcare_cost = healthcare_cost
        self.monthly_benefit = monthly_benefit
        self.stock_allocation = stock_allocation  # e.g. 0.6 = 60% stocks, 40% bonds

        # Historical volatilities (approximate)
        self.stock_volatility = 0.16
        self.bond_volatility = 0.05

    def monte_carlo_simulation(self, years, num_simulations=1000):
        """Vectorized Monte Carlo simulation for speed"""
        total_years = years
        balances = np.zeros((num_simulations, total_years + 1))
        balances[:, 0] = self.starting_balance

        # Precompute annual vol based on allocation
        portfolio_volatility = np.sqrt(
            (self.stock_allocation * self.stock_volatility) ** 2 +
            ((1 - self.stock_allocation) * self.bond_volatility) ** 2
        )

        for year in range(1, total_years + 1):
            # Random returns vectorized for all simulations at once
            returns = np.random.normal(self.expected_return, portfolio_volatility, num_simulations)

            withdrawals = np.where(
                year + self.current_age > self.retirement_age,
                self.retirement_budget_ratio * balances[:, year - 1],
                -self.annual_contribution
            )

            balances[:, year] = (balances[:, year - 1] - withdrawals) * (1 + returns)

        return balances

    def run_simulation(self, num_simulations=1000):
        years = self.life_expectancy - self.current_age
        balances = self.monte_carlo_simulation(years, num_simulations)

        # Success rate: % of simulations with money left at the end
        success_rate = np.mean(balances[:, -1] > 0) * 100

        # Create charts
        charts = {
            "projection_chart": self._create_projection_chart(balances),
            "distribution_chart": self._create_distribution_chart(balances[:, -1])
        }

        return {
            "success_rate": round(success_rate, 2),
            "charts": charts
        }

    def _create_projection_chart(self, balances):
        plt.figure(figsize=(8, 5))
        for i in range(min(50, balances.shape[0])):  # Plot first 50 runs
            plt.plot(balances[i], color="lightblue", alpha=0.3)
        plt.xlabel("Years")
        plt.ylabel("Portfolio Value (£)")
        plt.title("Monte Carlo Projection (Sample Runs)")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()
        return img_base64

    def _create_distribution_chart(self, ending_balances):
        plt.figure(figsize=(8, 5))
        plt.hist(ending_balances, bins=30, color="skyblue", edgecolor="black")
        plt.xlabel("Ending Portfolio Value (£)")
        plt.ylabel("Frequency")
        plt.title("Distribution of Ending Portfolio Values")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()
        return img_base64


@app.route("/calculate", methods=["POST"])
def calculate():
    data = request.json

    calc = EnhancedRetirementCalculator(
        starting_balance=data.get("starting_balance", 500000),
        annual_contribution=data.get("annual_contribution", 20000),
        retirement_budget_ratio=data.get("retirement_budget_ratio", 0.04),
        retirement_age=data.get("retirement_age", 65),
        current_age=data.get("current_age", 45),
        life_expectancy=data.get("life_expectancy", 90),
        expected_return=data.get("expected_return", 0.06),
        inflation_rate=data.get("inflation_rate", 0.02),
        healthcare_cost=data.get("healthcare_cost", 5000),
        monthly_benefit=data.get("monthly_benefit", 1500),
        stock_allocation=data.get("stock_allocation", 0.6)  # NEW: allow UI to set allocation
    )

    results = calc.run_simulation(num_simulations=data.get("num_simulations", 1000))
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True)
