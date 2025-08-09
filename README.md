# Enhanced Retirement Calculator Flask App

## Setup Instructions

### 1. Create Project Structure

```
retirement_calculator/
│
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── templates/
│   └── calculator.html    # HTML template
└── static/               # Optional: for additional CSS/JS files
```

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv retirement_env

# Activate virtual environment
# On Windows:
retirement_env\Scripts\activate
# On macOS/Linux:
source retirement_env/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. File Setup

1. **Save the Flask app code** as `app.py`
1. **Create a `templates` folder** and save the HTML code as `templates/calculator.html`
1. **Save the requirements.txt** file in the root directory

### 4. Run the Application

```bash
python app.py
```

The application will be available at `http://127.0.0.1:5000`

Web Link: https://retirement-calculator-ealv.onrender.com/

## Features Implemented

### From Your Original Jupyter Calculator:

- ✅ Pre-retirement portfolio growth projections
- ✅ Monte Carlo simulation (1,000 iterations)
- ✅ Multiple retirement spending phases
- ✅ Healthcare cost modeling with accelerated inflation
- ✅ Tax treatment for different account types
- ✅ Comprehensive visualization with 4 charts
- ✅ Year-by-year detailed calculations
- ✅ Success probability analysis

### Enhanced Web Features:

- 📱 Responsive design for mobile devices
- 🎨 Modern, professional interface
- ⚡ Real-time form validation
- 📊 Interactive chart generation
- 💡 Helpful tooltips for all parameters
- 🔄 Smooth animations and transitions

## Key Differences from Jupyter Version

1. **Charts**: Generated as PNG images (base64 encoded) instead of interactive matplotlib
1. **Data Tables**: Summarized in key metrics instead of full DataFrame display
1. **Interface**: Web form instead of ipywidgets sliders
1. **Validation**: Built-in form validation for better user experience

## Deployment Options

### Option 1: Local Development

- Run with `python app.py` for local testing

### Option 2: Production Deployment

- Deploy to platforms like Heroku, PythonAnywhere, or DigitalOcean
- Use a production WSGI server like Gunicorn
- Add environment variables for configuration

### Option 3: Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

## Customization Ideas

1. **Add Database Storage**: Save calculations for user sessions
1. **Export Features**: PDF reports, Excel downloads
1. **Comparison Tool**: Compare multiple scenarios
1. **Advanced Charts**: Interactive Plotly charts
1. **Authentication**: User accounts and saved plans
1. **API Endpoints**: RESTful API for external integrations

## Performance Notes

- Monte Carlo simulation runs 1,000 iterations (adjustable)
- Chart generation may take 2-3 seconds for complex scenarios
- Memory usage is optimized for web deployment
- All calculations run server-side for security
