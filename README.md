# GCN Network Optimization

This project aims to optimize a Graph Convolutional Network (GCN) model using Optuna for predicting confirmed COVID-19 cases in different regions. The model utilizes additional features along with the historical data to improve the prediction accuracy.

## Dataset

The dataset used in this project is sourced from the "confirmed_cases_by_region_and_date.json" file, which contains the daily confirmed COVID-19 cases for various regions. Additional data such as population, hospital resources, and water supply information are also used as features.

## Prerequisites

To run the project, you need to have the following software installed:

- Python (version 3.9.4)
- pyenv (version 2.3.17)

## Installation

To run this project, please ensure that you have the following dependencies installed:

- Python 3.9 or above
- PyTorch
- Optuna
- NetworkX
- Pandas
- NumPy
- scikit-learn
- torch-geometric

## Usage

1. Setup a virtual environment using pyenv (optional but recommended):
    
```bash
pyenv virtualenv 3.9.4 gcn
pyenv local gcn
```

2. Clone the repository:

```bash
git clone https://github.com/RianBrug/gcn_for_covid_sc_brazil.git
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Navigate to the project directory:

```bash
cd gcn_for_covid_sc_brazil
```

5. Prepare the dataset:

   - Ensure that the "confirmed_cases_by_region_and_date.json" file is located in the "assets" folder.
   - Additional feature files should be placed in the "assets" folder as well.


6. Active the virtual environment (if applicable):

```bash
pyenv activate gcn
```

7. Run the optimization:

```bash
python main.py
```

8. Explore the results:

   - The optimization history, intermediate values, high-dimensional parameter relationships, and parameter importances will be saved as images.
   - The best parameters found by Optuna will be displayed in the console output.

## Contributing

Contributions to this project are welcome. Feel free to submit bug reports, feature requests, or pull requests.

## License

This project is licensed under Open Access Licenses.
```
