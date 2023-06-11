# GCN Network Optimization for COVID-19 Predictions

This project optimizes a Graph Convolutional Network (GCN) model with Optuna to predict COVID-19 cases in various regions, incorporating additional features alongside historical data for improved accuracy.

## Table of Contents

- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Dataset

The primary dataset "confirmed_cases_by_region_and_date.json" contains daily confirmed COVID-19 cases across multiple regions. Complementary datasets for population demographics, hospital resources, and water supply information are also utilized as features.

## Prerequisites

Ensure the installation of the following software:

- Python 3.9.4 or above
- Pyenv 2.3.17 or above

## Installation

After installing Python and Pyenv, clone the repository and install the necessary Python packages. Use the commands below:

```bash
# Create a virtual environment (optional but recommended)
pyenv virtualenv 3.9.4 gcn
pyenv local gcn

# Clone the repository
git clone https://github.com/RianBrug/gcn_for_covid_sc_brazil.git

# Navigate to the project directory
cd gcn_for_covid_sc_brazil

# Install the required packages
pip install -r requirements.txt
```

## Usage
Prepare the dataset: Ensure "confirmed_cases_by_region_and_date.json" and additional feature files are located in the "assets" folder.

Activate the virtual environment (if applicable):

```bash
pyenv activate gcn
Run the optimization:

python gcn_network.py
or
python gat_network.py
```
## Analyze the results: Optuna Dashboard
Optuna comes with an interactive dashboard, which provides a rich interface for visualizing the optimization process. This can be very useful for understanding and interpreting the model's behavior and performance.

To launch the dashboard:

1. Start the Optuna dashboard server with your optimization database:

    ```bash
    optuna dashboard --storage sqlite:///example.db
    ```

    Replace "example.db" with the name of your SQLite database file that was used for storing the optimization results.

2. Open your web browser and navigate to the displayed URL (usually `localhost:8008`). The Optuna Dashboard will appear, displaying a variety of interactive plots about your optimization process. 

Please note that the Optuna Dashboard is read-only and does not support modifying the database. Always remember to save and backup your database.

## Contributing
Contributions to the project are always welcome. Feel free to submit bug reports, feature requests, or pull requests.

## License
This project is licensed under Open Access Licenses.