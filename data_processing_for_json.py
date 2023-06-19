from asyncio import constants
from datetime import date, timedelta
import time
from utils import Utils
import constants
import json
import pandas as pd
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

start_time = time.time()
# decide which path to follow
# you can skip the first part if you already have the confirmed_cases_by_region_and_date.json file and run_from_file_improved = True
run_load = False # if True, will load the csv file and create the json file
run_from_file = False # if True, will load the json file and run the model
mount_whole_csv_encoded = False # if True, will mount the whole csv encoded
run_from_file_improved = True # if True, will load the json file and run the model
home = '/Users/rcvb/Documents/tcc_rian/code'

if run_load:
    ## CHANGE TO YOUR OWN PATH
    ## Este conjunto de dados apresenta a relação de casos confirmados de COVID-19 no âmbito do Estado de Santa Catarina, conforme as recomendações da Open Knowlegde Foundation - Brasil (OKBR)
    ## Base de Dados do Governo do Estado - BOAVISTA
    f = open(f"{home}/assets/boavista_covid_dados_abertos.csv", "r")

    paciente = -1
    confirmed_cases_by_region_and_date = {} # dict region as a keys
    for linha in f:
        print(f'processing paciente n° {paciente}...')
    # '2022-06-11 16:00:07;SIM;2022-02-10;IGNORADO;FEBRE;;;NAO INTERNADO;NAO INTERNADO UTI;FEMININO;BOM RETIRO;NAO;NULL;27;MEIO OESTE E SERRA CATARINENSE;NAO INFOR'
        paciente += 1
        if paciente == 0: continue
        fields = str(linha).split(";")
        if len(fields) < 17: 
            continue
        if fields[13] == "NULL": # 
            continue
        data = str(fields[2]) # date yyyy-mm-dd
        if fields[14] not in confirmed_cases_by_region_and_date: # if macroregion does not exist
            confirmed_cases_by_region_and_date[fields[14]] = {} # dict dates as keys
        #para cada região, uma data corresponderá a janela da soma de infectados em cada dia anterior
        a_date = date.fromisoformat(data)
        #leitura de todas as colunas da linha
        for i in range(0,15):
            d = a_date + timedelta(days=i)
            novaData = d.isoformat()
            if novaData not in confirmed_cases_by_region_and_date[fields[14]]:
                confirmed_cases_by_region_and_date[fields[14]][novaData] = 0
            confirmed_cases_by_region_and_date[fields[14]][novaData] +=1 # adiciona somatorio dos casos confirmados na data

    # dump confirmed_cases_by_region_and_date dict to a json file
    with open(f"{home}/assets/confirmed_cases_by_region_and_date.json", "w") as outfile:
        json.dump(confirmed_cases_by_region_and_date, outfile)
    
    print("Process finished --- %s seconds ---" % round(time.time() - start_time))



elif run_from_file:
    dados = open("/Users/rcvb/Documents/tcc_rian/code/assets/confirmed_cases_by_region_and_date.json")
    # load confirmed_cases_by_region_and_date.json to a dataframe
    dados = json.load(dados)

    df = pd.DataFrame(dados)
    df.reset_index(inplace=True)
    df.rename(columns={'index':'collect_date'}, inplace=True)
    df.drop(df.columns[len(df.columns)-1], axis=1, inplace=True)
    df.fillna(0, inplace=True)
    melted_df = pd.melt(df, id_vars='collect_date', var_name='region', value_name='confirmed_cases')
    melted_df['collect_date'] = pd.to_datetime(melted_df['collect_date'], errors='coerce')
    melted_df['region'] = melted_df['region'].astype(str)
    melted_df['vizinhos'] = melted_df['region'].apply(lambda x: Utils.get_neighbors_of_region(x))
 
    melted_df['collect_date'] = melted_df['collect_date'].dt.strftime('%Y-%m-%d')
    melted_df.sort_values(by=['collect_date'], inplace=True)

    X = melted_df['confirmed_cases'].values
    # Y target variable should be the confirmed cases 15 days after the date of the input X
    # Y = melted_df['confirmed_cases'].shift(15).values.astype(float)
    Y = melted_df['confirmed_cases'].rolling(window=15, min_periods=1).sum().values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # Build the model
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(1,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(X_train, y_train, epochs=100000, batch_size=27, validation_data=(X_test, y_test))

    # Make predictions
    predictions = model.predict(X_test)

    # Flatten the predictions and ground truth arrays
    predictions = np.squeeze(predictions)
    y_test = np.squeeze(y_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # calculate accuracy between each prediction and test data
    accuracy = 0
    for i in range(len(predictions)):
        accuracy += abs(predictions[i] - y_test[i])
    accuracy = accuracy / len(predictions)

    print("Accuracy:", round(accuracy, 2))
    print("Mean Squared Error (MSE):", round(mse, 2))
    print("Mean Absolute Error (MAE):", round(mae, 2))
    print("R-squared (R2) Score:", r2)

    print("Process finished --- %s seconds ---" % round(time.time() - start_time))

    # melted_df['region_e'] = melted_df['region'].apply(lambda x: Utils.regional_str_to_encoded(x))
    # melted_df['region_e'] = melted_df['region_e'].to_numpy()
    # melted_df['vizinhos_e'] = melted_df['region'].apply(lambda x: Utils.get_encoded_neighbors_of_region(x))
    # melted_df['vizinhos_e'] = melted_df['vizinhos_e'].to_numpy()
    # melted_df['confirmed_cases'] = melted_df['confirmed_cases'].astype(float)

     # Assume 'melted_df' is the given DataFrame
    # Extract input (X) and output (Y) variables

elif run_from_file_improved:
    
    import json
    import pandas as pd
    from tensorflow import keras
    from tensorflow.keras import layers
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error, mean_squared_error
    import numpy as np


    # Load data from JSON file
    with open("/Users/rcvb/Documents/tcc_rian/code/assets/confirmed_cases_by_region_and_date.json") as file:
        dados = json.load(file)

    # Convert data to a DataFrame
    df = pd.DataFrame(dados)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'collect_date'}, inplace=True)
    df.drop(df.columns[len(df.columns) - 1], axis=1, inplace=True)
    df.fillna(0, inplace=True)
    
    evaluation_results = []

    # Melt the DataFrame to have "collect_date", "region", and "confirmed_cases" columns
    melted_df = pd.melt(df, id_vars='collect_date', var_name='region', value_name='confirmed_cases')
    melted_df['neighbors'] = melted_df['region'].apply(lambda x: Utils.get_neighbors_of_region(x))

    # Convert collect_date to datetime
    melted_df['collect_date'] = pd.to_datetime(melted_df['collect_date'], errors='coerce')

    # separate melted_df into a single DataFrame for each region
    region_dfs = []
    for region in melted_df['region'].unique():
        region_dfs.append(melted_df[melted_df['region'] == region])
    
    # Sort the DataFrames by collect_date
    for region_df in region_dfs:
        region_df.sort_values(by=['collect_date'], inplace=True)
        # target_cases by region
        # get first 15 rows of region_df['collect_date']['confirmed_cases'] to be X variable for training

        region_df['collect_date'] = region_df['collect_date'].apply(lambda x: x.timestamp())

        """ 

        region_df['target_cases'] = region_df['confirmed_cases'].shift(15).astype(float)
        region_df.dropna(subset=['target_cases'], inplace=True)
        # add region_df['target_cases'] to melted_df by region
        melted_df.loc[melted_df['region'] == region_df['region'].iloc[0], 'target_cases'] = region_df['target_cases']
    
    # drop rows with NaN values
    melted_df.dropna(inplace=True)

    # prepare X and Y variables inside a for loop for each region
    for region in melted_df['region'].unique():
        print("--------------------------------------------------")
        print(f"Training model for region {region}")
        print("--------------------------------------------------")
        print("--------------------------------------------------")
        X = melted_df[melted_df['region'] == region]['confirmed_cases'].values
        Y = melted_df[melted_df['region'] == region]['target_cases'].values
 """
        """ X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

        # Define the Keras model
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(1,)),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')

        # Train the model
        model.fit(X_train, y_train, epochs=1000, batch_size=63, validation_data=(X_test, y_test))

        # Make predictions
        predictions = model.predict(X_test)

        # Flatten the predictions and ground truth arrays
        predictions = np.squeeze(predictions)
        y_test = np.squeeze(y_test)

        # Calculate evaluation metrics
        scores = cross_val_score(model, X, Y, scoring='neg_mean_squared_error', cv=5)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        
        # calculate accuracy between each prediction and test data
        accuracy = 0
        for i in range(len(predictions)):
            accuracy += abs(predictions[i] - y_test[i])
        accuracy = accuracy / len(predictions)

        print("Accuracy:", round(accuracy, 2))
        print("Mean Squared Error (MSE):", round(mse, 2))
        print("Mean Absolute Error (MAE):", round(mae, 2))
        print("R-squared (R2) Score:", round(r2, 2))
        print("Cross Validation Score:", round(scores, 2))
        
        # initiate a variable to save results to after the loop we can compare the regions results
        evaluation_results.append({
            'region': region,
            'accuracy': round(accuracy, 2),
            'mse': round(mse, 2),
            'mae': round(mae, 2),
            'r2': round(r2, 2),
            'cross_validation_score': round(scores, 2)
        })
    print(evaluation_results)
    """
        """ 
        # Define the Keras model
        def create_model():
            model = keras.Sequential([
                layers.Dense(64, activation='relu', input_shape=(1,)),
                layers.Dropout(0.2),
                layers.Dense(64, activation='relu'),
                layers.Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            return model

        # Wrap the Keras model with the KerasRegressor
        # ignore deprecationwarning
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        model = KerasRegressor(build_fn=create_model, epochs=100, batch_size=32, verbose=0)

        # Perform cross-validation
        # kf = KFold(n_splits=5, shuffle=True, random_state=42)
        # scores = cross_val_score(model, X, Y, scoring='neg_mean_squared_error', cv=kf)
        cv_results = cross_val_score(model, X, Y, cv=10, scoring='r2')
        scores = cross_val_score(model, X, Y, scoring='neg_mean_squared_error', cv=10)
        mse_scores = -scores
        rmse_scores = np.sqrt(mse_scores)
        evaluation_results.append((region, rmse_scores.mean(), rmse_scores.std(), cv_results.mean(), cv_results.std()))

        # make predictions
        predictions = cross_val_predict(model, X, Y, cv=10)
        # print predictions
        print("--------------------------------------------------")
        print("--------------------------------------------------")
        print(f"Predictions: {predictions}")
        print("--------------------------------------------------")
        print("--------------------------------------------------")

        # calculate accuracy between each prediction and test data
        accuracy = 0
        for i in range(len(predictions)):
            accuracy += abs(predictions[i] - Y[i])
            accuracy = accuracy / len(predictions)
        
        # calculate evaluation metrics
        mse = mean_squared_error(Y, predictions)
        mae = mean_absolute_error(Y, predictions)
        r2 = r2_score(Y, predictions)
        print("--------------------------------------------------")
        print("--BY CROSS_VAL_PREDICT-------------------------------------")
        print("Accuracy:", round(accuracy, 2))
        print("Mean Squared Error (MSE):", round(mse, 2))
        print("Mean Absolute Error (MAE):", round(mae, 2))
        print("R-squared (R2) Score:", round(r2, 2))
        print("--------------------------------------------------")
        print("--------------------------------------------------")
        
        print("Mean Squared Error by cross_val_score: %.2f" % np.abs(scores).mean())
        print("Mean R2 Score by cross_val_score: %.2f" % np.abs(cv_results).mean())
        print("--------------------------------------------------")
        # inform errors rate
        print("--------------------------------------------------")
        
        # Print evaluation results
        for region, mean_score, std_score, cv_results, rmse_scores in evaluation_results:
            print(f"FROM EVALUTATION_RESULTS:::: Region: {region}, Mean RMSE: {mean_score}, Std RMSE: {std_score}, Mean CV R2 Results: {cv_results.mean()}")
        print("--------------------------------------------------")
        print("--------------------------------------------------")
        print("--------------------------------------------------")

        # implement a grid search to find the best parameters
        param_grid = {
            'epochs': [50, 100, 200, 300, 500, 1000],
            'batch_size': [7, 16, 21, 32, 42, 64, 128]
        }
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='r2', cv=5)
        grid_result = grid.fit(X, Y)
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        r2_results = grid_result.cv_results_['mean_test_score']
        
        print("R2 Scores:")
        for r2 in r2_results:
            print("%.3f" % r2)

        print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")

        for mean, std, param, r2_results in zip(means, stds, params, r2_results):
            print(f"GRID SEARCH::::  Mean: {mean}, Std: {std}, Params: {param}, R2 Results: {r2_results}")
        print("--------------------------------------------------")
        print("--------------------------------------------------")
        print("--------------------------------------------------")
 """


            # Collect the X variable
        X = []
        for i in range(15):
            X.append(region_df.iloc[i:i+15][['collect_date', 'confirmed_cases']].values.flatten())
            
        # Collect the Y variable
        Y = region_df.iloc[15:][['collect_date', 'confirmed_cases']].values.flatten()
        
        # X and Y 'collect_date' values are in datetime format, so we need to convert them something that can be used to train the model
        
        # Train the model
        model = RandomForestRegressor()
        model.fit(X, Y)

        # Predict future confirmed cases
        future_X = region_df.iloc[15:30][['collect_date', 'confirmed_cases']].values.flatten()
        future_Y = model.predict(future_X)

        # Define evaluation metrics
        def MDAPE(y_true, y_pred):
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        def Accuracy(y_true, y_pred):
            return 100 - MDAPE(y_true, y_pred)

        # Evaluate the model
        print('Mean Absolute Error:', mean_absolute_error(region_df.iloc[15:30]['confirmed_cases'], future_Y))
        print('R-Squared:', r2_score(region_df.iloc[15:30]['confirmed_cases'], future_Y))
        print('MDAPE:', MDAPE(region_df.iloc[15:30]['confirmed_cases'], future_Y))
        print('RMSE:', mean_squared_error(region_df.iloc[15:30]['confirmed_cases'], future_Y, squared=False))
        print('Accuracy:', Accuracy(region_df.iloc[15:30]['confirmed_cases'], future_Y))
        print('MAPE: ', mean_absolute_percentage_error(region_df.iloc[15:30]['confirmed_cases'], future_Y))
    print("Process finished --- %s seconds ---" % round(time.time() - start_time))
    print("--------------------------------------------------")
    
elif mount_whole_csv_encoded:
    removeSpace = lambda x: (x.replace('INTERNADO UTI',''))
    ignoradoToEmpty = lambda x: x.replace('IGNORADO','')

    df = pd.read_csv(f"{home}/assets/boavista_covid_dados_abertos.csv", sep=';', 
    true_values=["SIM", "CONFIRMADO", "INTERNADO"], low_memory=False,
    false_values=["NAO", "NAO INTERNADO"], 
    error_bad_lines=False, verbose=True, converters={'internacao_uti':removeSpace, 'collect_date':ignoradoToEmpty})
    
    df.drop(["data_publicacao", "sintomas", "comorbidades", "gestante", "municipio", "data_obito", "idade", "raca", 
    "codigo_ibge_municipio", "estado", "criterio_confirmacao", "municipio_notificacao", "codigo_ibge_municipio_notificacao",
    "latitude_notificacao", "longitude_notificacao", "classificacao", "nom_laboratorio", "data_internacao", "data_entrada_uti",
    "regional_saude", "data_evolucao_caso", "data_saida_uti", "bairro"], axis=1, inplace=True)
    df.dropna(inplace=True)
    df['data_inicio_sintomas'] = pd.to_datetime(df['data_inicio_sintomas'], errors='coerce')
    df['collect_date'] = pd.to_datetime(df['collect_date'], errors='coerce')
    df['data_resultado'] = pd.to_datetime(df['data_resultado'], errors='coerce')
    df['internacao_uti'].mask(df['internacao_uti'] == "NAO ", False, inplace=True)
    df['internacao_uti'].mask(df['internacao_uti'] == "", False, inplace=True)
    df['internacao_uti'] = df['internacao_uti'].astype(float)
    df['sexo'].mask(df['sexo'] == "FEMININO", 0, inplace=True)
    df['sexo'].mask(df['sexo'] == "MASCULINO", 1, inplace=True)
    df['sexo'].mask(df['sexo'] == "NAO INFORMADO", 0, inplace=True)
    df['sexo'] = df['sexo'].astype(float)
    df['recuperados'].mask(df['recuperados'] == True, 1, inplace=True)
    df['recuperados'].mask(df['recuperados'] == False, 0, inplace=True)
    df['recuperados'] = df['recuperados'].astype(float)
    df['internacao'].mask(df['internacao'] == True, 1, inplace=True)
    df['internacao'].mask(df['internacao'] == False, 0, inplace=True)
    df['internacao'] = df['internacao'].astype(float)
    df['internacao_uti'].mask(df['internacao_uti'] == True, 1, inplace=True)
    df['internacao_uti'].mask(df['internacao_uti'] == False, 0, inplace=True)
    df['obito'].mask(df['obito'] == True, 1, inplace=True)
    df['obito'].mask(df['obito'] == False, 0, inplace=True)
    df['obito'] = df['obito'].astype(float)
    df['origem_esus'].mask(df['origem_esus'] == True, 1, inplace=True)
    df['origem_esus'].mask(df['origem_esus'] == False, 0, inplace=True)
    df['origem_esus'] = df['origem_esus'].astype(float)
    df['origem_sivep'].mask(df['origem_sivep'] == True, 1, inplace=True)
    df['origem_sivep'].mask(df['origem_sivep'] == False, 0, inplace=True)
    df['origem_sivep'] = df['origem_sivep'].astype(float)
    df['origem_lacen'].mask(df['origem_lacen'] == True, 1, inplace=True)
    df['origem_lacen'].mask(df['origem_lacen'] == False, 0, inplace=True)
    df['origem_lacen'] = df['origem_lacen'].astype(float)
    df['origem_laboratorio_privado'].mask(df['origem_laboratorio_privado'] == True, 1, inplace=True)
    df['origem_laboratorio_privado'].mask(df['origem_laboratorio_privado'] == False, 0, inplace=True)
    df['origem_laboratorio_privado'] = df['origem_laboratorio_privado'].astype(float)
    df['fez_teste_rapido'].mask(df['fez_teste_rapido'] == True, 1, inplace=True)
    df['fez_teste_rapido'].mask(df['fez_teste_rapido'] == False, 0, inplace=True)
    df['fez_teste_rapido'] = df['fez_teste_rapido'].astype(float)
    df['fez_pcr'].mask(df['fez_pcr'] == True, 1, inplace=True)
    df['fez_pcr'].mask(df['fez_pcr'] == False, 0, inplace=True)
    df['fez_pcr'] = df['fez_pcr'].astype(float)
    df['regional_e'] = df['regional'].apply(lambda x: Utils.regional_str_to_encoded(x))
    df['regional_e'] = df['regional_e'].to_numpy()
    df['tipo_teste_e'] = df['tipo_teste'].apply(lambda x: Utils.tipoteste_str_to_encoded(x))
    df['tipo_teste_e'] = df['tipo_teste_e'].to_numpy()

    dados = open(f"{home}/assets/confirmed_cases_by_region_and_date.json")
    data = json.load(dados)

    
    df['vizinhos_e'] = df['vizinhos'].apply(lambda x: Utils.get_encoded_neighbors_of_region(x))

    df.drop(['recuperados', 'data_inicio_sintomas', 'internacao', 'internacao_uti', 'sexo', 'obito', 'data_resultado',
       'latitude', 'longitude', 'tipo_teste', 'origem_esus', 'origem_sivep', 'origem_lacen', 'origem_laboratorio_privado', 'fez_teste_rapido',
       'fez_pcr', 'tipo_teste_e'], axis=1, inplace=True)


    profile = ProfileReport(df, explorative=True, minimal=False)
    try:
        profile.to_widgets()         # view as widget in Notebook
    except:
        profile.to_file('df.html')

    print("Process finished --- %s seconds ---" % round(time.time() - start_time))

else:

    # enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS = Utils.transform_categorical_to_one_hot_encoding(constants.REGIONS_SEQ_AND_ITS_NEIGHBORS)
    encoded_regions = Utils.transform_categorical_to_one_hot_encoding(constants.REGIONS_INDEX)


    # formated_regions_and_neighbors = (enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[0], [enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[7], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[8]],
    #    enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[1], [enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[9], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[10], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[11]],
    #    enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[2], [enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[14], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[15], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[16], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[17]],
    #    enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[3], [enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[18], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[19], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[20], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[21]],
    #    enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[4], [enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[22], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[23], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[24]],
    #    enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[5], [enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[21]],
    #    enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[6], [enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[26], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[27], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[28], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[29]])

    # G = nx.from_pandas_edgelist(df, source='regional', target='obito', edge_attr=None, create_using=nx.DiGraph())
    profile = ProfileReport(df, explorative=True, minimal=False)
    try:
        profile.to_widgets()         # view as widget in Notebook
    except:
        profile.to_file('df.html')

    print("Process finished --- %s seconds ---" % round(time.time() - start_time))