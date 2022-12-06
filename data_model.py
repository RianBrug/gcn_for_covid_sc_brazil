from asyncio import constants
from datetime import date, timedelta
import numpy
import time
from utils import Utils
import constants
import json
import pandas as pd
import networkx as nx
import matplotlib
from pandas_profiling import ProfileReport

start_time = time.time()
run_load = False
run_from_file = False

if run_load:
    ## CHANGE TO YOUR OWN PATH
    ## Este conjunto de dados apresenta a relação de casos confirmados de COVID-19 no âmbito do Estado de Santa Catarina, conforme as recomendações da Open Knowlegde Foundation - Brasil (OKBR)
    ## Base de Dados do Governo do Estado - BOAVISTA
    f = open("/Users/rcvb/Documents/tcc_rian/code/assets/boavista_covid_dados_abertos.csv", "r")

    paciente = -1
    dados = {} # dict region as a keys
    utils = Utils()
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
        if fields[14] not in dados: # if macroregion does not exist
            dados[fields[14]] = {} # dict dates as keys
        #para cada região, uma data corresponderá a janela da soma de infectados em cada dia anterior
        a_date = date.fromisoformat(data)
        #leitura de todas as colunas da linha
        for i in range(0,15):
            d = a_date + timedelta(days=i)
            novaData = d.isoformat()
            if novaData not in dados[fields[14]]:
                dados[fields[14]][novaData] = 0
            dados[fields[14]][novaData] +=1 # adiciona somatorio dos casos confirmados na data
            
                    

elif run_from_file:
    dados = open("/Users/rcvb/Documents/tcc_rian/code/assets/dados_processads.json")

else:
    # df = pd.read_csv("/Users/rcvb/Documents/tcc_rian/code/assets/boavista_covid_dados_abertos.csv", error_bad_lines=False, sep=';')
    
    removeSpace = lambda x: (x.replace('INTERNADO UTI',''))
    ignoradoToEmpty = lambda x: x.replace('IGNORADO','')

    df_cols = pd.read_csv("/Users/rcvb/Documents/tcc_rian/code/assets/boavista_covid_dados_abertos.csv", sep=';', 
    true_values=["SIM", "CONFIRMADO", "INTERNADO"], low_memory=False,
    false_values=["NAO", "NAO INTERNADO"], 
    error_bad_lines=False, verbose=True, converters={'internacao_uti':removeSpace, 'data_coleta':ignoradoToEmpty})
    
    df_cols.drop(["data_publicacao", "sintomas", "comorbidades", "gestante", "municipio", "data_obito", "idade", "raca", 
    "codigo_ibge_municipio", "estado", "criterio_confirmacao", "municipio_notificacao", "codigo_ibge_municipio_notificacao",
    "latitude_notificacao", "longitude_notificacao", "classificacao", "nom_laboratorio", "data_internacao", "data_entrada_uti",
    "regional_saude", "data_evolucao_caso", "data_saida_uti", "bairro"], axis=1, inplace=True)
    df_cols.dropna(inplace=True)
    df_cols['internacao_uti'].mask(df_cols['internacao_uti'] == "NAO ", False, inplace=True)
    df_cols['sexo'].mask(df_cols['sexo'] == "FEMININO", 0, inplace=True)
    df_cols['sexo'].mask(df_cols['sexo'] == "MASCULINO", 1, inplace=True)
    df_cols['recuperados'].mask(df_cols['recuperados'] == True, 1, inplace=True)
    df_cols['recuperados'].mask(df_cols['recuperados'] == False, 0, inplace=True)
    df_cols['internacao'].mask(df_cols['internacao'] == True, 1, inplace=True)
    df_cols['internacao'].mask(df_cols['internacao'] == False, 0, inplace=True)
    df_cols['internacao_uti'].mask(df_cols['internacao_uti'] == True, 1, inplace=True)
    df_cols['internacao_uti'].mask(df_cols['internacao_uti'] == False, 0, inplace=True)
    df_cols['obito'].mask(df_cols['obito'] == True, 1, inplace=True)
    df_cols['obito'].mask(df_cols['obito'] == False, 0, inplace=True)
    df_cols['origem_esus'].mask(df_cols['origem_esus'] == True, 1, inplace=True)
    df_cols['origem_esus'].mask(df_cols['origem_esus'] == False, 0, inplace=True)
    df_cols['origem_sivep'].mask(df_cols['origem_sivep'] == True, 1, inplace=True)
    df_cols['origem_sivep'].mask(df_cols['origem_sivep'] == False, 0, inplace=True)
    df_cols['origem_lacen'].mask(df_cols['origem_lacen'] == True, 1, inplace=True)
    df_cols['origem_lacen'].mask(df_cols['origem_lacen'] == False, 0, inplace=True)
    df_cols['origem_laboratorio_privado'].mask(df_cols['origem_laboratorio_privado'] == True, 1, inplace=True)
    df_cols['origem_laboratorio_privado'].mask(df_cols['origem_laboratorio_privado'] == False, 0, inplace=True)
    df_cols['fez_teste_rapido'].mask(df_cols['fez_teste_rapido'] == True, 1, inplace=True)
    df_cols['fez_teste_rapido'].mask(df_cols['fez_teste_rapido'] == False, 0, inplace=True)
    df_cols['fez_pcr'].mask(df_cols['fez_pcr'] == True, 1, inplace=True)
    df_cols['fez_pcr'].mask(df_cols['fez_pcr'] == False, 0, inplace=True)
    df_cols['regional_e'] = df_cols['regional'].apply(lambda x: Utils.regional_str_to_encoded(x))
    df_cols['tipo_teste_e'] = df_cols['tipo_teste'].apply(lambda x: Utils.tipoteste_str_to_encoded(x))
    #df_cols['tipo_teste'].mask(df_cols['tipo_teste'] == False, 0, inplace=True)


# regions_keys = dados.keys()
# r_list = list(regions_keys)
# remove null values from array
# r_list.pop()
# r_arr = numpy.array(r_list)
#lambda argument(s): expression

print("Processed all lines in --- %s seconds ---" % round((time.time() - start_time), 2))

enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS = Utils.transform_categorical_to_one_hot_encoding(constants.REGIONS_SEQ_AND_ITS_NEIGHBORS)

# formated_regions_and_neighbors = {
    # enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[0]: [enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[7], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[8]],
    # enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[1]: [enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[9], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[10], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[11]],
    # enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[2]: [enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[14], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[15], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[16], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[17]],
    # enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[3]: [enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[18], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[19], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[20], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[21]],
    # enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[4]: [enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[22], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[23], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[24]],
    # enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[5]: [enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[21]],
    # enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[6]: [enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[26], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[27], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[28], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[29]]
# }

formated_regions_and_neighbors = (enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[0], [enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[7], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[8]],
    enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[1], [enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[9], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[10], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[11]],
    enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[2], [enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[14], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[15], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[16], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[17]],
    enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[3], [enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[18], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[19], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[20], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[21]],
    enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[4], [enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[22], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[23], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[24]],
    enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[5], [enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[21]],
    enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[6], [enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[26], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[27], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[28], enconded_REGIONS_SEQ_AND_ITS_NEIGHBORS[29]])


# G = nx.from_pandas_edgelist(df, source='regional', target='obito', edge_attr=None, create_using=nx.DiGraph())
profile = ProfileReport(df_cols, explorative=True, minimal=False)
try:
    profile.to_widgets()         # view as widget in Notebook
except:
    profile.to_file('df_cols.html')

print("Process finished --- %s seconds ---" % round(time.time() - start_time))