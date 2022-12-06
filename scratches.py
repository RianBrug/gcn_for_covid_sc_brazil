# dict_regions_and_neighbors = utils.get_regions_and_neighbors(r_arr)

#   exemplo de uso: coletando a soma dos infectactos de 15 dias anteriores 
#   a 17/12/2021 (considerando o próprio dia) para a região SUL

#print(dados["SUL"]["2021-12-17"])

# print(f' dict_regins_and_neighbors:  {dict_regions_and_neighbors}')

#resultados do teu codigo + as meso regioes adjacentes
#formato: dict { data | regiao | infectados [data em questao~14d anteriores] | regioes_adjacentes }

# data example dict::: 
# dict_keys(['SUL', 'PLANALTO NORTE E NORDESTE', 'GRANDE FLORIANOPOLIS', 'MEIO OESTE E SERRA CATARINENSE', 'ALTO VALE DO ITAJAI', 'GRANDE OESTE', 'FOZ DO RIO ITAJAI', 'NULL'])
# dados {k=regioes, v=data, valor somatorio}
# dict_items([('SUL', {'2020-09-06': 2949, '2020-09-07': 2924, ...}), ... **(key{value})**, ]))
#            )                                                       
