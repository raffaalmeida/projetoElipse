
import pandas as pd
from datetime import datetime
import numpy as np
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from funcs import create_features, separar_treino_teste
from funcs import DataFrameHandler

handler = DataFrameHandler()
maindf = handler.get_dataframe()

class ReservoirModel:
    def __init__(self, vazao_prev, entrada_agua, current_level, start_hour):
        self.N_STATES = 101  # Níveis do reservatório de 0% a 100%
        self.N_ACTIONS = 2   # 0: desligar a bomba, 1: ligar a bomba
        self.N_HOURS = 24    # 24 horas no dia
        self.H_PONTA = range(18, 21)  # Horário de ponta das 18h às 21h

        self.PENALIDADE_PONTA = 20
        self.PENALIDADE_NORMAL = 1
        self.PENALIDADE_NIVEL = 20
        self.PENALIDADE_TROCA = 5
        self.RECOMPENSA_NIVEL = 1

        self.vazao_prev = np.array(vazao_prev)
        self.entrada_agua = entrada_agua
        self.current_level = current_level
        self.start_hour = start_hour

    def recompensa(self, nivel, acao, hora, acao_anterior):
        if nivel < 20 or nivel > 70:
            return -self.PENALIDADE_NIVEL
        penalidade = self.PENALIDADE_PONTA if hora in self.H_PONTA else self.PENALIDADE_NORMAL
        penalidade_troca = self.PENALIDADE_TROCA if acao_anterior == 0 and acao == 1 else 0
        return self.RECOMPENSA_NIVEL - (penalidade * acao) - penalidade_troca

    def transicao(self, nivel, acao, vazao, entrada_agua):
        return max(0, min(100, nivel - vazao + entrada_agua * acao))

    def dynamic_min_level(self, hour, vazao_prev, future_hours=1):
        next_hours = hour + future_hours
        future_out = sum(vazao_prev[hour:next_hours])
        return 20 + future_out  # 30% da capacidade total + previsão de saída

    def q_learning(self, episodes=100000, alpha=0.1, gamma=0.9, epsilon=0.1):
        Q = np.zeros((self.N_STATES, self.N_HOURS, self.N_ACTIONS))

        for episode in range(episodes):
            nivel = np.random.randint(0, 101)
            acao_anterior = 0
            for h in range(self.N_HOURS):
                if np.random.uniform(0, 1) < epsilon:
                    acao = np.random.randint(0, self.N_ACTIONS)
                else:
                    acao = np.argmax(Q[nivel, h])

                proximo_nivel = self.transicao(nivel, acao, self.vazao_prev[h], self.entrada_agua)
                dynamic_min = self.dynamic_min_level(h, self.vazao_prev)
                if proximo_nivel < dynamic_min:
                    r = -self.PENALIDADE_NIVEL  # Penalidade alta se nível cair abaixo do limite mínimo dinâmico
                else:
                    r = self.recompensa(nivel, acao, h, acao_anterior)
                Q[nivel, h, acao] += alpha * (r + gamma * np.max(Q[int(proximo_nivel), (h + 1) % self.N_HOURS]) - Q[nivel, h, acao])
                nivel = proximo_nivel
                acao_anterior = acao

        return Q

    def create_schedule_hours(self):
        return [(self.start_hour + i) % self.N_HOURS for i in range(self.N_HOURS)]

    def simulate(self):
        schedule_hours = self.create_schedule_hours()
        Q = self.q_learning()
        politica_otima = np.argmax(Q, axis=2)

        nivel_reservatorio = self.current_level
        nivel_reservatorio_hist = [nivel_reservatorio]
        acoes = []
        custo_energia = 0
        acao_anterior = 0

        for t in range(self.N_HOURS):
            h = schedule_hours[t]
            acao = politica_otima[nivel_reservatorio, h]
            acoes.append(acao)
            custo_energia += (self.PENALIDADE_PONTA if h in self.H_PONTA else self.PENALIDADE_NORMAL) * acao
            if acao_anterior == 0 and acao == 1:
                custo_energia += self.PENALIDADE_TROCA
            nivel_reservatorio = self.transicao(nivel_reservatorio, acao, self.vazao_prev[h], self.entrada_agua)
            nivel_reservatorio_hist.append(nivel_reservatorio)
            acao_anterior = acao

        return nivel_reservatorio_hist, acoes, custo_energia, schedule_hours

def treinar_executar_modelo(treino_X, treino_y, teste_X):
    params_xgb = {'booster': 'gbtree', 'learning_rate': 0.1, 'max_depth': 1, 'n_estimators': 100}
    params_ada = {'estimator': DecisionTreeRegressor(max_depth=5), 'learning_rate': 0.01, 'n_estimators': 50}
    modeloXGB = xgb.XGBRegressor(**params_xgb)
    modeloXGB.fit(treino_X, treino_y)
    modeloADA = AdaBoostRegressor(**params_ada)
    modeloADA.fit(treino_X, treino_y)
    return modeloXGB.predict(teste_X), modeloADA.predict(teste_X)

def criar_result_df(teste_X, teste_y, pred_xgb, pred_ada):
    pred_df = pd.DataFrame({'XGB_PRED': pred_xgb, 'ADA_PRED': pred_ada}, index=teste_X.index)
    result_df = pd.concat([teste_X, teste_y, pred_df], axis=1)
    result_df = result_df[['vazao', 'XGB_PRED', 'ADA_PRED']]
    return result_df

def executar_pipeline(df=maindf, teste_inicio_str='2023-08-01 02:00:00'):
    df = df[['DATA/HORA', 'VAZAO SAIDA', 'VOLUME OCUPADO ATUAL']]
    df['DATA/HORA'] = pd.to_datetime(df['DATA/HORA'])
    df = df.rename(columns={'VAZAO SAIDA': 'vazao'})
    df.set_index('DATA/HORA', inplace=True)
    df = df[(df['vazao'] >= -100) & (df['vazao'] <= 100)]
    df = df.resample('H').mean()
    df = df[df['vazao'] != 0]
    df = df.dropna()
    df = create_features(df)
    treino_X, teste_X, treino_y, teste_y = separar_treino_teste(df, teste_inicio_str)
    treino_X.sort_index(inplace=True)
    treino_y.sort_index(inplace=True)
    teste_X.sort_index(inplace=True)
    teste_y.sort_index(inplace=True)
    pred_xgb, pred_ada = treinar_executar_modelo(treino_X, treino_y, teste_X)
    result_df = criar_result_df(teste_X, teste_y, pred_xgb, pred_ada)

    pred = pd.DataFrame({
        'vazao': teste_y,
        'pred_ada': pred_ada,
        'pred_xgb': pred_xgb
    })
    volume_ocupado_atual = df[['VOLUME OCUPADO ATUAL']]

    volume_ocupado_atual_reindexed = volume_ocupado_atual.reindex(pred.index)

    pred['VOLUME OCUPADO ATUAL'] = volume_ocupado_atual_reindexed['VOLUME OCUPADO ATUAL']
    return result_df, pred

def tempo_reservatorio_esvaziar(predicoes, dia_do_teste):
    volume = predicoes['VOLUME OCUPADO ATUAL'].iloc[0]
    niveis_caixa = [volume]
    dt = datetime.strptime(dia_do_teste, '%Y-%m-%d %H:%M:%S')
    horas_completas = [dt]
    horas = 0
    minutos = 0

    for indice in range(0, len(predicoes)):
        saida_ada = predicoes['pred_ada'].iloc[indice] * 3600
        if (volume - saida_ada) > 0:
            volume = volume - saida_ada
            niveis_caixa.append(volume)
            horas_completas.append(dt + pd.Timedelta(hours=indice + 1))
        else:
            minutos = int(volume * 60 / saida_ada)
            horas = indice
            hora_final = dt + pd.Timedelta(hours=indice) + pd.Timedelta(minutes=minutos)
            niveis_caixa.append(0)
            horas_completas.append(hora_final)
            break

    return horas, minutos, niveis_caixa, horas_completas


def calaculaHorariDePonta(df=maindf):
    # Adicionar a nova coluna 'GMB Combined'
    df['GMB Combined'] = df['GMB 1 (10 OFF/ 90 ON)'] | df['GMB 2(10 OFF/ 90 ON)']
    
    # Converter a coluna 'DATA/HORA' para datetime
    df['DATA/HORA'] = pd.to_datetime(df['DATA/HORA'], errors='coerce')
    
    # Extrair o dia da semana e a hora
    df['DIA_SEMANA'] = df['DATA/HORA'].dt.dayofweek
    df['HORA'] = df['DATA/HORA'].dt.hour
    
    # Criar a coluna 'HORARIO PONTA'
    df['HORARIO PONTA'] = 0
    df.loc[(df['HORA'] >= 18) & (df['HORA'] <= 21), 'HORARIO PONTA'] = 1
    
    # Somar os valores de 'DIFERENÇA MEDIÇÃO SEGUNDOS' para 'HORARIO PONTA' 0 e 1
    soma_ponta_0 = df[df['HORARIO PONTA'] == 0]['DIFERENÇA MEDIÇÃO SEGUNDOS'].sum()
    soma_ponta_1 = df[df['HORARIO PONTA'] == 1]['DIFERENÇA MEDIÇÃO SEGUNDOS'].sum()
    
    # Converter o resultado para horas
    soma_ponta_0_horas = soma_ponta_0 / 3600
    soma_ponta_1_horas = soma_ponta_1 / 3600
    
    return soma_ponta_0_horas, soma_ponta_1_horas
