import pandas as pd
import numpy as np

maindf = pd.read_csv('dados\dados_com_temperatura.csv')
class DataFrameHandler:
    def __init__(self,df =None):
        if df is None:
            self.df = pd.read_csv('dados\dados_com_temperatura.csv')
        else:
            self.df = df.copy()

    def get_coluna(self, nome_coluna):
        return self.df[nome_coluna]
    
    def get_coluna_filtrada(self, nome_coluna):
        return self.remove_outliers(nome_coluna)[nome_coluna]
    
    def remove_outliers(self, column_name, threshold=3):
        self.df['Z_score'] = np.abs((self.df[column_name] - self.df[column_name].mean()) / self.df[column_name].std())
        df_filtered = self.df[self.df['Z_score'] <= threshold].copy()
        df_filtered = df_filtered.drop(columns=['Z_score'])
        return df_filtered
    
    def get_dataframe(self):
        return self.df

def preencher_media_anterior(df):
    df['data'] = df.index
    df = df.sort_index()
    media_anterior = df.groupby(['dayofweek', 'hour'])['vazao'].shift().groupby([df['dayofweek'], df['hour']]).mean().reset_index()
    media_anterior.rename(columns={'vazao': 'media_anterior'}, inplace=True)
    df = df.merge(media_anterior[['dayofweek', 'hour', 'media_anterior']], on=['dayofweek', 'hour'], how='left')
    df.set_index('data', inplace=True)
    return df

def create_features(df_):
    df = df_.copy()
    df['hour'] = df.index.hour.astype(int)
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    df = preencher_media_anterior(df)
    return df

def separar_treino_teste(df, teste_inicio_str):
    teste_inicio = pd.Timestamp(teste_inicio_str)
    if teste_inicio > df.index.max():
        teste_inicio = df.index.max()
    teste_fim = teste_inicio + pd.DateOffset(days=1) - pd.Timedelta(seconds=1)
    treino_fim = teste_inicio - pd.Timedelta(seconds=1)
    treino_inicio = treino_fim - pd.DateOffset(months=2) + pd.Timedelta(seconds=1)
    if treino_inicio < df.index.min():
        treino_inicio = df.index.min()
    df_treino = df.loc[(df.index >= treino_inicio) & (df.index <= treino_fim)]
    df_teste = df.loc[(df.index >= teste_inicio) & (df.index <= teste_fim)]
    if df_treino.empty or df_teste.empty:
        raise ValueError("O DataFrame de treino ou de teste está vazio. Verifique os intervalos de datas.")
    FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year', 'media_anterior']
    TARGET = 'vazao'
    return df_treino[FEATURES], df_teste[FEATURES], df_treino[TARGET], df_teste[TARGET]

def get_vazao_media_por_temperatura(df=maindf):
    df_filtrado = df.rename(columns={'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)': 'TEMPERATURA'})
    df_filtrado = df_filtrado[['VAZAO SAIDA', 'TEMPERATURA']]
    df_filtrado['TEMPERATURA'] = df_filtrado['TEMPERATURA'].str.replace(',', '.').astype(float)
    df_vazao_media = df_filtrado.groupby('TEMPERATURA')['VAZAO SAIDA'].mean().reset_index()
    return df_vazao_media
