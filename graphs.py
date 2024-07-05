import plotly.graph_objs as go
import pandas as pd
import numpy as np
from funcs import *
from qmodel import *
from plotly.subplots import make_subplots

handler = DataFrameHandler()

def g_vazaoSaida():
    fig = go.Figure()
    momento = handler.get_coluna('DATA/HORA')
    vazao = handler.get_coluna('VAZAO SAIDA')
    
    # Substituir valores negativos pela média
    vazao_sem_negativos = np.where(vazao < 0, np.mean(vazao[vazao >= 0]), vazao)
    
    fig.add_trace(go.Scatter(x=momento, y=vazao_sem_negativos, mode='lines', name='Vazão de Saída'))
    fig.update_layout(
        title='Qual é a vazão de saída do reservatório a cada momento?',
        xaxis_title='Tempo (horas)', 
        yaxis_title='Vazão (litros por hora)',
        annotations=[
            dict(
                x=0.5,
                y=-0.3,
                xref='paper',
                yref='paper',
                text='Vazão de Saída da Caixa d\'Água ao Longo do Tempo',
                showarrow=False,
                font=dict(size=16),
                xanchor='center'
            )
        ]
    )
    
    return fig

def g_curvaTipica():
    df = handler.remove_outliers('VAZAO SAIDA')
    
    df['DATA/HORA'] = pd.to_datetime(df['DATA/HORA'])
    df['dia_da_semana'] = df['DATA/HORA'].dt.dayofweek 
    df['hora'] = df['DATA/HORA'].dt.hour
    df_uteis = df[df['dia_da_semana'] < 5]
    df_fim_de_semana = df[df['dia_da_semana'] >= 5]
    media_horaria_uteis = df_uteis.groupby('hora')['VAZAO SAIDA'].mean()
    media_horaria_fim_de_semana = df_fim_de_semana.groupby('hora')['VAZAO SAIDA'].mean()

    fig = go.Figure()

    # Gráfico de probabilidade
    fig.add_trace(go.Scatter(x=media_horaria_uteis.index, y=media_horaria_uteis.values,
                             mode='lines', name='Dias úteis'))
    fig.add_trace(go.Scatter(x=media_horaria_fim_de_semana.index, y=media_horaria_fim_de_semana.values,
                             mode='lines', name='Finais de semana'))

    # Layout do gráfico
    fig.update_layout(
        title='Qual é a curva típica da saída do reservatório ao longo de 24 horas durante os dias úteis e nos finais de semana?',
        xaxis_title='Hora do Dia',
        yaxis_title='Vazão Média',
        legend_title='Tipo de Dia',
        annotations=[
            dict(
                x=0.5,
                y=-0.3,
                xref='paper',
                yref='paper',
                text='Curva Típica de Vazão de Saída',
                showarrow=False,
                font=dict(size=16),
                xanchor='center'
            )
        ]
    )
    return fig

def g_previsaoBombas(vazao_prev, entrada_agua, current_level, start_hour):
    model = ReservoirModel(vazao_prev, entrada_agua, current_level, start_hour)
    nivel_reservatorio_hist, acoes, custo_energia, schedule_hours = model.simulate()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(model.N_HOURS + 1)),
        y=nivel_reservatorio_hist,
        mode='lines+markers',
        name='Nível do Reservatório'
    ))

    fig.add_trace(go.Scatter(
        x=list(range(model.N_HOURS + 1)),
        y=[20] * (model.N_HOURS + 1),
        mode='lines',
        name='Nível Mínimo Base (20%)',
        line=dict(color='red', dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=list(range(model.N_HOURS + 1)),
        y=[70] * (model.N_HOURS + 1),
        mode='lines',
        name='Nível Máximo (70%)',
        line=dict(color='green', dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=list(range(model.N_HOURS)),
        y=[model.dynamic_min_level(schedule_hours[t], model.vazao_prev) for t in range(model.N_HOURS)],
        mode='lines',
        name='Nível Mínimo Dinâmico',
        line=dict(color='red', dash='dash', width=1),
        opacity=0.5
    ))

    fig.update_layout(
        title='Nível do Reservatório ao Longo do Dia',
        xaxis_title='Hora',
        yaxis_title='Nível do Reservatório (%)',
        legend=dict(x=0.01, y=0.99),
        xaxis=dict(tickmode='array', tickvals=list(range(model.N_HOURS)), ticktext=schedule_hours),
        yaxis=dict(showgrid=True),
        showlegend=True,
        annotations=[
            dict(
                x=0.5,
                y=-0.2,
                xref='paper',
                yref='paper',
                text='Nível do Reservatório ao Longo do Dia',
                showarrow=False,
                font=dict(size=16),
                xanchor='center'
            )
        ]
    )
    return fig

def g_predSaida(result_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result_df.index,
        y=result_df['vazao'],
        mode='lines',
        name='VAZÃO SAIDA'
    ))
    fig.add_trace(go.Scatter(
        x=result_df.index,
        y=result_df['XGB_PRED'],
        mode='lines',
        name='XGB_PRED'
    ))
    fig.add_trace(go.Scatter(
        x=result_df.index,
        y=result_df['ADA_PRED'],
        mode='lines',
        name='ADA_PRED'
    ))

    fig.update_layout(
        title='Qual é a previsão da vazão de saída para as próximas 24 horas para um determinado dia e horário?',
        xaxis_title='Data',
        yaxis_title='Valor',
        legend_title='Séries',
        xaxis=dict(
            tickformat='%d-%m-%Y %H:%M'
        ),
        hovermode='x unified',
        hoverlabel=dict(
            namelength=-1
        ),
        annotations=[
            dict(
                x=0.5,
                y=-0.3,
                xref='paper',
                yref='paper',
                text='Comparação de VAZÃO SAIDA e Predições',
                showarrow=False,
                font=dict(size=16),
                xanchor='center'
            )
        ]
    )
    fig.update_traces(
        hovertemplate='%{x|%d-%m-%Y %H:%M}<br>%{y}'
    )
    return fig

def plotar_nivel_agua(horas, minutos, niveis_caixa, horas_completas):
    # Cria o gráfico de área
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.7, 0.3],  # Ajusta a largura das colunas
        specs=[[{"type": "scatter"}, {"type": "table"}]]
    )

    fig.add_trace(go.Scatter(
        x=horas_completas,
        y=niveis_caixa,
        mode='lines+markers',
        fill='tozeroy',
        name='Nível de Água'
    ), row=1, col=1)

    # Adiciona a tabela com as horas e minutos
    fig.add_trace(
        go.Table(
            header=dict(values=["Horas", "Minutos"],
                        font=dict(size=26, color="black"),
                        align="left"),
            cells=dict(values=[horas, minutos],
                       align="left")
        ),
        row=1, col=2
    )

    fig.update_layout(
        title='Nível de Água no Reservatório',
        xaxis_title='Hora',
        yaxis_title='Nível de Água (litros)',
        xaxis=dict(
            tickformat='%H:%M\n%d-%m-%Y'
        ),
        hovermode='x unified',
        hoverlabel=dict(
            namelength=-1
        )
    )

    fig.update_traces(
        hovertemplate='%{x|%H:%M %d-%m-%Y}<br>%{y} litros',
        row=1, col=1
    )

    return fig

import plotly.express as px

def plot_bomb_status(soma_ponta_0_horas, soma_ponta_1_horas):
    labels = ['Fora de Horário de Ponta', 'Dentro de Horário de Ponta']
    values = [soma_ponta_0_horas, soma_ponta_1_horas]
    
    fig = go.Figure(data=[go.Bar(x=labels, y=values)])
    fig.update_layout(title='Horas em Horário de Ponta vs Fora de Horário de Ponta',
                      xaxis_title='Estado das Bombas',
                      yaxis_title='Horas')
    
    return fig

def plot_pie_chart(soma_ponta_0_horas, soma_ponta_1_horas):
    labels = ['Horário de Ponta', 'Fora de Ponta']
    values = [soma_ponta_1_horas, soma_ponta_0_horas]
    
    fig = px.pie(names=labels, values=values, title='Tempo em Horário de Ponta vs Fora de Ponta')
    fig.update_traces(textinfo='percent+label')
    
    return fig

def plot_vazao_media_por_temperatura(df_vazao_media):
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_vazao_media['TEMPERATURA'],
        y=df_vazao_media['VAZAO SAIDA'],
        mode='lines+markers',
        name='Média de Vazão'
    ))
    
    fig.update_layout(
        title='Gráfico de Média de Vazão em Função da Temperatura',
        xaxis_title='Temperatura (°C)',
        yaxis_title='Média de Vazão',
        xaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=5
        ),
        yaxis=dict(
            tickformat=',.0f'
        ),
        hovermode='x unified'
    )
    
    return fig

def plot_correlacao_temperatura_vazao(df):
    # Calcular correlação entre 'TEMPERATURA' e 'VAZAO SAIDA'
    correlacao = df['TEMPERATURA'].corr(df['VAZAO SAIDA'])
    
    # Criar gráfico de dispersão com Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['TEMPERATURA'], y=df['VAZAO SAIDA'], mode='markers',
                             marker=dict(color='blue', opacity=0.7),
                             name=f'Correlação: {correlacao:.2f}'))
    
    # Personalizar layout do gráfico
    fig.update_layout(title='Correlação entre Temperatura e Vazão de Saída',
                      xaxis_title='Temperatura (°C)',
                      yaxis_title='Vazão de Saída',
                      showlegend=True,
                      legend=dict(x=0.02, y=0.98),
                      margin=dict(l=50, r=50, t=50, b=50),
                      hovermode='closest',
                      plot_bgcolor='rgba(0,0,0,0)')
    
    return fig