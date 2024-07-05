from flask import Flask, render_template, request, jsonify
from graphs import *



dia = '2023-08-01 02:00:00'
result_df, pred = executar_pipeline(teste_inicio_str=dia)
vazao_prev = [12, 11, 10, 11, 11, 12, 11, 14, 15, 18, 17, 18, 19, 19, 18, 18, 18, 18, 16, 16, 17, 15, 14, 14, 12, 10]
vazao_prev = np.array(vazao_prev)
entrada_agua = 23
current_level = 57
start_hour = 0 
g_predBombas = g_previsaoBombas(vazao_prev, entrada_agua, current_level, start_hour)

# Cria uma instância do aplicativo Flask
app = Flask(__name__)
# Rota para a página inicial
@app.route('/')
def index():
    return render_template('index.html')   

@app.route('/vazaosaida', methods=['GET', 'POST'])
def vazao():
    if request.method == 'GET':
        
        g1 = g_vazaoSaida()
        g2 = g_curvaTipica()
        

        g3 = g_predSaida(result_df)
        horas, minutos, niveis_caixa, horas_completas = tempo_reservatorio_esvaziar(pred, dia)
        g4 = plotar_nivel_agua(horas,minutos,niveis_caixa, horas_completas)
        return render_template('vazaosaida.html', plot1=g1.to_html(), plot2=g2.to_html(), plot3=g3.to_html(), plot4=g4.to_html())
    


@app.route('/bombas')
def bombas():
    soma_ponta_0_horas, soma_ponta_1_horas = calaculaHorariDePonta(maindf)
    g1, totais_text = plot_pie_chart_com_totais(soma_ponta_0_horas, soma_ponta_1_horas)
    return render_template('bombas.html', plot1=g_predBombas.to_html(), plot2=g1.to_html(), totais_text=totais_text)


@app.route('/dados')
def dados():
    g1 = plot_vazao_media_por_temperatura_com_correlacao(get_vazao_media_por_temperatura())
    return render_template('dados.html',plot1=g1.to_html())

if __name__ == '__main__':
    app.run(debug=True) 
    