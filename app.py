from flask import Flask, render_template, request, jsonify
from graphs import *

# Cria uma instância do aplicativo Flask
app = Flask(__name__)

# Rota para a página inicial
@app.route('/')
def index():
    return render_template('index.html')   

@app.route('/vazaosaida', methods=['GET', 'POST'])
def vazao():
    if request.method == 'GET':
        dia = '2023-08-01 02:00:00'
        g1 = g_vazaoSaida()
        g2 = g_curvaTipica()
        result_df, pred = executar_pipeline(teste_inicio_str=dia)
        g3 = g_predSaida(result_df)
        horas, minutos, niveis_caixa, horas_completas = tempo_reservatorio_esvaziar(pred, dia)
        g4 = plotar_nivel_agua(horas,minutos,niveis_caixa, horas_completas)
        return render_template('vazaosaida.html', plot1=g1.to_html(), plot2=g2.to_html(), plot3=g3.to_html(), plot4=g4.to_html())
    


@app.route('/bombas')
def bombas():
    
    
    return render_template('bombas.html')


@app.route('/dados')
def dados():
    g1 = plot_vazao_media_por_temperatura(get_vazao_media_por_temperatura())
    g2 = plot_correlacao_temperatura_vazao(get_vazao_media_por_temperatura())
    return render_template('dados.html',plot1=g1.to_html(),plot2=g2.to_html())

if __name__ == '__main__':
    app.run(debug=True) 