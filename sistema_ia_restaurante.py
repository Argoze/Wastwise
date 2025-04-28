import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, messagebox

# Função para carregar arquivo CSV
def carregar_arquivo():
    caminho = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if caminho:
        try:
            global dados
            dados = carregar_dados(caminho)
            messagebox.showinfo("Sucesso", "Dados carregados com sucesso!")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar os dados: {e}")

# Função para carregar dados
def carregar_dados(caminho):
    dados = pd.read_csv(caminho)
    dados['Data'] = pd.to_datetime(dados['Data'])  # Converter datas
    return dados

# Função de pré-processamento
def preprocessar_dados(dados):
    # Codificar variáveis categóricas
    dados = pd.get_dummies(dados, columns=['Prato', 'Sazonalidade'], drop_first=True)
    return dados

# Função para treinar o modelo
def treinar_modelo(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    
    # Prever e calcular métricas
    y_pred = modelo.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")
    return modelo, X_test, y_test, y_pred

# Função para treinar o modelo e prever
def executar_previsao():
    global modelo, X_test, y_test, y_pred
    if 'dados' not in globals():
        messagebox.showerror("Erro", "Carregue os dados primeiro!")
        return
    
    try:
        dados_processados = preprocessar_dados(dados)
        global X
        X = dados_processados.drop(columns=['Data', 'Demanda'])
        y = dados_processados['Demanda']
        modelo, X_test, y_test, y_pred = treinar_modelo(X, y)
        messagebox.showinfo("Sucesso", "Modelo treinado e previsões realizadas!")
    except Exception as e:
        messagebox.showerror("Erro", f"Erro ao executar a previsão: {e}")

# Função para exibir o gráfico de previsões
def exibir_grafico():
    if 'y_test' not in globals() or 'y_pred' not in globals():
        messagebox.showerror("Erro", "Execute a previsão primeiro!")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.6)
    ax.plot([0, max(y_test)], [0, max(y_test)], color='red', linestyle='--')
    ax.set_xlabel("Valores Reais")
    ax.set_ylabel("Previsões")
    ax.set_title("Valores Reais vs Previsões")

    canvas = FigureCanvasTkAgg(fig, master=janela)
    canvas.get_tk_widget().pack()
    canvas.draw()


# Função para prever o próximo dia
def prever_proximo_dia():
    if 'modelo' not in globals():
        messagebox.showerror("Erro", "Treine o modelo primeiro!")
        return

    # Dados fictícios para o próximo dia
    dados_proximo_dia = pd.DataFrame({
        "Data": ["2024-12-09", "2024-12-09", "2024-12-09"],
        "Prato": ["Pizza", "Sushi", "Hambúrguer"],
        "Sazonalidade": ["Normal", "Alta", "Baixa"]
    })

    # Guardar a coluna original de 'Prato' para exibição posterior
    pratos_originais = dados_proximo_dia["Prato"].copy()

    # Pré-processar os dados
    dados_proximo_dia = pd.get_dummies(dados_proximo_dia, columns=["Prato", "Sazonalidade"], drop_first=True)

    # Garantir que os dados tenham as mesmas colunas que o modelo espera
    for col in X.columns:
        if col not in dados_proximo_dia.columns:
            dados_proximo_dia[col] = 0

    # Remover colunas extras que não estavam no treinamento
    dados_proximo_dia = dados_proximo_dia[X.columns]

    # Fazer a previsão
    previsoes = modelo.predict(dados_proximo_dia)
    dados_proximo_dia["Demanda Prevista"] = previsoes

    # Adicionar os pratos originais de volta para exibição
    dados_proximo_dia["Prato"] = pratos_originais

    # Exibir os resultados em um popup
    resultado = dados_proximo_dia[["Prato", "Demanda Prevista"]].to_string(index=False)
    messagebox.showinfo("Previsão do Próximo Dia", f"Demanda Prevista:\n{resultado}")


# Criação da Interface
janela = tk.Tk()
janela.title("Sistema de IA para Restaurantes")
janela.geometry("600x500")

# Botões
btn_carregar = tk.Button(janela, text="Carregar Dados", command=carregar_arquivo, width=20)
btn_carregar.pack(pady=10)

btn_prever = tk.Button(janela, text="Executar Previsão", command=executar_previsao, width=20)
btn_prever.pack(pady=10)

btn_grafico = tk.Button(janela, text="Exibir Gráfico", command=exibir_grafico, width=20)
btn_grafico.pack(pady=10)

btn_prever_proximo = tk.Button(janela, text="Prever Próximo Dia", command=prever_proximo_dia, width=20)
btn_prever_proximo.pack(pady=10)

# Rodar o programa
janela.mainloop()
