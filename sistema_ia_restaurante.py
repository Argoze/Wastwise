# Importação das bibliotecas necessárias
import pandas as pd  # Biblioteca para manipulação de dados em tabelas (DataFrames)
import numpy as np  # Biblioteca para operações matemáticas e manipulação de arrays
from sklearn.model_selection import train_test_split  # Para dividir os dados em treino e teste
from sklearn.ensemble import RandomForestRegressor  # Modelo de aprendizado de máquina para regressão
from sklearn.metrics import mean_squared_error, r2_score  # Métricas de avaliação do modelo
import matplotlib.pyplot as plt  # Biblioteca para geração de gráficos
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Exibição de gráficos no Tkinter
import tkinter as tk  # Biblioteca para criação de interfaces gráficas
from tkinter import filedialog, messagebox, ttk  # Widgets adicionais para a interface gráfica

# Lista global para armazenar os pedidos temporariamente
pedidos_temporarios = []

# Configuração de cores para a interface gráfica
COR_FUNDO = "#6f737e"  # Cor do fundo da interface 
COR_BOTAO = "#3e4b51"  # Cor dos botões 

# Função para configurar estilos visuais da interface gráfica
def configurar_estilo():
    style = ttk.Style()  # Cria um objeto de estilo
    style.theme_use('clam')  # Define o tema como "clam"
    # Configuração para os botões
    style.configure("TButton", font=("Arial", 12), background=COR_BOTAO, foreground="white", padding=6)
    style.map("TButton", background=[("active", "#070705")])  # Muda a cor do botão quando o mouse passa por cima
    # Configuração para os rótulos
    style.configure("TLabel", font=("Arial", 12), background=COR_FUNDO, foreground="#333")
    # Configuração para os frames (caixas de conteúdo)
    style.configure("TFrame", background=COR_FUNDO)
    # Configuração para as entradas de texto
    style.configure("TEntry", font=("Arial", 12), padding=5)

# Função para carregar um arquivo CSV com dados
def carregar_arquivo():
    caminho = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])  # Abre uma janela para selecionar o arquivo
    if caminho:  # Verifica se o usuário selecionou um arquivo
        try:
            global dados  # Declara a variável 'dados' como global
            dados = carregar_dados(caminho)  # Chama a função para carregar os dados
            messagebox.showinfo("Sucesso", "Dados carregados com sucesso!")  # Exibe uma mensagem de sucesso
        except Exception as e:  # Captura erros durante o carregamento
            messagebox.showerror("Erro", f"Erro ao carregar os dados: {e}")  # Exibe uma mensagem de erro

# Função para processar o arquivo CSV e carregar os dados em um DataFrame
def carregar_dados(caminho):
    dados = pd.read_csv(caminho)  # Lê o arquivo CSV e converte para um DataFrame
    dados['Data'] = pd.to_datetime(dados['Data'])  # Converte a coluna 'Data' para o formato datetime
    return dados  # Retorna o DataFrame com os dados processados

# Função para pré-processar os dados antes do treinamento do modelo
def preprocessar_dados(dados):
    # Converte colunas categóricas ('Prato' e 'Sazonalidade') em variáveis binárias
    dados = pd.get_dummies(dados, columns=['Prato', 'Sazonalidade'], drop_first=True)
    return dados  # Retorna os dados pré-processados

# Função para treinar o modelo de aprendizado de máquina
def treinar_modelo(X, y):
    # Divide os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Define o modelo de regressão
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)  # Treina o modelo com os dados de treino
    y_pred = modelo.predict(X_test)  # Faz previsões com os dados de teste
    # Calcula métricas de avaliação
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Erro médio quadrático
    r2 = r2_score(y_test, y_pred)  # Coeficiente de determinação (R²)
    print(f"Raiz quadrada do erro-médio: {rmse:.2f}")  # Exibe o RMSE no console
    print(f"R²: {r2:.2f}")  # Exibe o R² no console
    return modelo, X_test, y_test, y_pred  # Retorna o modelo treinado e os dados de teste/predição

# Função para executar o treinamento do modelo e realizar previsões
def executar_previsao():
    global modelo, X_test, y_test, y_pred  # Declara variáveis como globais
    if 'dados' not in globals():  # Verifica se os dados foram carregados
        messagebox.showerror("Erro", "Carregue os dados primeiro!")  # Exibe mensagem de erro
        return
    
    try:
        dados_processados = preprocessar_dados(dados)  # Pré-processa os dados
        global X
        X = dados_processados.drop(columns=['Data', 'Demanda'])  # Define as features
        y = dados_processados['Demanda']  # Define o alvo
        modelo, X_test, y_test, y_pred = treinar_modelo(X, y)  # Treina o modelo
        # Calcula métricas de avaliação
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        # Exibe as métricas de avaliação em um popup
        messagebox.showinfo(
            "Previsão Realizada",
            f"Modelo treinado com sucesso!\n\n"
            f"Métricas de Avaliação:\n"
            f"- RMSE: {rmse:.2f}\n"
            f"- R²: {r2:.2f}"
        )
    except Exception as e:  # Captura erros durante o processo
        messagebox.showerror("Erro", f"Erro ao executar a previsão: {e}")

# Função para exibir o gráfico em um popup
def exibir_grafico():
    if 'y_test' not in globals() or 'y_pred' not in globals():  # Verifica se há previsões
        messagebox.showerror("Erro", "Execute a previsão primeiro!")  # Exibe mensagem de erro
        return

    # Cria uma nova janela para o gráfico
    popup_grafico = tk.Toplevel(janela)  # Cria uma nova janela
    popup_grafico.title("Gráfico: Valores Reais vs Previsões")  # Define o título do popup
    popup_grafico.geometry("800x600")  # Define o tamanho do popup

    # Cria um gráfico de dispersão
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.6)  # Plota os pontos reais vs previstos
    ax.plot([0, max(y_test)], [0, max(y_test)], color='red', linestyle='--')  # Linha de referência
    ax.set_xlabel("Valores Reais")
    ax.set_ylabel("Previsões")
    ax.set_title("Valores Reais vs Previsões")

    # Insere o gráfico na nova janela
    canvas = FigureCanvasTkAgg(fig, master=popup_grafico)
    canvas.get_tk_widget().pack(fill="both", expand=True)  # Ajusta o gráfico à janela
    canvas.draw()

    # Adiciona um botão para fechar o popup
    ttk.Button(popup_grafico, text="Fechar", command=popup_grafico.destroy).pack(pady=10)


# Função para prever a próxima semana com um popup de largura maior
def prever_proxima_semana():
    if 'modelo' not in globals():  # Verifica se o modelo foi treinado
        messagebox.showerror("Erro", "Treine o modelo primeiro!")  # Exibe mensagem de erro
        return

    # Identifica o último dia registrado nos dados
    ultimo_dia = dados["Data"].max()
    pratos_unicos = dados["Prato"].unique()
    sazonalidade_mais_comum = dados["Sazonalidade"].mode()[0]

    # Gera os próximos 7 dias para previsão
    proximos_dias = [ultimo_dia + pd.Timedelta(days=i) for i in range(1, 8)]
    dados_proxima_semana = pd.DataFrame({
        "Data": [dia for dia in proximos_dias for _ in pratos_unicos],
        "Prato": pratos_unicos.tolist() * len(proximos_dias),
        "Sazonalidade": [sazonalidade_mais_comum] * len(proximos_dias) * len(pratos_unicos)
    })

    pratos_originais = dados_proxima_semana[["Data", "Prato"]].copy()
    dados_proxima_semana = pd.get_dummies(dados_proxima_semana, columns=["Prato", "Sazonalidade"], drop_first=True)

    for col in X.columns:
        if col not in dados_proxima_semana.columns:
            dados_proxima_semana[col] = 0
    dados_proxima_semana = dados_proxima_semana[X.columns]

    previsoes = modelo.predict(dados_proxima_semana)
    dados_proxima_semana["Demanda Prevista"] = np.round(previsoes).astype(int)
    dados_proxima_semana = pd.concat([pratos_originais, dados_proxima_semana["Demanda Prevista"]], axis=1)

    # Formata o resultado para exibição
    resultado_formatado = f"{'Data':<12}{'Prato':<15}{'Demanda Prevista':<20}\n"
    resultado_formatado += "-" * 47 + "\n"
    for _, row in dados_proxima_semana.iterrows():
        resultado_formatado += f"{row['Data'].strftime('%Y-%m-%d'):<12}{row['Prato']:<15}{row['Demanda Prevista']:<20}\n"

    # Cria uma nova janela para exibir a previsão
    popup_previsao = tk.Toplevel(janela)  # Cria uma janela secundária
    popup_previsao.title("Previsão da Próxima Semana")  # Define o título da janela
    popup_previsao.geometry("700x400")  # Ajusta a largura e altura da janela

    # Cria um frame para exibir os dados formatados
    frame_previsao = ttk.Frame(popup_previsao, padding=20)
    frame_previsao.pack(fill="both", expand=True)

    # Cria um widget de texto para exibir os resultados
    texto_previsao = tk.Text(frame_previsao, wrap="none", font=("Courier", 12), height=20, width=70)
    texto_previsao.insert("1.0", resultado_formatado)  # Insere os dados formatados no widget de texto
    texto_previsao.config(state="disabled")  # Impede que o texto seja editado
    texto_previsao.pack(side="left", fill="both", expand=True)

    # Adiciona uma barra de rolagem horizontal
    scroll_x = tk.Scrollbar(frame_previsao, orient="horizontal", command=texto_previsao.xview)
    texto_previsao.configure(xscrollcommand=scroll_x.set)
    scroll_x.pack(side="bottom", fill="x")

    # Adiciona uma barra de rolagem vertical
    scroll_y = tk.Scrollbar(frame_previsao, orient="vertical", command=texto_previsao.yview)
    texto_previsao.configure(yscrollcommand=scroll_y.set)
    scroll_y.pack(side="right", fill="y")

# Função para adicionar um novo pedido à lista de pedidos temporários
def adicionar_novo_pedido():
    # Cria uma nova janela para entrada de dados
    janela_adicionar = tk.Toplevel(janela)
    janela_adicionar.title("Adicionar Novo Pedido")  # Define o título da janela
    janela_adicionar.geometry("400x400")  # Define o tamanho da janela
    janela_adicionar.configure(background=COR_FUNDO)  # Define a cor de fundo da janela

    # Label e campo de entrada para a data do pedido
    ttk.Label(janela_adicionar, text="Data do Pedido (YYYY-MM-DD):").pack(pady=5)
    entrada_data = ttk.Entry(janela_adicionar)
    entrada_data.pack(pady=5)

    # Botões de seleção de pratos
    ttk.Label(janela_adicionar, text="Escolha o Prato:").pack(pady=5)
    prato_selecionado = tk.StringVar(value="")  # Variável para armazenar o prato selecionado

    # Função para selecionar o prato e mudar a cor
    def selecionar_prato(prato, botao):
        prato_selecionado.set(prato)  # Define o prato selecionado
        for b in botoes_pratos:  # Restaura a cor original dos outros botões
            b.configure(bg=COR_BOTAO, fg="white")
        botao.configure(bg="#4CAF50", fg="white")  # Muda a cor do botão selecionado

    # Lista de pratos e botões
    pratos = ["Pizza", "Sushi", "Hambúrguer"]
    botoes_pratos = []
    for prato in pratos:
        botao = tk.Button(
            janela_adicionar,
            text=prato,
            bg=COR_BOTAO,
            fg="white",
            font=("Arial", 12),
            command=lambda p=prato, b=None: selecionar_prato(p, b),
        )
        botoes_pratos.append(botao)
        botao.pack(pady=5)

    # Label e campo de entrada para a demanda do prato
    ttk.Label(janela_adicionar, text="Demanda do Prato:").pack(pady=5)
    entrada_demanda = ttk.Entry(janela_adicionar)
    entrada_demanda.pack(pady=5)

    # Label para a sazonalidade
    ttk.Label(janela_adicionar, text="Escolha a Sazonalidade:").pack(pady=5)
    sazonalidade_var = tk.StringVar(value="Normal")  # Valor padrão

    # Botões para selecionar a sazonalidade
    def selecionar_sazonalidade(sazonalidade):
        sazonalidade_var.set(sazonalidade)  # Define a sazonalidade selecionada

    ttk.Button(janela_adicionar, text="Alta", command=lambda: selecionar_sazonalidade("Alta"))\
        .pack(pady=5, side="top")
    ttk.Button(janela_adicionar, text="Normal", command=lambda: selecionar_sazonalidade("Normal"))\
        .pack(pady=5, side="top")
    ttk.Button(janela_adicionar, text="Baixa", command=lambda: selecionar_sazonalidade("Baixa"))\
        .pack(pady=5, side="top")

    # Função para salvar o pedido na lista temporária
    def salvar_temp_pedido():
        data = entrada_data.get()  # Obtém a data inserida
        prato = prato_selecionado.get()  # Obtém o prato selecionado
        demanda = entrada_demanda.get()  # Obtém a demanda inserida
        sazonalidade = sazonalidade_var.get()   # Obtém a sazonalidade inserida

        # Verifica se todos os campos foram preenchidos
        if not data or not prato or not demanda or not sazonalidade:
            messagebox.showerror("Erro", "Por favor, preencha todos os campos!")
            return

        # Tenta converter a demanda para um número inteiro
        try:
            demanda = int(demanda)
        except ValueError:
            messagebox.showerror("Erro", "Demanda deve ser um número inteiro!")
            return

        # Adiciona o pedido à lista temporária
        pedidos_temporarios.append({"Data": data, "Prato": prato, "Demanda": demanda, "Sazonalidade": sazonalidade})
        messagebox.showinfo("Sucesso", f"Pedido adicionado: {prato}, {data}, {demanda}, {sazonalidade}")
        janela_adicionar.destroy()  # Fecha a janela

    # Botão para salvar o pedido
    ttk.Button(janela_adicionar, text="Adicionar Pedido", command=salvar_temp_pedido).pack(pady=20)


# Função para salvar os pedidos temporários em um arquivo CSV
def salvar_pedidos():
    if not pedidos_temporarios:  # Verifica se há pedidos na lista
        messagebox.showerror("Erro", "Nenhum pedido foi adicionado!")
        return

    # Cria um DataFrame com os pedidos e salva no arquivo CSV
    df_pedidos = pd.DataFrame(pedidos_temporarios)
    df_pedidos.to_csv("pedidos_adicionados.csv", index=False)
    messagebox.showinfo("Sucesso", "Todos os pedidos foram salvos no arquivo 'pedidos_adicionados.csv'!")

# Criação da interface principal
janela = tk.Tk()  # Inicializa a janela principal
janela.title("Sistema de IA para Restaurantes")  # Define o título da janela principal
janela.geometry("600x500")  # Define o tamanho da janela principal
janela.configure(background=COR_FUNDO)  # Define a cor de fundo da janela principal

# Configura o estilo visual
configurar_estilo()

# Criação de um frame com rolagem
frame_canvas = ttk.Frame(janela)
frame_canvas.pack(fill="both", expand=True, padx=10, pady=10)

canvas = tk.Canvas(frame_canvas, background=COR_FUNDO)
scrollbar = ttk.Scrollbar(frame_canvas, orient="vertical", command=canvas.yview)
scrollable_frame = ttk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="center")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# Criação de um sub-frame para os botões alinhados horizontalmente
frame_botoes = ttk.Frame(scrollable_frame)
frame_botoes.pack(pady=20)

# Botões principais alinhados horizontalmente
ttk.Button(frame_botoes, text="Adicionar Novo Pedido", command=adicionar_novo_pedido).pack(side="left", padx=50, pady=10)
ttk.Button(frame_botoes, text="Salvar Pedidos", command=salvar_pedidos).pack(side="left", padx=50, pady=10)
ttk.Button(frame_botoes, text="Carregar Dados", command=carregar_arquivo).pack(side="left", padx=50, pady=10)
ttk.Button(frame_botoes, text="Executar Previsão", command=executar_previsao).pack(side="left", padx=50, pady=10)
ttk.Button(frame_botoes, text="Prever Próxima Semana", command=prever_proxima_semana).pack(side="left", padx=50, pady=10)
ttk.Button(frame_botoes, text="Exibir Gráfico", command=exibir_grafico).pack(side="left", padx=50, pady=10)


# Inicia o loop principal da interface gráfica
janela.mainloop()
