import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import requests
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass

@dataclass
class DadosDengue:
    """Classe para armazenar dados históricos e previsões de casos de doenças"""
    dados_historicos: pd.DataFrame
    previsoes: pd.DataFrame
    metricas_modelo: Dict[str, float]

class PrevisaoDengue:
    """Classe para buscar e prever casos de doenças"""
    
    def __init__(self):
        self.url_base = "https://info.dengue.mat.br/api/alertcity"
        self.doencas_disponiveis = self._obter_doencas_disponiveis()
    
    def _obter_doencas_disponiveis(self) -> List[str]:
        """Obtém a lista de doenças disponíveis na API"""
        try:
            # Faz uma requisição para obter os metadados da API
            response = requests.get("https://info.dengue.mat.br/api/alertcity/config")
            if response.status_code == 200:
                # Extrai as doenças disponíveis da resposta
                # Nota: Ajuste o parsing conforme a estrutura real da resposta da API
                config = response.json()
                return config.get('diseases', ["dengue", "chikungunya", "zika"])  # fallback para lista padrão
            else:
                print("Aviso: Não foi possível obter a lista de doenças da API. Usando lista padrão.")
                return ["dengue", "chikungunya", "zika"]
        except Exception as e:
            #print(f"Erro ao obter doenças disponíveis: {e}")
            return ["dengue", "chikungunya", "zika"]
    
    def listar_doencas(self) -> List[str]:
        """Retorna a lista de doenças disponíveis para consulta"""
        return self.doencas_disponiveis
    
    def _validar_doenca(self, doenca: str) -> bool:
        """Valida se a doença especificada está disponível"""
        return doenca.lower() in [d.lower() for d in self.doencas_disponiveis]
    
    def _construir_url(self, geocodigo: int, doenca: str, se_inicio: int, 
                      se_fim: int, ano_inicio: int, ano_fim: int, 
                      formato_saida: str = "csv") -> str:
        """Constrói a URL da API para busca de dados"""
        parametros = (
            f"geocode={geocodigo}&disease={doenca}&format={formato_saida}&"
            f"ew_start={se_inicio}&ew_end={se_fim}&ey_start={ano_inicio}&ey_end={ano_fim}"
        )
        return f"{self.url_base}?{parametros}"
    
    def _buscar_dados(self, geocodigo: int, doenca: str, se_inicio: int,
                     se_fim: int, ano_inicio: int, ano_fim: int) -> Optional[pd.DataFrame]:
        """Busca dados de casos da API"""
        if not self._validar_doenca(doenca):
            print(f"Erro: Doença '{doenca}' não disponível. Doenças disponíveis: {', '.join(self.doencas_disponiveis)}")
            return None
            
        url = self._construir_url(geocodigo, doenca, se_inicio, se_fim, ano_inicio, ano_fim)
        try:
            resposta = requests.get(url)
            resposta.raise_for_status()
            from io import StringIO
            dados = StringIO(resposta.text)
            df = pd.read_csv(dados)
            if df.empty:
                print(f"Aviso: Nenhum dado encontrado para {doenca} no período especificado")
            return df
        except Exception as e:
            print(f"Erro ao buscar dados: {e}")
            return None
    
    def _pre_processar_dados(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pré-processa os dados brutos"""
        df['data_iniSE'] = pd.to_datetime(df['data_iniSE'], format='%Y-%m-%d')
        df['ano'] = df['data_iniSE'].dt.year
        df['mes'] = df['data_iniSE'].dt.month
        df['casos_totais'] = df[['casos', 'casos_est']].mean(axis=1)
        df_agg = df.groupby(['ano', 'mes']).agg({'casos_totais': 'sum'}).reset_index()
        df_agg['mes_ordinal'] = (df_agg['ano'] - df_agg['ano'].min()) * 12 + df_agg['mes']
        return df_agg
    
    def _obter_estacao(self, mes: int) -> str:
        """Determina a estação do ano com base no mês"""
        if mes in [12, 1, 2]:
            return 'Verão'
        elif mes in [3, 4, 5]:
            return 'Outono'
        elif mes in [6, 7, 8]:
            return 'Inverno'
        else:
            return 'Primavera'
    
    def _adicionar_estacoes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona variáveis dummy para as estações do ano"""
        df['estacao'] = df['mes'].apply(self._obter_estacao)
        return pd.get_dummies(df, columns=['estacao'], drop_first=True)
    
    def _treinar_modelo(self, df: pd.DataFrame) -> Tuple[RandomForestRegressor, Dict[str, float], np.ndarray]:
        """Treina o modelo e retorna as previsões"""
        X = df.drop(columns=['casos_totais'])
        y = df['casos_totais']
        
        tscv = TimeSeriesSplit(n_splits=5)
        parametros_grid = {
            'n_estimators': [100],
            'max_depth': [10],
            'min_samples_split': [2],
            'min_samples_leaf': [1]
        }
        
        rf = RandomForestRegressor(random_state=42)
        busca_grid = GridSearchCV(
            estimator=rf, param_grid=parametros_grid,
            scoring='neg_mean_squared_error', cv=tscv, n_jobs=-1
        )
        busca_grid.fit(X, y)
        
        melhor_modelo = busca_grid.best_estimator_
        y_pred = melhor_modelo.predict(X)
        
        metricas = {
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred)
        }
        
        return melhor_modelo, metricas, y_pred
    
    def _gerar_caracteristicas_futuras(self, df: pd.DataFrame, ano_alvo: int) -> pd.DataFrame:
        """Gera matriz de características para previsões futuras"""
        meses_futuros = pd.DataFrame({
            'ano': [ano_alvo] * 12,
            'mes': list(range(1, 13))
        })
        
        meses_futuros['mes_ordinal'] = (meses_futuros['ano'] - df['ano'].min()) * 12 + meses_futuros['mes']
        meses_futuros['estacao'] = meses_futuros['mes'].apply(self._obter_estacao)
        meses_futuros = pd.get_dummies(meses_futuros, columns=['estacao'], drop_first=True)
        
        return meses_futuros

    def prever(self, geocodigo: int, doenca: str = "dengue", 
               se_inicio: int = 1, se_fim: int = 53,
               ano_inicio: int = 2015, ano_fim: int = 2024,
               ano_previsao: int = 2025) -> Optional[DadosDengue]:
        """
        Método principal para buscar dados históricos e gerar previsões
        
        Parâmetros:
            geocodigo (int): Código do IBGE do município
            doenca (str): Nome da doença (use listar_doencas() para ver as disponíveis)
            se_inicio (int): Semana epidemiológica inicial
            se_fim (int): Semana epidemiológica final
            ano_inicio (int): Ano inicial para dados históricos
            ano_fim (int): Ano final para dados históricos
            ano_previsao (int): Ano para o qual as previsões serão geradas
        
        Retorna:
            Optional[DadosDengue]: Objeto com dados históricos, previsões e métricas do modelo
        """
        # Busca e processa dados históricos
        dados_brutos = self._buscar_dados(geocodigo, doenca, se_inicio, se_fim, ano_inicio, ano_fim)
        if dados_brutos is None:
            return None
            
        # Processa dados históricos
        dados_processados = self._pre_processar_dados(dados_brutos)
        dados_processados_com_estacoes = self._adicionar_estacoes(dados_processados)
        
        # Treina modelo e obtém previsões
        modelo, metricas, previsoes_historicas = self._treinar_modelo(dados_processados_com_estacoes)
        
        # Gera previsões futuras
        caracteristicas_futuras = self._gerar_caracteristicas_futuras(dados_processados, ano_previsao)
        colunas_X = dados_processados_com_estacoes.drop(columns=['casos_totais']).columns
        
        # Garante que as colunas correspondam aos dados de treinamento
        for col in set(colunas_X) - set(caracteristicas_futuras.columns):
            caracteristicas_futuras[col] = 0
        caracteristicas_futuras = caracteristicas_futuras[colunas_X]
        
        # Faz as previsões
        previsoes_futuras = modelo.predict(caracteristicas_futuras)
        caracteristicas_futuras['casos_previstos'] = previsoes_futuras
        
        # Adiciona previsões históricas aos dados processados
        dados_processados_com_estacoes['casos_previstos'] = previsoes_historicas
        
        return DadosDengue(
            dados_historicos=dados_processados_com_estacoes,
            previsoes=caracteristicas_futuras,
            metricas_modelo=metricas
        )
