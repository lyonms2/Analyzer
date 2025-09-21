# config.py - Configurações centralizadas do analisador de criptomoedas

# Configurações padrão dos indicadores
DEFAULT_SETTINGS = {
    'ema': {
        'fast': 12,
        'slow': 26,
        'trend': 200
    },
    'rsi': {
        'period': 14,
        'oversold': 30,
        'overbought': 70
    },
    'stoch_rsi': {
        'k_period': 14,
        'd_period': 3,
        'oversold': 20,
        'overbought': 80
    }
}

# Lista de criptomoedas disponíveis
CRYPTO_SYMBOLS = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD", 
    "Binance Coin": "BNB-USD",
    "Cardano": "ADA-USD",
    "Solana": "SOL-USD",
    "XRP": "XRP-USD",
    "Dogecoin": "DOGE-USD",
    "Polygon": "MATIC-USD",
    "Polkadot": "DOT-USD",
    "Avalanche": "AVAX-USD",
    "Chainlink": "LINK-USD",
    "Uniswap": "UNI-USD",
    "Litecoin": "LTC-USD",
    "Bitcoin Cash": "BCH-USD",
    "Stellar": "XLM-USD"
}

# Períodos disponíveis para análise
AVAILABLE_PERIODS = {
    "1 mês": "1mo",
    "3 meses": "3mo", 
    "6 meses": "6mo",
    "1 ano": "1y",
    "2 anos": "2y",
    "5 anos": "5y"
}

# Cores para os gráficos
CHART_COLORS = {
    'ema': ['orange', 'blue', 'green', 'red', 'purple'],
    'rsi': 'purple',
    'stoch_k': 'blue',
    'stoch_d': 'orange',
    'signals': {
        1: 'green',      # Compra forte
        0.5: 'lightgreen',  # Compra moderada
        0: 'gray',       # Neutro
        -0.5: 'orange',  # Venda moderada
        -1: 'red'        # Venda forte
    }
}

# Configurações dos gráficos
CHART_CONFIG = {
    'height': 800,
    'subplot_heights': [0.5, 0.2, 0.2, 0.1],
    'vertical_spacing': 0.03,
    'reference_lines': {
        'rsi': [30, 50, 70],
        'stoch_rsi': [20, 80]
    }
}

# Textos dos sinais
SIGNAL_TEXTS = {
    1: "🟢 COMPRA FORTE",
    0.5: "🟡 COMPRA MODERADA", 
    0: "⚪ NEUTRO",
    -0.5: "🟠 VENDA MODERADA",
    -1: "🔴 VENDA FORTE"
}

# Configurações da aplicação Streamlit
APP_CONFIG = {
    'page_title': "Analisador de Criptomoedas",
    'page_icon': "₿",
    'layout': "wide",
    'initial_sidebar_state': "expanded"
}

# Limites dos parâmetros configuráveis
PARAMETER_LIMITS = {
    'ema_fast': {'min': 5, 'max': 50},
    'ema_slow': {'min': 20, 'max': 200},
    'ema_trend': {'min': 50, 'max': 300},
    'rsi_period': {'min': 10, 'max': 30},
    'rsi_oversold': {'min': 20, 'max': 40},
    'rsi_overbought': {'min': 60, 'max': 80},
    'stoch_k': {'min': 10, 'max': 20},
    'stoch_d': {'min': 3, 'max': 10},
    'stoch_oversold': {'min': 10, 'max': 30},
    'stoch_overbought': {'min': 70, 'max': 90}
}