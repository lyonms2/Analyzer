# ema_barriers_config.py
# Configurações personalizadas das Barreiras EMA para cada criptomoeda
# Barreira 1 = EMA Rápida | Barreira 2 = EMA Média | Barreira 3 = EMA Lenta

# Lista de criptomoedas disponíveis
CRYPTO_LIST = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'HYPE/USDT', 'PUMP/USDT', 'ENA/USDT', 
    'FARTCOIN/USDT', 'BONK/USDT', 'BNB/USDT', 'ADA/USDT', 'XRP/USDT', 'DOGE/USDT',
    'TRX/USDT', 'LINK/USDT', 'LTC/USDT', 'PENGU/USDT', 'DOT/USDT', 'BCH/USDT', 
    'SHIB/USDT', 'AVAX/USDT', 'OP/USDT', 'UNI/USDT', 'ATOM/USDT', 'ETC/USDT', 
    'XLM/USDT', 'FIL/USDT', 'APT/USDT', 'SUI/USDT', 'HBAR/USDT', 'ZORA/USDT', 
    'AR/USDT', 'INJ/USDT', 'PEPE/USDT', 'NEAR/USDT', 'STX/USDT', 'ALGO/USDT', 
    'IMX/USDT', 'WIF/USDT', 'MINA/USDT', 'DYDX/USDT', 'TIA/USDT', 'JTO/USDT', 
    'AAVE/USDT', 'PYTH/USDT', 'SAND/USDT', 'CAKE/USDT', 'XMR/USDT', 'BLUR/USDT', 
    'GMX/USDT', 'LDO/USDT', 'FET/USDT', 'DYM/USDT', 'GMT/USDT', 'MEME/USDT', 
    'BOME/USDT', 'YGG/USDT', 'RUNE/USDT', 'CELO/USDT', 'WLD/USDT', 'ONDO/USDT', 
    'SEI/USDT', 'JUP/USDT', 'POPCAT/USDT', 'TAO/USDT', 'TON/USDT'
]

# Configurações das Barreiras EMA por ativo
# Configuração baseada na volatilidade e comportamento de cada ativo
EMA_BARRIERS = {
    # === TIER 1: PRINCIPAIS (Bitcoin, Ethereum) ===
    'BTC/USDT': {
        'barreira_1': 21,   # EMA rápida - movimentos de curto prazo
        'barreira_2': 50,   # EMA média - confirmação de tendência  
        'barreira_3': 200   # EMA lenta - tendência de longo prazo
    },
    
    'ETH/USDT': {
        'barreira_1': 21,
        'barreira_2': 50,
        'barreira_3': 200
    },
    
    # === TIER 2: LARGE CAPS ===
    'BNB/USDT': {
        'barreira_1': 20,
        'barreira_2': 50,
        'barreira_3': 200
    },
    
    'SOL/USDT': {
        'barreira_1': 18,   # Mais volátil, EMA mais rápida
        'barreira_2': 45,
        'barreira_3': 180
    },
    
    'ADA/USDT': {
        'barreira_1': 21,
        'barreira_2': 55,
        'barreira_3': 200
    },
    
    'XRP/USDT': {
        'barreira_1': 20,
        'barreira_2': 50,
        'barreira_3': 200
    },
    
    'DOGE/USDT': {
        'barreira_1': 15,   # Memecoin volátil
        'barreira_2': 40,
        'barreira_3': 150
    },
    
    'LINK/USDT': {
        'barreira_1': 21,
        'barreira_2': 50,
        'barreira_3': 200
    },
    
    'LTC/USDT': {
        'barreira_1': 21,
        'barreira_2': 50,
        'barreira_3': 200
    },
    
    'DOT/USDT': {
        'barreira_1': 20,
        'barreira_2': 50,
        'barreira_3': 200
    },
    
    'BCH/USDT': {
        'barreira_1': 21,
        'barreira_2': 50,
        'barreira_3':