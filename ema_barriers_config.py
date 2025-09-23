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
    'AAVE/USDT', 'PYTH/USDT', 'SAND/USDT', 'CAKE/USDT', 'BLUR/USDT', 
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
        'barreira_3': 100   # EMA lenta - tendência de longo prazo
    },
    
    'ETH/USDT': {
        'barreira_1': 21,
        'barreira_2': 50,
        'barreira_3': 100
    },
    
    # === TIER 2: LARGE CAPS ===
    'BNB/USDT': {
        'barreira_1': 21,
        'barreira_2': 50,
        'barreira_3': 100
    },
    
    'SOL/USDT': {
        'barreira_1': 17,   # Mais volátil, EMA mais rápida
        'barreira_2': 50,
        'barreira_3': 110
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
        'barreira_3': 200
    },
    
    'AVAX/USDT': {
        'barreira_1': 18,
        'barreira_2': 45,
        'barreira_3': 180
    },
    
    'ATOM/USDT': {
        'barreira_1': 20,
        'barreira_2': 50,
        'barreira_3': 200
    },
    
    'ETC/USDT': {
        'barreira_1': 21,
        'barreira_2': 50,
        'barreira_3': 200
    },
    
    'XLM/USDT': {
        'barreira_1': 20,
        'barreira_2': 50,
        'barreira_3': 200
    },
    
    # === TIER 3: MID CAPS ===
    'FIL/USDT': {
        'barreira_1': 20,
        'barreira_2': 50,
        'barreira_3': 200
    },
    
    'APT/USDT': {
        'barreira_1': 15,   # Layer 1 nova e volátil
        'barreira_2': 40,
        'barreira_3': 150
    },
    
    'SUI/USDT': {
        'barreira_1': 15,
        'barreira_2': 40,
        'barreira_3': 150
    },
    
    'NEAR/USDT': {
        'barreira_1': 18,
        'barreira_2': 45,
        'barreira_3': 180
    },
    
    'UNI/USDT': {
        'barreira_1': 21,
        'barreira_2': 50,
        'barreira_3': 200
    },
    
    'AAVE/USDT': {
        'barreira_1': 21,
        'barreira_2': 50,
        'barreira_3': 200
    },
    
    'HBAR/USDT': {
        'barreira_1': 20,
        'barreira_2': 50,
        'barreira_3': 200
    },
    
    'AR/USDT': {
        'barreira_1': 18,
        'barreira_2': 45,
        'barreira_3': 180
    },
    
    'INJ/USDT': {
        'barreira_1': 15,
        'barreira_2': 40,
        'barreira_3': 150
    },
    
    'STX/USDT': {
        'barreira_1': 18,
        'barreira_2': 45,
        'barreira_3': 180
    },
    
    'ALGO/USDT': {
        'barreira_1': 20,
        'barreira_2': 50,
        'barreira_3': 200
    },
    
    'IMX/USDT': {
        'barreira_1': 15,
        'barreira_2': 40,
        'barreira_3': 150
    },
    
    'MINA/USDT': {
        'barreira_1': 18,
        'barreira_2': 45,
        'barreira_3': 180
    },
    
    'DYDX/USDT': {
        'barreira_1': 21,
        'barreira_2': 37,
        'barreira_3': 95
    },
    
    'TIA/USDT': {
        'barreira_1': 12,   # Token novo e muito volátil
        'barreira_2': 35,
        'barreira_3': 120
    },
    
    'JTO/USDT': {
        'barreira_1': 12,
        'barreira_2': 35,
        'barreira_3': 120
    },
    
    'PYTH/USDT': {
        'barreira_1': 12,
        'barreira_2': 35,
        'barreira_3': 120
    },
    
    'SAND/USDT': {
        'barreira_1': 18,
        'barreira_2': 45,
        'barreira_3': 180
    },
    
    'CAKE/USDT': {
        'barreira_1': 20,
        'barreira_2': 50,
        'barreira_3': 200
    },
    
    'XMR/USDT': {
        'barreira_1': 21,
        'barreira_2': 50,
        'barreira_3': 200
    },
    
    'BLUR/USDT': {
        'barreira_1': 12,
        'barreira_2': 35,
        'barreira_3': 120
    },
    
    'GMX/USDT': {
        'barreira_1': 15,
        'barreira_2': 40,
        'barreira_3': 150
    },
    
    'LDO/USDT': {
        'barreira_1': 18,
        'barreira_2': 45,
        'barreira_3': 180
    },
    
    'FET/USDT': {
        'barreira_1': 15,
        'barreira_2': 40,
        'barreira_3': 150
    },
    
    'OP/USDT': {
        'barreira_1': 15,
        'barreira_2': 40,
        'barreira_3': 150
    },
    
    'RUNE/USDT': {
        'barreira_1': 15,
        'barreira_2': 40,
        'barreira_3': 150
    },
    
    'CELO/USDT': {
        'barreira_1': 18,
        'barreira_2': 45,
        'barreira_3': 180
    },
    
    'WLD/USDT': {
        'barreira_1': 12,
        'barreira_2': 35,
        'barreira_3': 120
    },
    
    'ONDO/USDT': {
        'barreira_1': 12,
        'barreira_2': 35,
        'barreira_3': 120
    },
    
    'SEI/USDT': {
        'barreira_1': 12,
        'barreira_2': 35,
        'barreira_3': 120
    },
    
    'JUP/USDT': {
        'barreira_1': 12,
        'barreira_2': 35,
        'barreira_3': 120
    },
    
    'TON/USDT': {
        'barreira_1': 15,
        'barreira_2': 40,
        'barreira_3': 150
    },
    
    'TAO/USDT': {
        'barreira_1': 10,   # AI token muito volátil
        'barreira_2': 30,
        'barreira_3': 100
    },
    
    # === TIER 4: SMALL CAPS / ALTCOINS ===
    'TRX/USDT': {
        'barreira_1': 20,
        'barreira_2': 50,
        'barreira_3': 200
    },
    
    'SHIB/USDT': {
        'barreira_1': 12,   # Memecoin volátil
        'barreira_2': 35,
        'barreira_3': 120
    },
    
    'PEPE/USDT': {
        'barreira_1': 10,   # Memecoin extremamente volátil
        'barreira_2': 30,
        'barreira_3': 100
    },
    
    'BONK/USDT': {
        'barreira_1': 10,
        'barreira_2': 30,
        'barreira_3': 100
    },
    
    'WIF/USDT': {
        'barreira_1': 10,
        'barreira_2': 30,
        'barreira_3': 100
    },
    
    'POPCAT/USDT': {
        'barreira_1': 8,    # Memecoin nova e muito volátil
        'barreira_2': 25,
        'barreira_3': 80
    },
    
    'DYM/USDT': {
        'barreira_1': 12,
        'barreira_2': 35,
        'barreira_3': 120
    },
    
    'GMT/USDT': {
        'barreira_1': 15,
        'barreira_2': 40,
        'barreira_3': 150
    },
    
    'MEME/USDT': {
        'barreira_1': 8,
        'barreira_2': 25,
        'barreira_3': 80
    },
    
    'BOME/USDT': {
        'barreira_1': 8,
        'barreira_2': 25,
        'barreira_3': 80
    },
    
    'YGG/USDT': {
        'barreira_1': 15,
        'barreira_2': 40,
        'barreira_3': 150
    },
    
    # === TIER 5: TOKENS NOVOS/ESPECIAIS ===
    'HYPE/USDT': {
        'barreira_1': 8,    # Token muito novo e volátil
        'barreira_2': 25,
        'barreira_3': 80
    },
    
    'PUMP/USDT': {
        'barreira_1': 15,
        'barreira_2': 31,
        'barreira_3': 92
    },
    
    'ENA/USDT': {
        'barreira_1': 10,
        'barreira_2': 30,
        'barreira_3': 100
    },
    
    'FARTCOIN/USDT': {
        'barreira_1': 6,    # Memecoin extremamente volátil
        'barreira_2': 20,
        'barreira_3': 60
    },
    
    'PENGU/USDT': {
        'barreira_1': 8,
        'barreira_2': 25,
        'barreira_3': 80
    },
    
    'ZORA/USDT': {
        'barreira_1': 15,
        'barreira_2': 37,
        'barreira_3': 85
    }
}

# Configurações padrão para tokens não listados
DEFAULT_EMA_CONFIG = {
    'barreira_1': 15,
    'barreira_2': 40, 
    'barreira_3': 120
}

# Função para obter configuração de um ativo
def get_ema_config(symbol: str) -> dict:
    """
    Retorna a configuração EMA para um símbolo específico
    Se não encontrar, retorna configuração padrão
    """
    return EMA_BARRIERS.get(symbol, DEFAULT_EMA_CONFIG)

# Função para adicionar novo ativo
def add_new_asset(symbol: str, barreira_1: int, barreira_2: int, barreira_3: int):
    """
    Adiciona configuração para um novo ativo
    Útil para expandir a lista sem editar o código principal
    """
    EMA_BARRIERS[symbol] = {
        'barreira_1': barreira_1,
        'barreira_2': barreira_2,
        'barreira_3': barreira_3
    }
    
    if symbol not in CRYPTO_LIST:
        CRYPTO_LIST.append(symbol)

# Função para listar ativos por categoria
def get_assets_by_volatility():
    """
    Retorna ativos categorizados por volatilidade baseado nas configurações EMA
    """
    low_volatility = []    # Barreira 1 >= 20
    medium_volatility = [] # Barreira 1 entre 10-19
    high_volatility = []   # Barreira 1 < 10
    
    for symbol, config in EMA_BARRIERS.items():
        if config['barreira_1'] >= 20:
            low_volatility.append(symbol)
        elif config['barreira_1'] >= 10:
            medium_volatility.append(symbol)
        else:
            high_volatility.append(symbol)
    
    return {
        'low_volatility': low_volatility,
        'medium_volatility': medium_volatility,
        'high_volatility': high_volatility
    }

# Validação das configurações
def validate_ema_config():
    """
    Valida se todas as configurações EMA fazem sentido
    Barreira 1 < Barreira 2 < Barreira 3
    """
    invalid_configs = []
    
    for symbol, config in EMA_BARRIERS.items():
        b1, b2, b3 = config['barreira_1'], config['barreira_2'], config['barreira_3']
        
        if not (b1 < b2 < b3):
            invalid_configs.append({
                'symbol': symbol,
                'config': config,
                'error': f"Barreiras devem ser crescentes: {b1} < {b2} < {b3}"
            })
    
    return invalid_configs

# Executar validação se o arquivo for executado diretamente
if __name__ == "__main__":
    print("🔍 Validando configurações EMA...")
    
    # Validar configurações
    invalid = validate_ema_config()
    if invalid:
        print("❌ Configurações inválidas encontradas:")
        for item in invalid:
            print(f"  • {item['symbol']}: {item['error']}")
    else:
        print("✅ Todas as configurações EMA estão válidas!")
    
    # Estatísticas
    print(f"\n📊 Estatísticas:")
    print(f"  • Total de ativos configurados: {len(EMA_BARRIERS)}")
    print(f"  • Total de ativos na lista: {len(CRYPTO_LIST)}")
    
    # Categorização por volatilidade
    volatility_categories = get_assets_by_volatility()
    print(f"\n📈 Categorização por Volatilidade:")
    print(f"  • Baixa volatilidade (EMA ≥ 20): {len(volatility_categories['low_volatility'])} ativos")
    print(f"  • Média volatilidade (EMA 10-19): {len(volatility_categories['medium_volatility'])} ativos")
    print(f"  • Alta volatilidade (EMA < 10): {len(volatility_categories['high_volatility'])} ativos")
    
    print(f"\n💡 Para editar configurações, modifique o dicionário EMA_BARRIERS neste arquivo.")
