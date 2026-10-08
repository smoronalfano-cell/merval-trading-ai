import datetime
import numpy as np
import pandas as pd
import ta
import yfinance as yf
import os
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

app = FastAPI(title="Merval Trading AI Engine - Multi-Variable Precision Engine")

# Soporte para PostgreSQL (Supabase) o SQLite como fallback
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./merval_yfinance.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

connect_args = {"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Recomendacion(Base):
    __tablename__ = "recomendaciones"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True)
    fecha = Column(DateTime, default=datetime.datetime.utcnow)
    precio_entrada = Column(Float)
    rsi_valor = Column(Float)
    sma_20 = Column(Float)
    sma_50 = Column(Float)
    rec_tecnica = Column(String)
    elliott_fase = Column(String)
    recomendacion = Column(String)
    stop_loss = Column(Float)
    take_profit = Column(Float)


class PosicionPortafolio(Base):
    __tablename__ = "portafolio"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True)
    cantidad = Column(Float)
    precio_compra = Column(Float)
    fecha_compra = Column(DateTime, default=datetime.datetime.utcnow)


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class PosicionSchema(BaseModel):
    ticker: str
    cantidad: float
    precio_compra: float


def estimar_onda_elliott(df: pd.DataFrame) -> dict:
    if len(df) < 30:
        return {"fase": "Datos insuficientes", "bias": "NEUTRAL"}

    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values

    p_actual = close[-1]
    p_min_60 = np.min(low[-60:])
    p_max_60 = np.max(high[-60:])

    rango = p_max_60 - p_min_60
    if rango == 0:
        return {"fase": "Sin Volatilidad", "bias": "NEUTRAL"}

    nivel_relativo = (p_actual - p_min_60) / rango
    idx_max = np.argmax(high[-60:])
    idx_min = np.argmin(low[-60:])

    if idx_min < idx_max:
        if nivel_relativo > 0.85:
            if idx_max > 50:
                fase = "Onda 5 (Impulso Final Alcista)"
                bias = "PRECAUCION_COMPRA"
            else:
                fase = "Onda 3 (Fuerte Impulso Alcista)"
                bias = "ALCISTA_FUERTE"
        elif 0.382 <= nivel_relativo <= 0.618:
            fase = "Onda 4 (Corrección de Consolidación)"
            bias = "OPORTUNIDAD_COMPRA"
        elif nivel_relativo < 0.382:
            fase = "Onda A/C (Corrección A-B-C)"
            bias = "BAJISTA"
        else:
            fase = "Onda 1/2 (Desarrollo Inicial)"
            bias = "ALCISTA"
    else:
        if nivel_relativo < 0.20:
            fase = "Onda C (Agotamiento Bajista / Suelo)"
            bias = "OPORTUNIDAD_COMPRA"
        elif nivel_relativo < 0.50:
            fase = "Onda B (Rebote Correctivo Temporal)"
            bias = "BAJISTA"
        else:
            fase = "Inicio Nuevo Ciclo (Onda 1)"
            bias = "ALCISTA"

    return {"fase": fase, "bias": bias}


def procesar_ticker_yfinance(ticker_input: str):
    ticker_clean = ticker_input.strip().upper()
    ticker_yf = f"{ticker_clean}.BA" if not ticker_clean.endswith(".BA") else ticker_clean

    ticker_obj = yf.Ticker(ticker_yf)
    df = ticker_obj.history(period="6mo", interval="1d")

    if df.empty or len(df) < 30:
        raise HTTPException(
            status_code=404,
            detail=f"No se obtuvieron suficientes datos en Yahoo Finance para {ticker_yf}."
        )

    close_series = df['Close']
    df['RSI'] = ta.momentum.RSIIndicator(close=close_series, window=14).rsi()
    df['SMA_20'] = ta.trend.SMAIndicator(close=close_series, window=20).sma_indicator()
    df['SMA_50'] = ta.trend.SMAIndicator(close=close_series, window=50).sma_indicator()
    
    # NUEVOS INDICADORES DE PRECISIÓN
    # MACD
    macd_ind = ta.trend.MACD(close=close_series)
    df['MACD'] = macd_ind.macd()
    df['MACD_SIGNAL'] = macd_ind.macd_signal()
    df['MACD_HIST'] = macd_ind.macd_diff()

    # VOLUMEN SMA 20
    df['VOL_SMA_20'] = ta.trend.SMAIndicator(close=df['Volume'], window=20).sma_indicator()

    # ATR (Average True Range para Stop Loss / Take Profit dinámico)
    df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=close_series, window=14).average_true_range()

    ultimo = df.iloc[-1]
    precio_actual = round(float(ultimo['Close']), 2)

    if pd.isna(precio_actual) or precio_actual <= 0:
        raise HTTPException(status_code=400, detail="El precio obtenido no es válido.")

    rsi_val = round(float(ultimo['RSI']), 2) if pd.notnull(ultimo['RSI']) else 50.0
    sma20_val = round(float(ultimo['SMA_20']), 2) if pd.notnull(ultimo['SMA_20']) else precio_actual
    sma50_val = round(float(ultimo['SMA_50']), 2) if pd.notnull(ultimo['SMA_50']) else precio_actual
    
    macd_hist = float(ultimo['MACD_HIST']) if pd.notnull(ultimo['MACD_HIST']) else 0.0
    vol_actual = float(ultimo['Volume']) if pd.notnull(ultimo['Volume']) else 0
    vol_sma20 = float(ultimo['VOL_SMA_20']) if pd.notnull(ultimo['VOL_SMA_20']) else 1
    atr_val = float(ultimo['ATR']) if pd.notnull(ultimo['ATR']) else (precio_actual * 0.04)

    # REGLAS DE FILTRADO MULTIVARIABLE
    volumen_fuerte = vol_actual > vol_sma20
    vela_alcista = float(ultimo['Close']) > float(ultimo['Open'])
    tendencia_alcista = sma20_val > sma50_val and macd_hist > 0

    # Puntuación Técnica Integrada
    if rsi_val < 38 and tendencia_alcista and volumen_fuerte and vela_alcista:
        rec_tecnica = "COMPRA FUERTE (ALTA CONVICCION)"
    elif (rsi_val < 48 and tendencia_alcista) or (rsi_val < 35 and vela_alcista):
        rec_tecnica = "COMPRA"
    elif rsi_val > 70 or (rsi_val > 62 and macd_hist < 0 and not vela_alcista):
        rec_tecnica = "VENTA FUERTE"
    elif rsi_val > 58 and not tendencia_alcista:
        rec_tecnica = "VENTA"
    else:
        rec_tecnica = "MANTENER"

    elliott_info = estimar_onda_elliott(df)
    elliott_fase = elliott_info["fase"]
    elliott_bias = elliott_info["bias"]

    # MATRIZ FINAL DE DECISIÓN
    if "COMPRA FUERTE" in rec_tecnica and ("COMPRA" in elliott_bias or "ALCISTA" in elliott_bias):
        rec_final = "COMPRA FUERTE (ALTA CONFLUENCIA)"
        mult_sl, mult_tp = 1.5, 3.0
    elif "COMPRA" in rec_tecnica and "BAJISTA" not in elliott_bias:
        rec_final = "COMPRA"
        mult_sl, mult_tp = 1.2, 2.2
    elif "VENTA FUERTE" in rec_tecnica or ("VENTA" in rec_tecnica and "BAJISTA" in elliott_bias):
        rec_final = "VENTA FUERTE"
        mult_sl, mult_tp = 1.2, 2.0
    elif "VENTA" in rec_tecnica or elliott_bias == "BAJISTA":
        rec_final = "VENTA"
        mult_sl, mult_tp = 1.0, 1.8
    else:
        rec_final = "NEUTRAL / MANTENER"
        mult_sl, mult_tp = 1.0, 1.5

    # Stop Loss y Take Profit adaptados a la volatilidad real (ATR)
    sl = round(precio_actual - (atr_val * mult_sl), 2)
    tp = round(precio_actual + (atr_val * mult_tp), 2)

    return {
        "ticker": ticker_yf,
        "precio": precio_actual,
        "rsi": rsi_val,
        "sma_20": sma20_val,
        "sma_50": sma50_val,
        "rec_tecnica": rec_tecnica,
        "elliott_fase": elliott_fase,
        "recomendacion": rec_final,
        "stop_loss": sl,
        "take_profit": tp
    }


@app.get("/", response_class=HTMLResponse)
def index():
    html_content = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Merval AI Engine - Portfolio & Analysis</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-900 text-gray-100 min-h-screen p-6 font-sans">
        <div class="max-w-6xl mx-auto space-y-6">
            
            <header class="flex justify-between items-center border-b border-gray-800 pb-4">
                <h1 class="text-2xl font-bold text-emerald-400">📊 Merval AI Engine (Multi-Variable)</h1>
                <nav class="flex gap-4">
                    <button id="tabAnalizadorBtn" onclick="verTab('analizador')" class="px-4 py-2 bg-emerald-600 font-bold rounded">Analizador</button>
                    <button id="tabPortafolioBtn" onclick="verTab('portafolio')" class="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded">Mi Portafolio</button>
                </nav>
            </header>

            <!-- SECCION ANALIZADOR -->
            <div id="secAnalizador" class="space-y-6">
                <!-- Buscador -->
                <section class="bg-gray-800 p-4 rounded-lg shadow-lg flex gap-4">
                    <input id="tickerInput" type="text" placeholder="Ej: EDN, YPFD, GGAL, BMA, PAMP..." 
                           class="flex-1 bg-gray-700 text-white px-4 py-2 rounded focus:outline-none focus:ring-2 focus:ring-emerald-500 uppercase">
                    <button onclick="consultarTicker()" 
                            class="bg-emerald-600 hover:bg-emerald-500 text-white font-bold px-6 py-2 rounded transition">
                        Consultar Ticker
                    </button>
                </section>

                <!-- Resultado -->
                <section id="resultadoCard" class="hidden bg-gray-800 p-6 rounded-lg shadow-lg space-y-4">
                    <div class="flex justify-between items-center border-b border-gray-700 pb-2">
                        <h2 id="resTicker" class="text-2xl font-bold text-emerald-400"></h2>
                        <span id="resPrecio" class="text-2xl font-bold text-white"></span>
                    </div>

                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div class="bg-gray-700/60 p-4 rounded-lg border border-gray-600">
                            <p class="text-xs text-gray-400 font-semibold uppercase tracking-wider">Variable 1: Análisis Multi-Técnico (RSI, MACD, Vol, SMA)</p>
                            <p id="resTecnica" class="text-lg font-bold text-emerald-300 mt-1"></p>
                        </div>
                        <div class="bg-gray-700/60 p-4 rounded-lg border border-gray-600">
                            <p class="text-xs text-indigo-300 font-semibold uppercase tracking-wider">Variable 2: Ondas de Elliott</p>
                            <p id="resElliott" class="text-lg font-bold text-indigo-200 mt-1"></p>
                        </div>
                    </div>

                    <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                        <div class="bg-gray-700 p-3 rounded">
                            <p class="text-gray-400 text-sm">Señal Final</p>
                            <p id="resRec" class="text-lg font-bold mt-1 text-emerald-400"></p>
                        </div>
                        <div class="bg-gray-700 p-3 rounded">
                            <p class="text-gray-400 text-sm">RSI (14)</p>
                            <p id="resRsi" class="text-lg font-semibold mt-1"></p>
                        </div>
                        <div class="bg-gray-700 p-3 rounded">
                            <p class="text-gray-400 text-sm">Stop Loss Dinámico (ATR)</p>
                            <p id="resSL" class="text-lg font-semibold text-red-400 mt-1"></p>
                        </div>
                        <div class="bg-gray-700 p-3 rounded">
                            <p class="text-gray-400 text-sm">Take Profit Dinámico (ATR)</p>
                            <p id="resTP" class="text-lg font-semibold text-green-400 mt-1"></p>
                        </div>
                    </div>
                </section>

                <!-- Historial -->
                <section class="bg-gray-800 p-6 rounded-lg shadow-lg">
                    <div class="flex justify-between items-center mb-4">
                        <h2 class="text-xl font-bold">Historial de Señales Guardadas</h2>
                        <button onclick="cargarHistorial()" class="text-sm bg-gray-700 hover:bg-gray-600 px-3 py-1 rounded">Actualizar</button>
                    </div>
                    
                    <div id="resumenStats" class="mb-4 text-sm text-gray-300"></div>

                    <div class="overflow-x-auto">
                        <table class="w-full text-left text-sm text-gray-300">
                            <thead class="bg-gray-700 text-gray-400 uppercase text-xs">
                                <tr>
                                    <th class="p-3">Fecha</th>
                                    <th class="p-3">Ticker</th>
                                    <th class="p-3">P. Entrada</th>
                                    <th class="p-3">Var 1: Técnica</th>
                                    <th class="p-3">Var 2: Elliott</th>
                                    <th class="p-3">Señal Final</th>
                                    <th class="p-3">P. Actual</th>
                                    <th class="p-3">Variación</th>
                                    <th class="p-3">Resultado</th>
                                    <th class="p-3 text-center">Acción</th>
                                </tr>
                            </thead>
                            <tbody id="tablaHistorial" class="divide-y divide-gray-700">
                                <tr><td colspan="10" class="p-4 text-center text-gray-500">Cargando datos...</td></tr>
                            </tbody>
                        </table>
                    </div>
                </section>
            </div>

            <!-- SECCION PORTAFOLIO -->
            <div id="secPortafolio" class="hidden space-y-6">
                <!-- Formulario Agregar Posicion -->
                <section class="bg-gray-800 p-6 rounded-lg shadow-lg">
                    <h2 class="text-xl font-bold mb-4 text-emerald-400">💼 Registrar Acción Comprada</h2>
                    <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
                        <input id="portTicker" type="text" placeholder="Ticker (ej: YPFD)" class="bg-gray-700 px-3 py-2 rounded text-white uppercase focus:outline-none">
                        <input id="portCantidad" type="number" placeholder="Cantidad (ej: 50)" class="bg-gray-700 px-3 py-2 rounded text-white focus:outline-none">
                        <input id="portPrecio" type="number" step="0.01" placeholder="Precio Compra $ (ej: 25000)" class="bg-gray-700 px-3 py-2 rounded text-white focus:outline-none">
                        <button onclick="guardarPosicion()" class="bg-emerald-600 hover:bg-emerald-500 font-bold px-4 py-2 rounded transition">Guardar en Portafolio</button>
                    </div>
                </section>

                <!-- Tabla de Tenencias y Recomendaciones de Venta -->
                <section class="bg-gray-800 p-6 rounded-lg shadow-lg">
                    <div class="flex justify-between items-center mb-4">
                        <h2 class="text-xl font-bold">Monitoreo y Recomendaciones de Venta</h2>
                        <button onclick="cargarPortafolio()" class="text-sm bg-gray-700 hover:bg-gray-600 px-3 py-1 rounded">Actualizar Mercado</button>
                    </div>

                    <div id="resumenPortafolio" class="mb-4 text-sm text-gray-300"></div>

                    <div class="overflow-x-auto">
                        <table class="w-full text-left text-sm text-gray-300">
                            <thead class="bg-gray-700 text-gray-400 uppercase text-xs">
                                <tr>
                                    <th class="p-3">Ticker</th>
                                    <th class="p-3">Cantidad</th>
                                    <th class="p-3">P. Compra</th>
                                    <th class="p-3">P. Actual</th>
                                    <th class="p-3">Valor Total</th>
                                    <th class="p-3">Rendimiento ($)</th>
                                    <th class="p-3">Rendimiento (%)</th>
                                    <th class="p-3">Recomendación Venta AI</th>
                                    <th class="p-3 text-center">Acción</th>
                                </tr>
                            </thead>
                            <tbody id="tablaPortafolio" class="divide-y divide-gray-700">
                                <tr><td colspan="9" class="p-4 text-center text-gray-500">Cargando portafolio...</td></tr>
                            </tbody>
                        </table>
                    </div>
                </section>
            </div>

        </div>

        <script>
            function verTab(tab) {
                const secAna = document.getElementById('secAnalizador');
                const secPort = document.getElementById('secPortafolio');
                const btnAna = document.getElementById('tabAnalizadorBtn');
                const btnPort = document.getElementById('tabPortafolioBtn');

                if (tab === 'analizador') {
                    secAna.classList.remove('hidden');
                    secPort.classList.add('hidden');
                    btnAna.className = "px-4 py-2 bg-emerald-600 font-bold rounded";
                    btnPort.className = "px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded";
                } else {
                    secAna.classList.add('hidden');
                    secPort.classList.remove('hidden');
                    btnPort.className = "px-4 py-2 bg-emerald-600 font-bold rounded";
                    btnAna.className = "px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded";
                    cargarPortafolio();
                }
            }

            async function consultarTicker() {
                const ticker = document.getElementById('tickerInput').value.trim();
                if (!ticker) return alert('Ingresa un ticker válido');

                const card = document.getElementById('resultadoCard');
                try {
                    const res = await fetch(`/api/consultar/${ticker}`);
                    if (!res.ok) throw new Error('No se encontraron datos para el ticker');
                    const data = await res.json();

                    document.getElementById('resTicker').innerText = data.ticker;
                    document.getElementById('resPrecio').innerText = `$ ${data.precio}`;
                    document.getElementById('resTecnica').innerText = data.rec_tecnica;
                    document.getElementById('resElliott').innerText = data.elliott_fase;
                    document.getElementById('resRec').innerText = data.recomendacion;
                    document.getElementById('resRsi').innerText = data.rsi;
                    document.getElementById('resSL').innerText = `$ ${data.stop_loss}`;
                    document.getElementById('resTP').innerText = `$ ${data.take_profit}`;

                    card.classList.remove('hidden');
                    cargarHistorial();
                } catch (err) {
                    alert(err.message);
                }
            }

            async function eliminarRegistro(id) {
                if (!confirm('¿Seguro que deseas eliminar este registro?')) return;
                try {
                    const res = await fetch(`/api/eliminar/${id}`, { method: 'DELETE' });
                    if (res.ok) cargarHistorial();
                } catch (err) {
                    console.error(err);
                }
            }

            async function cargarHistorial() {
                try {
                    const res = await fetch('/api/comparar-historial');
                    const data = await res.json();
                    const tbody = document.getElementById('tablaHistorial');
                    tbody.innerHTML = '';

                    if (data.mensaje || !data.historial || data.historial.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="10" class="p-4 text-center">No hay datos registrados aún</td></tr>';
                        return;
                    }

                    document.getElementById('resumenStats').innerText = 
                        `Total: ${data.resumen.total_registros} | Evaluados: ${data.resumen.evaluados} | Win Rate: ${data.resumen.tasa_acierto_pct}`;

                    data.historial.forEach(r => {
                        const tr = document.createElement('tr');
                        let resBadge = '<span class="px-2 py-1 bg-gray-600 rounded text-xs">SIN DATOS</span>';
                        if (r.resultado === 'ACIERTO') resBadge = '<span class="px-2 py-1 bg-green-900 text-green-300 rounded text-xs font-bold">ACIERTO</span>';
                        if (r.resultado === 'FALLO') resBadge = '<span class="px-2 py-1 bg-red-900 text-red-300 rounded text-xs font-bold">FALLO</span>';

                        const precioEntradaTxt = r.precio_original ? `$ ${r.precio_original}` : '<span class="text-red-400 font-bold">null</span>';

                        tr.innerHTML = `
                            <td class="p-3">${r.fecha}</td>
                            <td class="p-3 font-bold">${r.ticker}</td>
                            <td class="p-3">${precioEntradaTxt}</td>
                            <td class="p-3 text-emerald-300 text-xs">${r.rec_tecnica || 'N/A'}</td>
                            <td class="p-3 text-indigo-300 text-xs">${r.elliott_fase || 'N/A'}</td>
                            <td class="p-3 font-bold">${r.recomendacion}</td>
                            <td class="p-3">$ ${r.precio_actual || '-'}</td>
                            <td class="p-3">${r.variacion_real_pct}</td>
                            <td class="p-3">${resBadge}</td>
                            <td class="p-3 text-center">
                                <button onclick="eliminarRegistro(${r.id})" class="bg-red-600 hover:bg-red-700 text-white px-2 py-1 rounded text-xs font-bold transition">🗑️ Borrar</button>
                            </td>
                        `;
                        tbody.appendChild(tr);
                    });
                } catch (err) {
                    console.error("Error al cargar el historial:", err);
                }
            }

            // FUNCIONES PORTAFOLIO
            async function guardarPosicion() {
                const ticker = document.getElementById('portTicker').value.trim();
                const cantidad = parseFloat(document.getElementById('portCantidad').value);
                const precio = parseFloat(document.getElementById('portPrecio').value);

                if (!ticker || !cantidad || !precio) return alert('Por favor completa todos los campos');

                try {
                    const res = await fetch('/api/portafolio/agregar', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ ticker: ticker, cantidad: cantidad, precio_compra: precio })
                    });
                    if (res.ok) {
                        document.getElementById('portTicker').value = '';
                        document.getElementById('portCantidad').value = '';
                        document.getElementById('portPrecio').value = '';
                        cargarPortafolio();
                    } else {
                        alert('Error al guardar la posición');
                    }
                } catch (err) {
                    console.error(err);
                }
            }

            async function borrarPosicion(id) {
                if (!confirm('¿Deseas quitar esta posición del portafolio?')) return;
                try {
                    const res = await fetch(`/api/portafolio/eliminar/${id}`, { method: 'DELETE' });
                    if (res.ok) cargarPortafolio();
                } catch (err) {
                    console.error(err);
                }
            }

            async function cargarPortafolio() {
                try {
                    const res = await fetch('/api/portafolio');
                    const data = await res.json();
                    const tbody = document.getElementById('tablaPortafolio');
                    tbody.innerHTML = '';

                    if (!data.posiciones || data.posiciones.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="9" class="p-4 text-center text-gray-500">No tienes acciones en tu portafolio aún.</td></tr>';
                        document.getElementById('resumenPortafolio').innerText = '';
                        return;
                    }

                    document.getElementById('resumenPortafolio').innerText = 
                        `Inversión Total: $ ${data.resumen.inversion_total} | Valor Actual: $ ${data.resumen.valor_actual_total} | Rendimiento Global: $ ${data.resumen.ganancia_total_monto} (${data.resumen.ganancia_total_pct}%)`;

                    data.posiciones.forEach(p => {
                        const tr = document.createElement('tr');

                        let pnlColor = p.pnl_monto >= 0 ? 'text-green-400 font-bold' : 'text-red-400 font-bold';
                        
                        let recBadge = '<span class="px-2 py-1 bg-gray-700 text-gray-300 rounded text-xs font-bold">MANTENER</span>';
                        if (p.recomendacion_venta.includes('VENTA') || p.recomendacion_venta.includes('STOP LOSS')) {
                            recBadge = `<span class="px-2 py-1 bg-red-900 text-red-200 rounded text-xs font-bold">${p.recomendacion_venta}</span>`;
                        } else if (p.recomendacion_venta.includes('TAKE PROFIT')) {
                            recBadge = `<span class="px-2 py-1 bg-green-900 text-green-200 rounded text-xs font-bold">${p.recomendacion_venta}</span>`;
                        }

                        tr.innerHTML = `
                            <td class="p-3 font-bold">${p.ticker}</td>
                            <td class="p-3">${p.cantidad}</td>
                            <td class="p-3">$ ${p.precio_compra}</td>
                            <td class="p-3">$ ${p.precio_actual}</td>
                            <td class="p-3">$ ${p.valor_total_actual}</td>
                            <td class="p-3 ${pnlColor}">$ ${p.pnl_monto}</td>
                            <td class="p-3 ${pnlColor}">${p.pnl_pct}%</td>
                            <td class="p-3">${recBadge}</td>
                            <td class="p-3 text-center">
                                <button onclick="borrarPosicion(${p.id})" class="bg-red-600 hover:bg-red-700 text-white px-2 py-1 rounded text-xs font-bold transition">🗑️ Eliminar</button>
                            </td>
                        `;
                        tbody.appendChild(tr);
                    });
                } catch (err) {
                    console.error("Error al cargar portafolio:", err);
                }
            }

            window.onload = cargarHistorial;
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/api/consultar/{ticker}")
def consultar_y_registrar(ticker: str, db: Session = Depends(get_db)):
    datos = procesar_ticker_yfinance(ticker)

    nueva_rec = Recomendacion(
        ticker=datos["ticker"],
        precio_entrada=datos["precio"],
        rsi_valor=datos["rsi"],
        sma_20=datos["sma_20"],
        sma_50=datos["sma_50"],
        rec_tecnica=datos["rec_tecnica"],
        elliott_fase=datos["elliott_fase"],
        recomendacion=datos["recomendacion"],
        stop_loss=datos["stop_loss"],
        take_profit=datos["take_profit"]
    )
    db.add(nueva_rec)
    db.commit()
    db.refresh(nueva_rec)

    return {
        "status": "Consulta guardada con éxito",
        "id_registro": nueva_rec.id,
        "fecha_consulta": nueva_rec.fecha.strftime("%Y-%m-%d %H:%M:%S"),
        **datos
    }


@app.delete("/api/eliminar/{id_registro}")
def eliminar_registro(id_registro: int, db: Session = Depends(get_db)):
    registro = db.query(Recomendacion).filter(Recomendacion.id == id_registro).first()
    if not registro:
        raise HTTPException(status_code=404, detail="Registro no encontrado")
    db.delete(registro)
    db.commit()
    return {"mensaje": f"Registro {id_registro} eliminado con éxito"}


@app.get("/api/comparar-historial")
def comparar_historial(db: Session = Depends(get_db)):
    registros = db.query(Recomendacion).order_by(Recomendacion.fecha.desc()).all()
    if not registros:
        return {"mensaje": "No hay consultas grabadas en la base de datos"}

    tickers_unicos = list(set([r.ticker for r in registros]))
    precios_actuales = {}

    for t in tickers_unicos:
        try:
            fast_data = yf.Ticker(t).fast_info
            precios_actuales[t] = round(fast_data.last_price, 2)
        except Exception:
            precios_actuales[t] = None

    resultado = []
    total_evaluados = 0
    total_aciertos = 0

    for r in registros:
        precio_hoy = precios_actuales.get(r.ticker)
        if precio_hoy and r.precio_entrada:
            var_pct = round(((precio_hoy - r.precio_entrada) / r.precio_entrada) * 100, 2)

            es_compra = "COMPRA" in r.recomendacion
            es_venta = "VENTA" in r.recomendacion

            if (es_compra and var_pct > 0) or (es_venta and var_pct < 0):
                estado = "ACIERTO"
                total_aciertos += 1
            elif var_pct == 0:
                estado = "SIN CAMBIOS"
            else:
                estado = "FALLO"

            total_evaluados += 1
        else:
            var_pct = 0.0
            estado = "SIN DATOS"

        resultado.append({
            "id": r.id,
            "fecha": r.fecha.strftime("%Y-%m-%d %H:%M"),
            "ticker": r.ticker.replace(".BA", ""),
            "precio_original": r.precio_entrada,
            "rec_tecnica": getattr(r, "rec_tecnica", "N/A"),
            "elliott_fase": getattr(r, "elliott_fase", "N/A"),
            "recomendacion": r.recomendacion,
            "stop_loss": r.stop_loss,
            "take_profit": r.take_profit,
            "precio_actual": precio_hoy,
            "variacion_real_pct": f"{var_pct}%",
            "resultado": estado
        })

    win_rate = round((total_aciertos / total_evaluados * 100), 2) if total_evaluados > 0 else 0.0

    return {
        "resumen": {
            "total_registros": len(registros),
            "evaluados": total_evaluados,
            "aciertos": total_aciertos,
            "tasa_acierto_pct": f"{win_rate}%"
        },
        "historial": resultado
    }


# ENDPOINTS DE PORTAFOLIO
@app.post("/api/portafolio/agregar")
def agregar_portafolio(pos: PosicionSchema, db: Session = Depends(get_db)):
    ticker_clean = pos.ticker.strip().upper()
    ticker_yf = f"{ticker_clean}.BA" if not ticker_clean.endswith(".BA") else ticker_clean

    nueva_pos = PosicionPortafolio(
        ticker=ticker_yf,
        cantidad=pos.cantidad,
        precio_compra=pos.precio_compra
    )
    db.add(nueva_pos)
    db.commit()
    db.refresh(nueva_pos)
    return {"mensaje": "Posición agregada con éxito", "id": nueva_pos.id}


@app.delete("/api/portafolio/eliminar/{id_posicion}")
def eliminar_portafolio(id_posicion: int, db: Session = Depends(get_db)):
    pos = db.query(PosicionPortafolio).filter(PosicionPortafolio.id == id_posicion).first()
    if not pos:
        raise HTTPException(status_code=404, detail="Posición no encontrada")
    db.delete(pos)
    db.commit()
    return {"mensaje": "Posición eliminada"}


@app.get("/api/portafolio")
def obtener_portafolio(db: Session = Depends(get_db)):
    posiciones = db.query(PosicionPortafolio).all()
    if not posiciones:
        return {"posiciones": [], "resumen": {}}

    inversion_total = 0.0
    valor_actual_total = 0.0
    resultado_lista = []

    for p in posiciones:
        try:
            analisis = procesar_ticker_yfinance(p.ticker)
            precio_hoy = analisis["precio"]
            rsi_hoy = analisis["rsi"]
            rec_tecnica = analisis["rec_tecnica"]
            elliott_bias = analisis["elliott_fase"]
        except Exception:
            precio_hoy = p.precio_compra
            rsi_hoy = 50.0
            rec_tecnica = "MANTENER"
            elliott_bias = "NEUTRAL"

        inversion_item = p.cantidad * p.precio_compra
        valor_item = p.cantidad * precio_hoy
        pnl_monto = valor_item - inversion_item
        pnl_pct = round(((precio_hoy - p.precio_compra) / p.precio_compra) * 100, 2)

        inversion_total += inversion_item
        valor_actual_total += valor_item

        # Reglas AI para recomendar VENTA
        if pnl_pct <= -6.0:
            rec_venta = "EJECUTAR STOP LOSS (CERRAR)"
        elif pnl_pct >= 15.0:
            rec_venta = "EJECUTAR TAKE PROFIT (ASEGURAR)"
        elif "VENTA" in rec_tecnica or "BAJISTA" in elliott_bias:
            rec_venta = "VENTA SUGERIDA (SEÑAL TÉCNICA)"
        elif rsi_hoy > 68:
            rec_venta = "VENTA PARCIAL (SOBRECOMPRA)"
        else:
            rec_venta = "MANTENER POSICIÓN"

        resultado_lista.append({
            "id": p.id,
            "ticker": p.ticker.replace(".BA", ""),
            "cantidad": p.cantidad,
            "precio_compra": p.precio_compra,
            "precio_actual": precio_hoy,
            "valor_total_actual": round(valor_item, 2),
            "pnl_monto": round(pnl_monto, 2),
            "pnl_pct": pnl_pct,
            "recomendacion_venta": rec_venta
        })

    ganancia_total_monto = valor_actual_total - inversion_total
    ganancia_total_pct = round((ganancia_total_monto / inversion_total * 100), 2) if inversion_total > 0 else 0.0

    return {
        "resumen": {
            "inversion_total": round(inversion_total, 2),
            "valor_actual_total": round(valor_actual_total, 2),
            "ganancia_total_monto": round(ganancia_total_monto, 2),
            "ganancia_total_pct": ganancia_total_pct
        },
        "posiciones": resultado_lista
    }
