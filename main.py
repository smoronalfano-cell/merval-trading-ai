import datetime
import numpy as np
import pandas as pd
import ta
import yfinance as yf
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

app = FastAPI(title="Merval Trading AI Engine - Dual Signal")

DATABASE_URL = "sqlite:///./merval_yfinance.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
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


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


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

    if df.empty or len(df) < 10:
        raise HTTPException(
            status_code=404,
            detail=f"No se obtuvieron suficientes datos en Yahoo Finance para {ticker_yf}."
        )

    close_series = df['Close']
    df['RSI'] = ta.momentum.RSIIndicator(close=close_series, window=14).rsi()
    df['SMA_20'] = ta.trend.SMAIndicator(close=close_series, window=20).sma_indicator()
    df['SMA_50'] = ta.trend.SMAIndicator(close=close_series, window=50).sma_indicator()

    ultimo = df.iloc[-1]
    precio_actual = round(float(ultimo['Close']), 2)

    if pd.isna(precio_actual) or precio_actual <= 0:
        raise HTTPException(status_code=400, detail="El precio obtenido no es válido.")

    rsi_val = round(float(ultimo['RSI']), 2) if pd.notnull(ultimo['RSI']) else 50.0
    sma20_val = round(float(ultimo['SMA_20']), 2) if pd.notnull(ultimo['SMA_20']) else precio_actual
    sma50_val = round(float(ultimo['SMA_50']), 2) if pd.notnull(ultimo['SMA_50']) else precio_actual

    if rsi_val < 35 and precio_actual > sma20_val:
        rec_tecnica = "COMPRA FUERTE"
    elif rsi_val < 45:
        rec_tecnica = "COMPRA"
    elif rsi_val > 70:
        rec_tecnica = "VENTA FUERTE"
    elif rsi_val > 60:
        rec_tecnica = "VENTA"
    else:
        rec_tecnica = "MANTENER"

    elliott_info = estimar_onda_elliott(df)
    elliott_fase = elliott_info["fase"]
    elliott_bias = elliott_info["bias"]

    if "COMPRA" in rec_tecnica and ("COMPRA" in elliott_bias or "ALCISTA" in elliott_bias):
        rec_final = "COMPRA FUERTE (CONFLUENCIA)"
        sl = round(precio_actual * 0.94, 2)
        tp = round(precio_actual * 1.12, 2)
    elif "COMPRA" in rec_tecnica or "OPORTUNIDAD_COMPRA" in elliott_bias:
        rec_final = "COMPRA"
        sl = round(precio_actual * 0.96, 2)
        tp = round(precio_actual * 1.08, 2)
    elif "VENTA" in rec_tecnica and "BAJISTA" in elliott_bias:
        rec_final = "VENTA FUERTE (CONFLUENCIA)"
        sl = round(precio_actual * 1.04, 2)
        tp = round(precio_actual * 0.88, 2)
    elif "VENTA" in rec_tecnica or elliott_bias == "BAJISTA":
        rec_final = "VENTA"
        sl = round(precio_actual * 1.03, 2)
        tp = round(precio_actual * 0.92, 2)
    else:
        rec_final = "MANTENER"
        sl = round(precio_actual * 0.97, 2)
        tp = round(precio_actual * 1.05, 2)

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
        <title>Merval Trading AI - Dual Strategy</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-900 text-gray-100 min-h-screen p-6 font-sans">
        <div class="max-w-6xl mx-auto space-y-6">
            
            <header class="flex justify-between items-center border-b border-gray-800 pb-4">
                <h1 class="text-2xl font-bold text-emerald-400">📊 Merval AI Engine (Técnico + Elliott)</h1>
                <span class="text-sm text-gray-400">FastAPI + Multi-Variable AI Engine</span>
            </header>

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
                        <p class="text-xs text-gray-400 font-semibold uppercase tracking-wider">Variable 1: Análisis Técnico (RSI / SMA)</p>
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
                        <p class="text-gray-400 text-sm">Stop Loss (SL)</p>
                        <p id="resSL" class="text-lg font-semibold text-red-400 mt-1"></p>
                    </div>
                    <div class="bg-gray-700 p-3 rounded">
                        <p class="text-gray-400 text-sm">Take Profit (TP)</p>
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

        <script>
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
                    if (res.ok) {
                        cargarHistorial();
                    } else {
                        alert('Error al eliminar el registro');
                    }
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
                                <button onclick="eliminarRegistro(${r.id})" class="bg-red-600 hover:bg-red-700 text-white px-2 py-1 rounded text-xs font-bold transition">
                                    🗑️ Borrar
                                </button>
                            </td>
                        `;
                        tbody.appendChild(tr);
                    });
                } catch (err) {
                    console.error("Error al cargar el historial:", err);
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
