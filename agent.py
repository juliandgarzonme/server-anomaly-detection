"""
agent.py — Agente de captura de métricas del servidor
======================================================
Captura las métricas operativas necesarias para el análisis y
las envía al workflow de n8n en intervalos configurables.

Variables capturadas y su correspondencia con el dataset:
  Dataset variable              → Fuente psutil
  ─────────────────────────────────────────────────────
  cpu_usage            (%)      → cpu_percent(interval)
  memory_usage         (%)      → virtual_memory().percent
  network_traffic      (bytes)  → net_io_counters() delta bytes/s
  power_consumption    (W)      → sensors_battery() o estimación por CPU
  num_executed_instructions     → cpu_stats().ctx_switches delta/s (proxy)
  execution_time       (s)      → cpu_times().user + system delta/s
  energy_efficiency    (ratio)  → cpu_usage / max(memory_usage, 1)
  disk                 (%)      → disk_usage('/').percent  [extra]

Uso:
    python agent.py                       # intervalo por defecto: 30s
    python agent.py --interval 10         # cada 10 segundos
    python agent.py --interval 10 --once  # una sola captura y sale
"""

import argparse
import logging
import time
from datetime import datetime, timezone

import psutil
import requests

# ── Configuración ────────────────────────────────────────────────────
WEBHOOK_URL = "http://localhost:5678/webhook/metrics"
INTERVAL    = 30      # segundos entre capturas
TIMEOUT     = 5       # timeout de la petición HTTP en segundos

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────

def _net_bytes_per_second(interval: float) -> dict:
    """Calcula bytes de red enviados y recibidos por segundo."""
    snap1 = psutil.net_io_counters()
    time.sleep(interval)
    snap2 = psutil.net_io_counters()

    sent  = max(snap2.bytes_sent - snap1.bytes_sent, 0)
    recv  = max(snap2.bytes_recv - snap1.bytes_recv, 0)
    return {
        "bytes_sent_per_s": sent,
        "bytes_recv_per_s": recv,
        "total_bytes_per_s": sent + recv,
    }


def _disk_io_per_second(interval: float) -> dict:
    """Calcula operaciones de disco por segundo. Devuelve 0 si no está disponible."""
    try:
        snap1 = psutil.disk_io_counters()
        if snap1 is None:
            return {"disk_read_bytes_per_s": 0, "disk_write_bytes_per_s": 0}
        time.sleep(interval)
        snap2 = psutil.disk_io_counters()
        return {
            "disk_read_bytes_per_s":  max(snap2.read_bytes  - snap1.read_bytes,  0),
            "disk_write_bytes_per_s": max(snap2.write_bytes - snap1.write_bytes, 0),
        }
    except Exception:
        return {"disk_read_bytes_per_s": 0, "disk_write_bytes_per_s": 0}


def _cpu_stats_delta(interval: float) -> dict:
    """
    Calcula deltas de estadísticas de CPU por segundo.
    ctx_switches es el proxy más cercano a 'num_executed_instructions'
    disponible en psutil sin acceso a registros de hardware.
    """
    snap1 = psutil.cpu_stats()
    time.sleep(interval)
    snap2 = psutil.cpu_stats()
    return {
        "ctx_switches_per_s":    max(snap2.ctx_switches    - snap1.ctx_switches,    0),
        "interrupts_per_s":      max(snap2.interrupts      - snap1.interrupts,      0),
        "soft_interrupts_per_s": max(snap2.soft_interrupts - snap1.soft_interrupts, 0),
    }


def _cpu_times_delta(interval: float) -> dict:
    """
    Calcula tiempo de CPU activo (user + system) en el intervalo.
    Equivalente a execution_time del dataset.
    """
    snap1 = psutil.cpu_times()
    time.sleep(interval)
    snap2 = psutil.cpu_times()
    user_delta   = max(snap2.user   - snap1.user,   0.0)
    system_delta = max(snap2.system - snap1.system, 0.0)
    return {
        "cpu_time_user_s":   round(user_delta,   4),
        "cpu_time_system_s": round(system_delta, 4),
        "execution_time":    round(user_delta + system_delta, 4),
    }


def _power_consumption() -> float:
    """
    Intenta obtener consumo de energía en vatios.
    - En laptops: usa sensors_battery() si está disponible.
    - En servidores Linux: lee /sys/class/power_supply.
    - Fallback: estimación proporcional al uso de CPU (útil para desarrollo).
    """
    # Intento 1: batería (laptop)
    try:
        battery = psutil.sensors_battery()
        if battery and hasattr(battery, 'power_plugged'):
            # psutil no da vatios directamente, usamos estimación
            pass
    except Exception:
        pass

    # Intento 2: sensor de hardware en Linux
    try:
        with open("/sys/class/power_supply/BAT0/power_now") as f:
            microwatts = int(f.read().strip())
            return round(microwatts / 1_000_000, 2)  # convertir a vatios
    except Exception:
        pass

    # Fallback: estimación basada en CPU
    # Un servidor típico consume entre 50W (idle) y 500W (carga completa)
    cpu = psutil.cpu_percent(interval=0)
    estimated = round(50 + (cpu / 100) * 450, 2)
    return estimated


# ── Captura principal ────────────────────────────────────────────────

def capture_metrics(sample_interval: float = 1.0) -> dict:
    """
    Captura todas las métricas en una sola muestra.

    Las métricas basadas en deltas (red, disco, CPU stats, CPU times)
    se miden durante `sample_interval` segundos para calcular tasas/segundo.
    Las métricas instantáneas (CPU %, memoria, disco) se toman al final.

    Args:
        sample_interval: duración en segundos del período de muestreo
                         para métricas de delta. Recomendado: 1.0s.
    Returns:
        dict con todas las métricas listas para enviar a n8n.
    """
    # ── Métricas de delta (requieren esperar interval segundos) ───────
    # Se capturan en paralelo usando snapshots inicial/final
    net_snap1  = psutil.net_io_counters()
    stat_snap1 = psutil.cpu_stats()
    time_snap1 = psutil.cpu_times()

    try:
        disk_snap1 = psutil.disk_io_counters()
    except Exception:
        disk_snap1 = None

    time.sleep(sample_interval)

    net_snap2  = psutil.net_io_counters()
    stat_snap2 = psutil.cpu_stats()
    time_snap2 = psutil.cpu_times()

    try:
        disk_snap2 = psutil.disk_io_counters()
    except Exception:
        disk_snap2 = None

    # ── Calcular deltas ───────────────────────────────────────────────
    bytes_sent = max(net_snap2.bytes_sent - net_snap1.bytes_sent, 0)
    bytes_recv = max(net_snap2.bytes_recv - net_snap1.bytes_recv, 0)
    network_traffic = bytes_sent + bytes_recv     # total bytes/s

    ctx_switches = max(stat_snap2.ctx_switches - stat_snap1.ctx_switches, 0)
    interrupts   = max(stat_snap2.interrupts   - stat_snap1.interrupts,   0)

    cpu_user_delta   = max(time_snap2.user   - time_snap1.user,   0.0)
    cpu_system_delta = max(time_snap2.system - time_snap1.system, 0.0)
    execution_time   = round(cpu_user_delta + cpu_system_delta, 4)

    if disk_snap1 and disk_snap2:
        disk_read_ps  = max(disk_snap2.read_bytes  - disk_snap1.read_bytes,  0)
        disk_write_ps = max(disk_snap2.write_bytes - disk_snap1.write_bytes, 0)
    else:
        disk_read_ps  = 0
        disk_write_ps = 0

    # ── Métricas instantáneas (al final del intervalo) ────────────────
    cpu_usage    = psutil.cpu_percent(interval=None)   # sin bloqueo adicional
    vm           = psutil.virtual_memory()
    memory_usage = vm.percent
    disk_usage   = psutil.disk_usage('/').percent
    power        = _power_consumption()

    # energy_efficiency: ratio CPU activo / memoria usada
    # Valores altos → buena eficiencia (mucho CPU con poca memoria)
    # Valores bajos → baja eficiencia (poca CPU con mucha memoria)
    energy_efficiency = round(cpu_usage / max(memory_usage, 1), 4)

    # ── Construir payload ─────────────────────────────────────────────
    return {
        # ── Variables principales del dataset ─────────────────────────
        "cpu_usage":                  round(cpu_usage, 2),
        "memory_usage":               round(memory_usage, 2),
        "network_traffic":            network_traffic,
        "power_consumption":          power,
        "num_executed_instructions":  ctx_switches,    # proxy: ctx_switches/s
        "execution_time":             execution_time,
        "energy_efficiency":          energy_efficiency,

        # ── Variables de apoyo (disco + red desglosada) ───────────────
        "disk_usage_pct":             round(disk_usage, 2),
        "bytes_sent_per_s":           bytes_sent,
        "bytes_recv_per_s":           bytes_recv,
        "disk_read_bytes_per_s":      disk_read_ps,
        "disk_write_bytes_per_s":     disk_write_ps,
        "interrupts_per_s":           interrupts,

        # ── Variables de contexto del sistema ─────────────────────────
        "cpu_freq_mhz":    round(psutil.cpu_freq().current, 1) if psutil.cpu_freq() else 0,
        "cpu_count":       psutil.cpu_count(logical=True),
        "swap_usage_pct":  psutil.swap_memory().percent,
        "load_avg_1m":     round(psutil.getloadavg()[0], 2),

        # ── Metadatos de captura ──────────────────────────────────────
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "sample_interval": sample_interval,
    }


# ── Envío a n8n ──────────────────────────────────────────────────────

def send_metrics(metrics: dict, url: str = WEBHOOK_URL) -> bool:
    """
    Envía el payload a n8n via POST.
    Retorna True si fue exitoso, False si hubo error.
    """
    try:
        response = requests.post(url, json=metrics, timeout=TIMEOUT)
        response.raise_for_status()
        log.info(f"✅ Métricas enviadas — status {response.status_code} "
                 f"| cpu={metrics['cpu_usage']}% "
                 f"| mem={metrics['memory_usage']}% "
                 f"| net={metrics['network_traffic']} bytes/s")
        return True
    except requests.exceptions.ConnectionError:
        log.error(f"❌ No se pudo conectar con {url}")
    except requests.exceptions.Timeout:
        log.error(f"❌ Timeout al conectar con {url}")
    except requests.exceptions.HTTPError as e:
        log.error(f"❌ HTTP error: {e}")
    except Exception as e:
        log.error(f"❌ Error inesperado: {e}")
    return False


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Agente de captura de métricas del servidor")
    parser.add_argument("--url",      default=WEBHOOK_URL,  help="URL del webhook de n8n")
    parser.add_argument("--interval", type=int, default=INTERVAL, help="Segundos entre capturas")
    parser.add_argument("--once",     action="store_true",  help="Capturar una sola vez y salir")
    args = parser.parse_args()

    log.info(f"Agente iniciado — enviando a: {args.url}")
    log.info(f"Intervalo de captura: {args.interval}s")

    if args.once:
        metrics = capture_metrics(sample_interval=1.0)
        send_metrics(metrics, args.url)
        return

    while True:
        try:
            metrics = capture_metrics(sample_interval=1.0)
            send_metrics(metrics, args.url)
            # El intervalo total incluye el 1s de muestreo interno
            remaining = max(args.interval - 1, 0)
            if remaining > 0:
                time.sleep(remaining)
        except KeyboardInterrupt:
            log.info("Agente detenido por el usuario.")
            break
        except Exception as e:
            log.error(f"Error en el ciclo principal: {e}")
            time.sleep(5)   # esperar antes de reintentar


if __name__ == "__main__":
    main()
