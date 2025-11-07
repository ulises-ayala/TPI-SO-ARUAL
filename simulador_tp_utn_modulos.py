# simulador_tp_utn_modulos.py
# Simulador de Asignación de Memoria (Particiones Fijas, Best-Fit) y Planificación SRTF
# Alineado con la guía de cátedra y el diagrama de flujo (MÓDULO 1..6).
# Uso:
#   python simulador_tp_utn_modulos.py              # corre con demo embebida
#   python simulador_tp_utn_modulos.py procesos.csv # lee CSV: id,tam,arribo,irrupcion

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import heapq
import csv
import sys
from pathlib import Path

# ================================
# Constantes de Estados
# ================================
EST_NEW = "Nuevo"
EST_READY = "Listo"
EST_RUN  = "Ejecución"
EST_SUSP = "Listo/Suspendido"
EST_TERM = "Terminado"

MAX_PROC = 10
DEGREE_MP = 5  # grado de multiprogramación

# ================================
# MÓDULO 1: INICIALIZACIÓN, CARGA Y CONTROL DE PROCESOS
# - Leer archivo de procesos (id, tamaño, arribo, irrupción)
# - Definir particiones fijas (SO:100K, TG:250K, TM:150K, TP:50K)
# - Inicializar reloj, contadores y colas (Nuevo, Listo, Listo/Suspendido, Ejecución, Terminado)
# ================================

@dataclass
class Proceso:
    pid: str
    tam: int                # tamaño del proceso (K)
    arribo: int             # tiempo de llegada
    irrupcion: int          # CPU total requerida
    restante: int = field(init=False)
    estado: str = field(default=EST_NEW)
    inicio: Optional[int] = None
    fin: Optional[int] = None

    def __post_init__(self):
        self.restante = self.irrupcion

    # Métricas
    def retorno(self) -> Optional[int]:
        return None if self.fin is None else (self.fin - self.arribo)

    def espera(self) -> Optional[int]:
        r = self.retorno()
        return None if r is None else (r - self.irrupcion)

@dataclass
class Particion:
    idp: str
    inicio: int
    tam: int
    pid_asignado: Optional[str] = None

    def libre(self) -> bool:
        return self.pid_asignado is None

    def frag_interna_con(self, proc: Dict[str, Proceso]) -> int:
        if self.pid_asignado is None:
            return 0
        req = proc[self.pid_asignado].tam
        return max(0, self.tam - req)

class GestorMemoria:
    """Particiones fijas con Best-Fit (sin compactar)."""
    def __init__(self):
        # 100K SO (reservado, no asignable)
        # 3 particiones disponibles: 250K (G), 150K (M), 50K (P)
        self.particiones: List[Particion] = [
            Particion("SO",    0,   100, pid_asignado="SO"),
            Particion("G-250", 100, 250, None),
            Particion("M-150", 350, 150, None),
            Particion("P-50",  500,  50, None),
        ]

    def snapshot(self, procesos: Dict[str, Proceso]) -> List[Tuple[str,int,int,Optional[str],int]]:
        out = []
        for p in self.particiones:
            if p.idp == "SO":
                out.append((p.idp, p.inicio, p.tam, p.pid_asignado, 0))
            else:
                out.append((p.idp, p.inicio, p.tam, p.pid_asignado, p.frag_interna_con(procesos)))
        return out

    def best_fit_idx(self, tam_proc: int) -> Optional[int]:
        candidatas = [
            (i, part.tam)
            for i, part in enumerate(self.particiones)
            if part.idp != "SO" and part.libre() and part.tam >= tam_proc
        ]
        if not candidatas:
            return None
        i_mejor, _ = min(candidatas, key=lambda t: t[1])
        return i_mejor

    def asignar(self, pid: str, tam_proc: int) -> bool:
        idx = self.best_fit_idx(tam_proc)
        if idx is None:
            return False
        self.particiones[idx].pid_asignado = pid
        return True

    def liberar(self, pid: str):
        for part in self.particiones:
            if part.pid_asignado == pid:
                part.pid_asignado = None
                return

# ================================
# Planificador SRTF (preemptivo)
# ================================

class PlanificadorSRTF:
    # heap por (restante, tie_arribo, pid)
    def __init__(self):
        self.h: List[Tuple[int,int,str]] = []

    def agregar(self, proc: Proceso):
        heapq.heappush(self.h, (proc.restante, proc.arribo, proc.pid))

    def quitar(self, pid: str):
        self.h = [(r,a,p) for (r,a,p) in self.h if p != pid]
        heapq.heapify(self.h)

    def proximo(self) -> Optional[str]:
        if not self.h:
            return None
        return self.h[0][2]

    def pop(self) -> Optional[str]:
        if not self.h:
            return None
        _,_,pid = heapq.heappop(self.h)
        return pid

    def vacio(self) -> bool:
        return not self.h

# ================================
# Simulador principal
# ================================

class Simulador:
    def __init__(self, procesos: List[Proceso], interactivo: bool = True):
        if len(procesos) > MAX_PROC:
            raise ValueError(f"Máximo {MAX_PROC} procesos.")
        # Reloj de simulación
        self.time = 0
        self.interactivo = interactivo
        # Estructuras principales
        self.proc: Dict[str, Proceso] = {p.pid: p for p in procesos}
        self.mm = GestorMemoria()
        self.sched = PlanificadorSRTF()
        self.running: Optional[str] = None  # pid ejecutando
        # Colas/Conjuntos
        self.admitidos: List[str] = []      # en el sistema (incluye suspendidos)
        self.ready: List[str] = []          # cola “lógica” de listos (aparte del heap)
        self.suspendidos: List[str] = []    # sin partición, admitidos
        self.terminados: List[str] = []

    # ---------- Utilidades de presentación ----------
    def pausa_evento(self, titulo: str):
        if self.interactivo:
            input(f"\n[ENTER] {titulo}")

    def imprimir_estado(self, motivo: str):
        print("\n" + "="*78)
        print(f"[t={self.time}] EVENTO: {motivo}")
        print("-"*78)
        print("CPU:", f"{self.running if self.running else '—(idle)'}")
        print("\nTabla de particiones (id, inicio, tam, pid, frag.int.):")
        for row in self.mm.snapshot(self.proc):
            print(f"  {row[0]:<6}  {row[1]:>4}K  {row[2]:>4}K  "
                  f"{(row[3] if row[3] else '-'):>6}  {row[4]:>4}K")
        print("\nCola de Listos:", self.ready if self.ready else "—")
        print("Cola de Listo/Suspendidos:", self.suspendidos if self.suspendidos else "—")
        print("="*78)

    # ---------- Políticas y reglas de la cátedra ----------
    def puede_admitir_mas(self) -> bool:
        # Grado de multiprogramación: admitir hasta 5 procesos al sistema (Listo o Suspendido).
        return len(self.admitidos) < DEGREE_MP

    # ================================
    # MÓDULO 2: CICLO PRINCIPAL (ADMISIÓN + ASIGNACIÓN DE MEMORIA Best-Fit)
    # ================================
    def intentar_admitir(self, pid: str):
        """Admite desde Nuevo → (Listo o Listo/Suspendido) respetando DEGREE_MP y Best-Fit."""
        p = self.proc[pid]
        if not self.puede_admitir_mas():
            # Se queda en Nuevo (no se incrementa MP)
            return False

        if self.mm.asignar(pid, p.tam):
            # Entra a una partición → Listo
            p.estado = EST_READY
            self.admitidos.append(pid)   # incrementa MP
            self.ready.append(pid)
            self.sched.agregar(p)
            self.imprimir_estado(f"Llegada/Admisión {pid} a Listo (Best-Fit OK)")
            self.pausa_evento("continuar")
            return True
        else:
            # No hay partición → Listo/Suspendido
            p.estado = EST_SUSP
            self.admitidos.append(pid)   # incrementa MP
            self.suspendidos.append(pid)
            self.imprimir_estado(f"Llegada/Admisión {pid} a Listo/Suspendido (sin partición)")
            self.pausa_evento("continuar")
            return True

    def llegada(self):
        """Admite todos los procesos que arriban en t, si hay cupo de multiprogramación."""
        llegados = [p.pid for p in self.proc.values() if p.arribo == self.time and p.estado == EST_NEW]
        for pid in sorted(llegados, key=lambda x: self.proc[x].pid):
            if not self.puede_admitir_mas():
                continue
            self.intentar_admitir(pid)

    # ================================
    # MÓDULO 4: TRATA PROCESOS EN LISTO/SUSPENDIDO
    # - Tras liberarse una partición, intentar ingresar un suspendido
    # ================================
    def traer_suspendido_si_encaja(self) -> Optional[str]:
        if not self.suspendidos:
            return None
        # Elijan el suspendido con MENOR restante (consistente con SRTF)
        candidatos = sorted(self.suspendidos, key=lambda pid: (self.proc[pid].restante, self.proc[pid].arribo))
        for pid in candidatos:
            p = self.proc[pid]
            if self.mm.asignar(pid, p.tam):
                self.suspendidos.remove(pid)
                p.estado = EST_READY
                self.ready.append(pid)
                self.sched.agregar(p)
                self.imprimir_estado(f"Ingreso desde Listo/Suspendido → Listo: {pid}")
                self.pausa_evento("continuar")
                return pid
        return None

    # ================================
    # MÓDULO 3: CPU (Planificación SRTF y ejecución)
    # ================================
    def despachar(self):
        """Selecciona proceso por SRTF y aplica preempción si corresponde."""
        candidato = self.sched.proximo()
        if candidato is None:
            return

        if self.running is None:
            pid = self.sched.pop()
            self.running = pid
            p = self.proc[pid]
            if p.inicio is None:
                p.inicio = self.time
            p.estado = EST_RUN
            self.imprimir_estado(f"Dispatch {pid}")
            self.pausa_evento("continuar")
            return

        # Comparar restante del candidato vs el que corre
        p_run = self.proc[self.running]
        p_cand = self.proc[candidato]
        if p_cand.restante < p_run.restante:
            # Preempción: el que estaba corriendo vuelve a Listo
            p_run.estado = EST_READY
            if p_run.pid not in self.ready:
                self.ready.append(p_run.pid)
            self.sched.agregar(p_run)
            # Candidato toma CPU
            self.sched.quitar(candidato)
            self.running = p_cand.pid
            if p_cand.inicio is None:
                p_cand.inicio = self.time
            p_cand.estado = EST_RUN
            self.imprimir_estado(f"Preempción: entra {p_cand.pid}")
            self.pausa_evento("continuar")

    # ================================
    # MÓDULO 5: TRATA PROCESOS E INFORME PARCIAL (decrementos, impresión por evento)
    # ================================
    def tick_cpu(self):
        """Ejecuta 1 unidad de CPU del proceso en ejecución. Maneja finalización."""
        if self.running is None:
            return
        p = self.proc[self.running]
        # Asegurar que no esté en ready mientras corre
        if self.running in self.ready:
            self.ready.remove(self.running)
        p.restante -= 1
        if p.restante == 0:
            # Finaliza
            p.fin = self.time + 1
            p.estado = EST_TERM
            self.terminados.append(p.pid)
            # Liberar partición
            self.mm.liberar(p.pid)
            finalizado = self.running
            self.running = None
            # Al liberar, intentar traer un suspendido si encaja
            self.traer_suspendido_si_encaja()
            self.imprimir_estado(f"Finaliza {finalizado} (libera partición)")
            self.pausa_evento("continuar")

    def step(self):
        """Un paso de simulación: llegada, planificación, 1 tick de CPU, tratar suspendidos, avanzar reloj."""
        # MÓDULO 2: Admisión + Best-Fit (para procesos que llegan en este tick)
        self.llegada()
        # MÓDULO 3: SRTF (dispatch / posible preempción)
        self.despachar()
        # MÓDULO 5: Ejecutar 1 tick de CPU (y mostrar si hay evento)
        self.tick_cpu()
        # MÓDULO 4: (adicional) Si liberar memoria permitió que entre un suspendido, ya se trató
        #           en traer_suspendido_si_encaja() (llamado dentro de finalización).
        # Avanzar reloj
        self.time += 1

    def todo_terminado(self) -> bool:
        return all(p.estado == EST_TERM for p in self.proc.values())

    # ================================
    # MÓDULO 6: SALIDA FINAL Y ESTADÍSTICAS
    # ================================
    def informe_final(self):
        filas = []
        for p in sorted(self.proc.values(), key=lambda x: x.pid):
            filas.append({
                "pid": p.pid,
                "arribo": p.arribo,
                "tam": p.tam,
                "irrup": p.irrupcion,
                "inicio": p.inicio,
                "fin": p.fin,
                "retorno": p.retorno(),
                "espera": p.espera(),
            })
        prom_retorno = round(sum(f["retorno"] for f in filas) / len(filas), 2)
        prom_espera = round(sum(f["espera"] for f in filas) / len(filas), 2)
        rendimiento = round(len(self.terminados) / self.time, 4) if self.time > 0 else 0.0

        print("\n" + "#"*78)
        print("# INFORME FINAL (MÓDULO 6)")
        print("#"*78)
        for f in filas:
            print(f)
        print("\nPromedios:")
        print("  Tiempo de retorno promedio:", prom_retorno)
        print("  Tiempo de espera promedio :", prom_espera)
        print("Rendimiento (jobs/tick):", rendimiento)
        print("#"*78)

    def run(self):
        """Bucle de simulación paso a paso hasta que todos estén Terminados (no corre de una sola vez)."""
        while not self.todo_terminado():
            self.step()
        self.informe_final()

# ================================
# Carga de procesos desde CSV (id,tam,arribo,irrupcion)
# ================================

def cargar_procesos(desde_csv: Optional[str]) -> List[Proceso]:
    if desde_csv is None:
        # Demo interna si no se especifica CSV (máx 10)
        demo = [
            Proceso("P1", 60,  0, 8),
            Proceso("P2", 120, 1, 4),
            Proceso("P3", 30,  2, 2),
            Proceso("P4", 200, 3, 5),
            Proceso("P5", 140, 4, 3),
            Proceso("P6", 40,  6, 4),
        ]
        return demo[:MAX_PROC]

    path = Path(desde_csv)
    if not path.exists():
        print(f"Archivo no encontrado: {desde_csv}")
        sys.exit(1)
    procs: List[Proceso] = []
    with open(path, newline='', encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            pid = str(row["id"]).strip()
            tam = int(row["tam"])
            arr = int(row["arribo"])
            irr = int(row["irrupcion"])
            procs.append(Proceso(pid, tam, arr, irr))
            if len(procs) == MAX_PROC:
                break
    return procs

# ================================
# Main CLI
# ================================

def main():
    # CSV ejemplo (cabecera obligatoria):
    # id,tam,arribo,irrupcion
    # P1,60,0,8
    # P2,120,1,4
    # P3,30,2,2
    # P4,200,3,5
    # P5,140,4,3
    # P6,40,6,4
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    procesos = cargar_procesos(csv_path)
    procesos.sort(key=lambda p: (p.arribo, p.pid))
    sim = Simulador(procesos, interactivo=True)
    sim.run()

if __name__ == "__main__":
    main()
