import numpy as np
import matplotlib.pyplot as plt

# Vetor de decisão otimizado: 1 = bomba ligada, 0 = bomba desligada
estado_otimo = np.array([
    1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1,
    1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1
])

def vetor_x(estado):
    inicios, duracoes = [], []
    ligada = False
    for t in range(len(estado)):
        if estado[t] == 1:
            if not ligada:
                ligada = True
                inicios.append(t)
                duracoes.append(1)
            else:
                duracoes[-1] += 1
        else:
            ligada = False
    return inicios, duracoes

# Funções de consumo
def Q_VC_min(t):
    return np.clip(
        1.19333e-7 * t ** 7 - 6.54846e-5 * t**  6 +
        4.1432e-3 * t**  5 - 0.100585 * t ** 4 +
        1.05575 * t ** 3 - 3.85966 * t ** 2 -
        1.32657 * t + 75.393,
        0, None)

def Q_VC_max(t):
    return np.clip(
        -1.19333e-7 * t**  7 - 4.90754e-5 * t**  6 +
        3.733e-3 * t ** 5 - 0.09621 * t ** 4 +
        1.03965 * t**  3 - 3.8645 * t**  2 -
        1.0124 * t + 75.393,
        0, None )
def Q_R(t):
    return -0.004 * t ** 3 + 0.09 * t ** 2 + 0.1335 * t + 20

# Parâmetros
horas = 24
A = 185
h0 = 4
h_seg_min, h_seg_max = 2, 7
h_min, h_max = 0, 9
Qp_max = 100

f = 0.02
L1, L2 = 2500, 5000
rho, g = 1000, 9.81
d = 0.3

ef = 0.65
a1, a2 = 260, -0.002

tarifas = np.array([
    0.0713, 0.0713, 0.0651, 0.0651, 0.0593, 0.0593,
    0.0778, 0.0778, 0.0851, 0.0851, 0.0923, 0.0923,
    0.0968, 0.0968, 0.10094, 0.10094, 0.10132, 0.10132,
    0.10230, 0.10230, 0.10189, 0.10189, 0.10132, 0.10132
])

# Solver de Qp
def solve_Qp(h_prev, qvc, qr):
    def energy_residual(Qp):
        qs_p = Qp / 3600.0
        qs_pr = max(Qp - qr, 0) / 3600.0
        qs_r = qr / 3600.0
        hf1 = (32 * f * L1) * (qs_r ** 2) / (d ** 5 * g * np.pi ** 2)
        hf2 = (32 * f * L2) * (qs_pr ** 2) / (d ** 5 * g * np.pi ** 2)
        H_p = a1 + a2 * Qp ** 2
        H_agua = h_prev + (Qp - qr - qvc) / A
        return H_p - hf1 - hf2 - H_agua

    low, high = 0.0, Qp_max
    f_low, f_high = energy_residual(low), energy_residual(high)
    if f_low * f_high > 0:
        return Qp_max
    for _ in range(30):
        mid = 0.5 * (low + high)
        f_mid = energy_residual(mid)
        if f_low * f_mid <= 0:
            high, f_high = mid, f_mid
        else:
            low, f_low = mid, f_mid
    return 0.5 * (low + high)

# Simulação com controle de limites
def simula(Q_VC_fun, penalizar=True, forcar_limites=True):
    nivel = np.zeros(horas + 1)
    nivel[0] = h0
    estado = estado_otimo.copy()
    Qp_hist = np.zeros(horas)
    energia = np.zeros(horas)
    custo_e = np.zeros(horas)
    penal = np.zeros(horas)

    for t in range(horas):
        h = nivel[t]
        qvc = Q_VC_fun(t)
        qr = Q_R(t)
        qout = qvc + qr

        if estado[t] == 1:
            Qp_opt = solve_Qp(h, qvc, qr)
            desejado = qout + 20

            if forcar_limites:
                vol_max = (h_seg_max - h) * A
                vol_min = (h_seg_min - h) * A
                Qp_sup = vol_max + qout
                Qp_inf = max(vol_min + qout, 0)
                Qp = min(max(Qp_opt, Qp_inf), Qp_sup, desejado)
            else:
                Qp = min(Qp_opt, desejado, (h_max - h) * A + qout)
        else:
            Qp = 0.0

        Qp_hist[t] = Qp
        nivel[t + 1] = h + (Qp - qout) / A

        if penalizar:
            if nivel[t + 1] < h_seg_min or nivel[t + 1] > h_seg_max:
                penal[t] = penal[t - 1] + 5 if t > 0 else 5

        if Qp > 0:
            qs = Qp / 3600.0
            hf_tot = (32 * f * (L1 + L2)) * (qs ** 2) / (d ** 5 * g * np.pi ** 2)
            Hp = (a1 + a2 * Qp ** 2) + hf_tot + h
            P = rho * g * qs * Hp / ef
            Eh = P / 3600.0
            energia[t] = Eh
            custo_e[t] = Eh * tarifas[t]

    return nivel, Qp_hist, energia, custo_e, penal

# Verificador de limites
def verifica_niveis(nome, nivel):
    abaixo = np.sum(nivel < h_seg_min)
    acima = np.sum(nivel > h_seg_max)
    if abaixo > 0 or acima > 0:
        print(f"[{nome}] Atenção: {abaixo} abaixo de {h_seg_min}m, {acima} acima de {h_seg_max}m.")
    else:
        print(f"[{nome}] OK: Todos os níveis dentro dos limites seguros.")

# Simulações
nivel_min, Qp_min, energia_min, custo_min, _ = simula(Q_VC_min, penalizar=False, forcar_limites=True)
nivel_max, Qp_max_, energia_max, custo_max, _ = simula(Q_VC_max, penalizar=False, forcar_limites=True)

# Tarefa 4.2 – Penalização sem forçar limites seguros (mas respeitando limites físicos)
nivel_otimo, Qp_otimo, energia_otimo, custo_otimo, penal_otimo = simula(
    lambda t: (Q_VC_min(t) + Q_VC_max(t)) / 2,
    penalizar=True,
    forcar_limites=False
)

# Verificação de segurança dos níveis
verifica_niveis("Tarefa 4.1 – QVC Mín", nivel_min)
verifica_niveis("Tarefa 4.1 – QVC Máx", nivel_max)

# Mostrar vetor x (inícios + durações)
inicios, duracoes = vetor_x(estado_otimo)
print(f"x = {inicios} + {duracoes}")

# Resultados e gráficos
cenarios_resultados = [
    ("Tarefa 4.1 – QVC Mín", energia_min, custo_min, np.zeros_like(custo_min), nivel_min),
    ("Tarefa 4.1 – QVC Máx", energia_max, custo_max, np.zeros_like(custo_max), nivel_max),
    ("Tarefa 4.2 – Otimização", energia_otimo, custo_otimo, penal_otimo, nivel_otimo)
]

for nome, energia, custo, penal, nivel in cenarios_resultados:
    print(f"\n{nome}:")
    print(f"Consumo energético total: {energia.sum():.2f} kWh")
    print(f"Custo energia: {custo.sum():.2f} €")
    print(f"Custo penalizações: {penal.sum():.2f} €")
    print(f"Custo total: {(custo + penal).sum():.2f} €")

    # Gráficos
    t = np.arange(horas)
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axs[0].step(t, estado_otimo, where='mid', label='Estado Bomba', color='gray')
    axs[0].plot(t, nivel[:-1], label='Nível Reservatório [m]', color='royalblue')
    axs[0].axhline(h_seg_min, color='red', ls='--', label='Limite Inferior Seguro')
    axs[0].axhline(h_seg_max, color='red', ls='--', label='Limite Superior Seguro')
    axs[0].set_ylabel('Nível [m] e Estado da Bomba')
    axs[0].legend()
    axs[0].grid(True, ls='--')

    axs[1].plot(t, energia, label='Energia [kWh]', color='green')
    axs[1].plot(t, custo, label='Custo [€]', color='cyan')
    if penal.sum() > 0:
        axs[1].plot(t, penal, label='Penalização [€]', color='deeppink')
    axs[1].set_xlabel('Hora')
    axs[1].set_ylabel('Energia, Custo e Penalização')
    axs[1].legend()
    axs[1].grid(True, ls='--')

    plt.suptitle(nome)
    plt.tight_layout()
    plt.show()
