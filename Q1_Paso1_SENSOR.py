import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pathlib import Path

# =========================
#   Configuración básica
# =========================
USE_FLOAT64 = False  # si ves inestabilidad numérica, pon True
device = (
    torch.device("cuda:0") if torch.cuda.is_available()
    else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else torch.device("cpu")
)
dtype = torch.float64 if USE_FLOAT64 else torch.float32
torch.set_default_dtype(dtype)

print("Device:", device, "| dtype:", dtype)

SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)

T_INF = 293.15  # [K] ambiente

# Pesos
W_DATA = 1.0
W_PHYS = 1.0
W_IC   = 0.1
W_REG  = 1e-4

# Red
HIDDEN = 64
LAYERS = 4

# Entrenamiento
EPOCHS = int(input("EPOCAS = "))
LR     = 2e-3

# Potencia (constante por defecto)
Q1 = 1e8          # [W/m^3]
R  = 0.1e-3       # [m]
h  = 10e-3        # [m]
V1 = np.pi * R**2 * h
P1_const = Q1 * V1  # [W]

# =========================
#   Carga de datos
# =========================
def load_from_mat(path: Path):
    D = loadmat(path.as_posix())
    D1 = D["DATA"][0, 0]

    t       = np.asarray(D1["t"],        dtype=float).ravel()
    TFeNi42 = np.asarray(D1["T_FeNi42"], dtype=float).ravel()
    TCem    = np.asarray(D1["T_Cem"],    dtype=float).ravel()
    TVid    = np.asarray(D1["T_Vid"],    dtype=float).ravel()
    TAlum   = np.asarray(D1["T_Alum"],   dtype=float).ravel()
    TSensor = np.asarray(D1["T_Sensor"], dtype=float).ravel()  
    
      #RUIDO GAUSSIANO AÑADIR
    sigma2 = float(input("Ruido gaussiano sigma2 = "))
    sigma = np.sqrt(sigma2)
  
    TFeNi42 = TFeNi42 + np.random.normal(0.0, sigma, size=TFeNi42.shape)
    TCem    = TCem    + np.random.normal(0.0, sigma, size=TCem.shape)
    TVid    = TVid    + np.random.normal(0.0, sigma, size=TVid.shape)
    TAlum   = TAlum   + np.random.normal(0.0, sigma, size=TAlum.shape)
    TSensor = TSensor + np.random.normal(0.0, sigma, size=TSensor.shape)

    # Potencia: si quieres leer pdis.mat, sustituye aquí.
    # Por defecto: constante.
    P1 = np.full_like(t, P1_const, dtype=float)

    return t, TFeNi42, TCem, TVid, TAlum, TSensor , P1

mat_path = Path("DATA.mat")
t, TFeNi42, TCem, TVid, TAlum, TSensor, P1 = load_from_mat(mat_path)

# =========================
#   Normalización estilo
#   z = (T - Tinf) / dT
# =========================
def robust_dT(*arrs, qlo=0.05, qhi=0.95):
    cat = np.concatenate([np.asarray(a, float).ravel() for a in arrs])
    lo = float(np.quantile(cat, qlo))
    hi = float(np.quantile(cat, qhi))
    return max(hi - lo, 1e-3)

dT_val = robust_dT(TFeNi42, TCem, TVid, TAlum, TSensor)
Tinf_val = T_INF

Z_data = np.stack([
    (TFeNi42 - Tinf_val) / dT_val,
    (TCem    - Tinf_val) / dT_val,
    (TVid    - Tinf_val) / dT_val,
    (TAlum   - Tinf_val) / dT_val,
    (TSensor - Tinf_val) / dT_val,
], axis=1)  # (N,5)

# Normalización de tiempo a [0,1]
t = np.asarray(t, float).ravel()
t_min, t_max = float(t.min()), float(t.max())
t01 = (t - t_min) / max(t_max - t_min, 1e-12)

# Tensores
t_t01 = torch.tensor(t01,    device=device, dtype=dtype).view(-1, 1)  # input NN
t_t   = torch.tensor(t,      device=device, dtype=dtype).view(-1)     # (N,)
Z_t   = torch.tensor(Z_data, device=device, dtype=dtype)              # (N,5)

P1_t   = torch.tensor(np.asarray(P1, float).ravel(), device=device, dtype=dtype).view(-1)  # (N,)
Tinf_t = torch.tensor([Tinf_val], device=device, dtype=dtype)  # (1,)
dT_t   = torch.tensor([dT_val],   device=device, dtype=dtype)  # (1,)

dt_full = (t_t[1:] - t_t[:-1]).clamp_min(torch.tensor(1e-12, device=device, dtype=dtype))  # (N-1,)
is_uniform = bool(torch.max(torch.abs(dt_full - dt_full[0])) < 1e-12)

# =========================
#   Red neuronal (salida libre en z)
# =========================
class MLP(nn.Module):
    def __init__(self, in_dim=1, out_dim=5, hidden=64, layers=4):
        super().__init__()
        mods = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            mods += [nn.Linear(hidden, hidden), nn.Tanh()]
        mods += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*mods)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x01):
        return self.net(x01)

# =========================
#   Parámetros SIN bounds (log-parametrización)
# =========================
class RCParamsNoBounds(nn.Module):
    """
    Parámetros libres sin cotas superiores/inferiores explícitas.
    Para mantener positividad física: param = exp(raw_log_param).
    """
    def __init__(self, init_C=1.0, init_R=10.0):
        super().__init__()
        eps = 1e-12

        # logs iniciales (puedes ajustar init_C/init_R si quieres otro arranque)
        lc = float(np.log(max(init_C, eps)))
        lr = float(np.log(max(init_R, eps)))

        # Capacidades (logs)
        self.raw_log_CFeNi42 = nn.Parameter(torch.tensor(lc, dtype=dtype, device=device))
        self.raw_log_CCem    = nn.Parameter(torch.tensor(lc, dtype=dtype, device=device))
        self.raw_log_CVid    = nn.Parameter(torch.tensor(lc, dtype=dtype, device=device))
        self.raw_log_CAlm    = nn.Parameter(torch.tensor(lc, dtype=dtype, device=device))
        self.raw_log_CSen    = nn.Parameter(torch.tensor(lc, dtype=dtype, device=device))

        # Resistencias (logs)
        self.raw_log_RFeVid  = nn.Parameter(torch.tensor(lr, dtype=dtype, device=device))
        self.raw_log_RFeCem  = nn.Parameter(torch.tensor(lr, dtype=dtype, device=device))
        self.raw_log_RFe_a   = nn.Parameter(torch.tensor(lr, dtype=dtype, device=device))
        self.raw_log_RVidCem = nn.Parameter(torch.tensor(lr, dtype=dtype, device=device))
        self.raw_log_RVid_a  = nn.Parameter(torch.tensor(lr, dtype=dtype, device=device))
        self.raw_log_RCemAl  = nn.Parameter(torch.tensor(lr, dtype=dtype, device=device))
        self.raw_log_RAl_a   = nn.Parameter(torch.tensor(lr, dtype=dtype, device=device))
        self.raw_log_RFeSen  = nn.Parameter(torch.tensor(lr, dtype=dtype, device=device))
        self.raw_log_RVidSen  = nn.Parameter(torch.tensor(lr, dtype=dtype, device=device))

    def physical(self):
        # Capacidades (J/K)
        CFeNi42 = torch.exp(self.raw_log_CFeNi42)
        CCem    = torch.exp(self.raw_log_CCem)
        CVid    = torch.exp(self.raw_log_CVid)
        CAlm    = torch.exp(self.raw_log_CAlm)
        CSen    = torch.exp(self.raw_log_CSen)

        # Resistencias (K/W)
        RFeVid  = torch.exp(self.raw_log_RFeVid)
        RFeCem  = torch.exp(self.raw_log_RFeCem)
        RFe_a   = torch.exp(self.raw_log_RFe_a)
        RVidCem = torch.exp(self.raw_log_RVidCem)
        RVid_a  = torch.exp(self.raw_log_RVid_a)
        RCemAl  = torch.exp(self.raw_log_RCemAl)
        RAl_a   = torch.exp(self.raw_log_RAl_a)

        RFeSen  = torch.exp(self.raw_log_RFeSen)
        RVidSen = torch.exp(self.raw_log_RVidSen)

        return CFeNi42, CCem, CVid, CAlm, CSen , RFeVid, RFeCem, RFe_a, RVidCem, RVid_a, RCemAl, RAl_a, RFeSen, RVidSen

    def reg(self):
        # L2 suave en el espacio log (evita explosión numérica sin imponer bounds)
        logs = torch.stack([
            self.raw_log_CFeNi42, self.raw_log_CCem, self.raw_log_CVid, self.raw_log_CAlm, self.raw_log_CSen , 
            self.raw_log_RFeVid, self.raw_log_RFeCem, self.raw_log_RFe_a, self.raw_log_RVidCem,
            self.raw_log_RVid_a, self.raw_log_RCemAl, self.raw_log_RAl_a, self.raw_log_RFeSen,
            self.raw_log_RVidSen
            
        ])
        return (logs.pow(2)).mean()

# =========================
#   Dinámica y discreción exacta (matrix_exp) para 5 nodos
# =========================
def cont_mats_4(CFe, CCem, CVid, CAl, CSen ,
                GFeVid, GFeCem, GFe_a, GVidCem, GVid_a, GCemAl, GAl_a, GFeSen, GVidSen ):
    # Estado X = [TFe, TCem, TVid, TAl, TSen]
    A = torch.zeros(5, 5, device=device, dtype=dtype)
    B = torch.zeros(5, 1, device=device, dtype=dtype)  # P entra en Fe
    e = torch.zeros(5, 1, device=device, dtype=dtype)  # coef * Tinf

    # Fe
    A[0, 0] = -(GFeVid + GFeCem + GFe_a + GFeSen) / CFe
    A[0, 1] =  (GFeCem) / CFe
    A[0, 2] =  (GFeVid) / CFe
    A[0, 4] =  (GFeSen) / CFe
    B[0, 0] =  1.0 / CFe
    e[0, 0] =  (GFe_a) / CFe

    # Cem
    A[1, 0] =  (GFeCem) / CCem
    A[1, 1] = -(GCemAl + GVidCem + GFeCem) / CCem
    A[1, 2] =  (GVidCem) / CCem
    A[1, 3] =  (GCemAl) / CCem

    # Vid
    A[2, 0] =  (GFeVid) / CVid
    A[2, 1] =  (GVidCem) / CVid
    A[2, 2] = -(GFeVid + GVidCem + GVid_a + GVidSen) / CVid
    A[2, 4] =  (GVidSen) / CVid
    e[2, 0] =  (GVid_a) / CVid

    # Al
    A[3, 1] =  (GCemAl) / CAl
    A[3, 3] = -(GCemAl + GAl_a) / CAl
    e[3, 0] =  (GAl_a) / CAl

    # Sensor
    A[4, 0] =  (GFeSen) / CSen
    A[4, 2] =  (GVidSen) / CSen
    A[4, 4] = -(GFeSen + GVidSen) / CSen

    return A, B, e

# def discrete_step_uniform_4(A, B, e, dt):
#     # augment: [X; u; Tinf] -> tamaño 6
#     M = torch.zeros(6, 6, device=device, dtype=dtype)
#     M[:4, :4] = A
#     M[:4, 4:5] = B
#     M[:4, 5:6] = e
#     E = torch.linalg.matrix_exp(M * dt)
#     Phi = E[:4, :4]
#     Gu  = E[:4, 4:5]
#     Ge  = E[:4, 5:6]
#     return Phi, Gu, Ge

def discrete_step_uniform_4(A, B, e, dt):
    # augment: [X; u; Tinf] -> tamaño 7
    M = torch.zeros(7, 7, device=device, dtype=dtype)
    M[:5, :5] = A
    M[:5, 5:6] = B
    M[:5, 6:7] = e
    E = torch.linalg.matrix_exp(M * dt)
    Phi = E[:5, :5]
    Gu  = E[:5, 5:6]
    Ge  = E[:5, 6:7]
    return Phi, Gu, Ge

# def discrete_step_nonuniform_4(A, B, e, dt_full_vec):
#     M = torch.zeros(6, 6, device=device, dtype=dtype)
#     M[:4, :4] = A
#     M[:4, 4:5] = B
#     M[:4, 5:6] = e
#     dtv = dt_full_vec.view(-1, 1, 1)
#     E = torch.linalg.matrix_exp(M.unsqueeze(0) * dtv)  # (N-1,5,5)
#     Phi = E[:, :4, :4]
#     Gu  = E[:, :4, 4:5]
#     Ge  = E[:, :4, 5:6]
#     return Phi, Gu, Ge

def discrete_step_nonuniform_4(A, B, e, dt_full_vec):
    M = torch.zeros(7, 7, device=device, dtype=dtype)
    M[:5, :5] = A
    M[:5, 5:6] = B
    M[:5, 6:7] = e
    dtv = dt_full_vec.view(-1, 1, 1)
    E = torch.linalg.matrix_exp(M.unsqueeze(0) * dtv)  # (N-1,6,6)
    Phi = E[:, :5, :5]
    Gu  = E[:, :5, 5:6]
    Ge  = E[:, :5, 6:7]
    return Phi, Gu, Ge


# =========================
#   Instancias
# =========================
pinn = MLP(hidden=HIDDEN, layers=LAYERS, out_dim=5).to(device)
pars = RCParamsNoBounds(init_C=1.0, init_R=10.0).to(device)

# =========================
#   Pérdida
# =========================
def pinn_loss():
    # 1) Predicción en z
    Z_hat = pinn(t_t01)                # (N,5)
    X_hat = Tinf_t + dT_t * Z_hat      # (N,5) Kelvin

    # 2) Datos en z
    L_data = F.mse_loss(Z_hat, Z_t)

    # 3) Parámetros físicos
    CFe, CCem, CVid, CAl, CSen, RFeVid, RFeCem, RFe_a, RVidCem, RVid_a, RCemAl, RAl_a , RFeSen, RVidSen = pars.physical()

    GFeVid  = 1.0 / RFeVid
    GFeCem  = 1.0 / RFeCem
    GFe_a   = 1.0 / RFe_a
    GVidCem = 1.0 / RVidCem
    GVid_a  = 1.0 / RVid_a
    GCemAl  = 1.0 / RCemAl
    GAl_a   = 1.0 / RAl_a

    GFeSen   = 1.0 / RFeSen   
    GVidSen   = 1.0 / RVidSen
    A, B, e = cont_mats_4(CFe, CCem, CVid, CAl, CSen, 
                          GFeVid, GFeCem, GFe_a, GVidCem, GVid_a, GCemAl, GAl_a, GFeSen, GVidSen )

    # 4) Residuo físico discreto exacto
    Xk   = X_hat[:-1, :]                  # (N-1,5)
    Xkp1 = X_hat[1:, :]                   # (N-1,5)
    Pk   = P1_t[:-1].view(-1, 1)          # (N-1,1)

    if is_uniform:
        Phi, Gu, Ge = discrete_step_uniform_4(A, B, e, dt_full[0])
        X_model = Xk @ Phi.T + Pk @ Gu.T + (Tinf_t.view(1, 1) * Ge.T)
    else:
        Phi_all, Gu_all, Ge_all = discrete_step_nonuniform_4(A, B, e, dt_full)
        X_model = (
            torch.bmm(Xk.unsqueeze(1), Phi_all.transpose(1, 2)).squeeze(1)
            + (Pk.squeeze(1)).unsqueeze(1) * Gu_all.squeeze(-1)
            + Tinf_t * Ge_all.squeeze(-1)
        )

    Rz = (Xkp1 - X_model) / dT_t  # residuo en z
    L_phys = (Rz.pow(2)).mean()

    # 5) IC en z
    L_ic = F.mse_loss(Z_hat[0:1, :], Z_t[0:1, :])

    # 6) regularización suave (logs)
    L_reg = W_REG * pars.reg()

    loss = W_DATA * L_data + W_PHYS * L_phys + W_IC * L_ic + L_reg
    aux  = {"L_data": float(L_data.detach()), "L_phys": float(L_phys.detach()), "L_ic": float(L_ic.detach())}
    return loss, aux, X_hat, Z_hat

# =========================
#   Entrenamiento
# =========================
opt   = torch.optim.Adam(list(pinn.parameters()) + list(pars.parameters()), lr=LR)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.9, patience=1000)

best = (1e30, None)

loss_hist = []
data_hist = []
ode_hist  = []
ic_hist   = []

for e in range(1, EPOCHS + 1):
    opt.zero_grad(set_to_none=True)
    loss, aux, X_hat, Z_hat = pinn_loss()

    loss_hist.append(loss.item())
    data_hist.append(aux["L_data"])
    ode_hist.append(aux["L_phys"])
    ic_hist.append(aux["L_ic"])


    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(pinn.parameters()) + list(pars.parameters()), max_norm=5.0)
    opt.step()
    sched.step(float(loss.detach().cpu()))

    if e == 1 or e % 500 == 0:
        CFe_p, CCem_p, CVid_p, CAl_p, CSen_p,  RFeVid_p, RFeCem_p, RFe_a_p, RVidCem_p, RVid_a_p, RCemAl_p, RAl_a_p, RFeSen_p, RVidSen_p = [
            p.item() for p in pars.physical()
        ]
        print(f"[{e:05d}] loss={loss.item():.3e}  data={aux['L_data']:.3e}  ode={aux['L_phys']:.3e}  ic={aux['L_ic']:.3e}")
        print(
            f"  phys    ≈ C=[{CFe_p:.3g},{CCem_p:.3g},{CVid_p:.3g},{CAl_p:.3g}, {CSen_p:.3g}]  "
            f"R=[{RFeVid_p:.3g},{RFeCem_p:.3g},{RFe_a_p:.3g},{RVidCem_p:.3g},{RVid_a_p:.3g},{RCemAl_p:.3g},{RAl_a_p:.3g},{RFeSen_p:.3g}, {RVidSen_p:.3g} , ]"
        )

    if loss.item() < best[0]:
        best = (loss.item(), [p.detach().cpu().clone() for p in list(pinn.parameters()) + list(pars.parameters())])

# Restaura mejores pesos
with torch.no_grad():
    params = list(pinn.parameters()) + list(pars.parameters())
    for p, pb in zip(params, best[1]):
        p.copy_(pb.to(device))

# =========================
#   Resultados + plots
# =========================
with torch.no_grad():
    _, _, X_hat, Z_hat = pinn_loss()

tt = t_t.detach().cpu().numpy().ravel()
X_np = X_hat.detach().cpu().numpy()

TFe_hat, TCem_hat, TVid_hat, TAl_hat, TSen_hat = X_np[:, 0], X_np[:, 1], X_np[:, 2], X_np[:, 3], X_np[:, 4]

print("\n=== PARÁMETROS IDENTIFICADOS ===")
CFe, CCem, CVid, CAl, CSen, RFeVid, RFeCem, RFe_a, RVidCem, RVid_a, RCemAl, RAl_a, RFeSen, RVidSen = [p.item() for p in pars.physical()]
print(f"CFeNi42={CFe:.6g} J/K,  CCem={CCem:.6g} J/K,  CVid={CVid:.6g} J/K,  CAl={CAl:.6g} J/K, CSen={CSen:.6g} J/K")
print(f"RFeVid={RFeVid:.6g} K/W,  RFeCem={RFeCem:.6g} K/W,  RFe_a={RFe_a:.6g} K/W,  "
      f"RVidCem={RVidCem:.6g} K/W,  RVid_a={RVid_a:.6g} K/W,  RCemAl={RCemAl:.6g} K/W,  "
      f"RAl_a={RAl_a:.6g} K/W , RFeSen={RFeSen:.6g} K/W, RVidSen={RVidSen:.6g} K/W, ")

TFe_d, TCem_d, TVid_d, TAl_d , TSen_d = TFeNi42, TCem, TVid, TAlum, TSensor

# Submuestreo marcadores
Np = 30
idx = np.linspace(0, len(tt) - 1, Np, dtype=int)

plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.35})

# --- Tile layout 3x2 (incluye sensor) ---
fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex=True)

def plot_ax(ax, y_hat, y_dat, name):
    ax.plot(tt, y_hat, "-", lw=2.2, label=f"{name} (PINN)")
    ax.plot(tt[idx], y_dat[idx], "-", color='red' ,  label=f"{name} (datos)")
    ax.set_title(name)
    ax.set_ylabel("T [K]")
    ax.legend()

plot_ax(axs[0, 0], TFe_hat,  TFe_d,  "FeNi42")
plot_ax(axs[0, 1], TCem_hat, TCem_d, "Cemento")
plot_ax(axs[1, 0], TVid_hat, TVid_d, "Vidrio")
plot_ax(axs[1, 1], TAl_hat,  TAl_d,  "Alúmina")
plot_ax(axs[2, 0], TSen_hat, TSen_d, "Sensor")

axs[2, 0].set_xlabel("t [s]")
fig.delaxes(axs[2, 1])

fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

epochs = np.arange(1, EPOCHS + 1)

TFe_hat = TFe_hat - 273.15 
TFe_d = TFe_d - 273.15 

TCem_hat = TCem_hat - 273.15 
TCem_d = TCem_d - 273.15 

TVid_hat = TVid_hat - 273.15 
Vid_d = TVid_d - 273.15 

TAl_hat = TAl_hat - 273.15 
TAl_d = TAl_d - 273.15 

TSen_hat = TSen_hat - 273.15 
TSen_d = TSen_d - 273.15


# plt.figure(figsize=(12,6)) 

# plt.plot(tt, TAl_hat, "-", lw=2.5, label="TAl (PINN)") 
# plt.plot(tt[idx], TAl_d[idx], "-", lw=2.5, color="red", label="TAl (datos)") 
# plt.xlabel("t [s]", fontsize=20) 
# plt.ylabel("T [Cº]", fontsize=20) 
# plt.xticks(fontsize=20) 
# plt.yticks(fontsize=20) 
# plt.legend(fontsize=20)
# plt.show()

# plt.figure(figsize=(12,6)) 

# plt.plot(tt, TFe_hat, "-", lw=2.5, label="TFe (PINN)") 
# plt.plot(tt[idx], TFe_d[idx], "-", lw=2.5, color="red", label="TFe (datos)") 
# plt.xlabel("t [s]", fontsize=20) 
# plt.ylabel("T [Cº]", fontsize=20) 
# plt.xticks(fontsize=20) 
# plt.yticks(fontsize=20) 
# plt.legend(fontsize=20)
# plt.show()


plt.figure(figsize=(12,6)) 

plt.plot(tt, TSen_hat, "-", lw=2.5, label="Sensor (PINN)") 
plt.plot(tt[idx], TSen_d[idx], "-", lw=2.5, color="red", label="Sensor (datos)") 
plt.xlabel("t [s]", fontsize=20) 
plt.ylabel("T [Cº]", fontsize=20) 
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20) 
plt.legend(fontsize=20)
plt.show()


# plt.figure(figsize=(12,6)) 

# plt.plot(epochs, loss_hist, label="LOSS total", lw=2)
# plt.plot(epochs, data_hist, "--", label="DATA")
# plt.plot(epochs, ode_hist,  "--", label="ODE")
# plt.plot(epochs, ic_hist,   "--", label="IC")
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.legend(fontsize=20)
# # plt.ylim(1e-10, 1e-1)

# plt.yscale("log")
# plt.xlabel("Épocas", fontsize=20)
# plt.ylabel("Loss", fontsize=20)

# plt.grid(True, which="both", alpha=0.35)
# plt.tight_layout()
# plt.show()

