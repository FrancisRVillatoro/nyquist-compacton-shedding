#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.optimize as opt
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def build_ops(N: int, h: float):
    r = np.arange(N)
    rows, cols, av, lv = [], [], [], []
    ca = np.array([1, 26, 66, 26, 1], dtype=float) / 120.0
    cb = np.array([-1, -10, 0, 10, 1], dtype=float) / (24.0 * h)
    cc = np.array([-1, 2, 0, -2, 1], dtype=float) / (2.0 * h**3)
    for kk, k in enumerate([-2, -1, 0, 1, 2]):
        c = (r + k) % N
        rows.extend(r)
        cols.extend(c)
        av.extend(np.full(N, ca[kk]))
        lv.extend(np.full(N, cb[kk] + cc[kk]))
    A = sp.csc_matrix((av, (rows, cols)), shape=(N, N))
    L = sp.csc_matrix((lv, (rows, cols)), shape=(N, N))
    return A, L


def compacton(x: np.ndarray, x0: float, Ldom: float, c: float = 1.0):
    z = (x - x0 + Ldom / 2.0) % Ldom - Ldom / 2.0
    u = np.zeros_like(x)
    mask = np.abs(z) <= 2.0 * np.pi
    u[mask] = (4.0 * c / 3.0) * np.cos(z[mask] / 4.0) ** 2
    return u


def midpoint_step(U, A, L, dt, store=False):
    W = np.asarray(U).copy()
    for _ in range(7):
        G = (2.0 / dt) * (A @ (W - U)) + L @ (np.abs(W) * W)
        J = (2.0 / dt) * A + L @ sp.diags(2.0 * np.abs(W), 0, format="csc")
        d = spla.spsolve(J, -G, permc_spec="NATURAL")
        W += d
        if np.max(np.abs(d)) < 8.0e-13:
            break
    U1 = 2.0 * W - U
    if not store:
        return U1
    Jp = A / dt + L @ sp.diags(np.abs(W), 0, format="csc")
    Jm = A / dt - L @ sp.diags(np.abs(W), 0, format="csc")
    return U1, spla.splu(Jp, permc_spec="NATURAL"), Jm


def one_cell_map(U, A, L, dt, kappa, record=False):
    V = np.asarray(U).copy()
    factors = []
    for _ in range(int(kappa)):
        if record:
            V, lu, Jm = midpoint_step(V, A, L, dt, store=True)
            factors.append((lu, Jm))
        else:
            V = midpoint_step(V, A, L, dt)
    V = np.roll(V, -1)
    return (V, factors) if record else V


def solve_traveling_compacton(Uguess, A, L, dt, kappa, tol=2e-10):
    def residual(U):
        return one_cell_map(np.asarray(U), A, L, dt, kappa) - U

    try:
        Us = opt.newton_krylov(
            residual,
            Uguess,
            f_tol=tol,
            maxiter=10,
            inner_maxiter=20,
            rdiff=1e-7,
        )
    except opt.NoConvergence as exc:
        Us = np.asarray(exc.args[0])
    res = float(np.linalg.norm(residual(Us), np.inf))
    return Us, res


def apply_P(Z, factors):
    Y = np.asarray(Z)
    isvec = Y.ndim == 1
    if isvec:
        Y = Y[:, None]
    iscomplex = np.iscomplexobj(Y)
    for lu, Jm in factors:
        rhs = Jm @ Y
        if iscomplex:
            Y = lu.solve(rhs.real) + 1j * lu.solve(rhs.imag)
        else:
            Y = lu.solve(rhs)
    Y = np.roll(Y, -1, axis=0)
    return Y[:, 0] if isvec else Y


def block_ritz(apply, N, seed_vectors, nb=10, maxiter=500, check_every=20, tol=6e-9):
    rng = np.random.default_rng(20260903)
    Q = rng.normal(size=(N, nb)) * 1e-5
    for j, vec in enumerate(seed_vectors[:nb]):
        Q[:, j] += vec
    Q, _ = np.linalg.qr(Q)

    last = None
    stable = 0
    prev_rho = None
    for it in range(1, maxiter + 1):
        Q, _ = np.linalg.qr(apply(Q))
        if it % check_every == 0 or it == maxiter:
            PQ = apply(Q)
            H = Q.conj().T @ PQ
            ew, ev = np.linalg.eig(H)
            order = np.argsort(np.abs(ew))[::-1]
            ew, ev = ew[order], ev[:, order]
            v = Q @ ev[:, 0]
            resid = float(np.linalg.norm(apply(v) - ew[0] * v) / (np.linalg.norm(v) + 1e-300))
            rho = float(abs(ew[0]))
            last = (ew, ev, Q, it, resid)
            if prev_rho is not None and abs(rho - prev_rho) < 2e-9 and resid < tol:
                stable += 1
            else:
                stable = 0
            prev_rho = rho
            if stable >= 2:
                break
    return last


def analyze_case(h, kappa, phi, Ldom, Uguess=None, nb=10, maxiter=500):
    dt = h / kappa
    N = int(round(Ldom / h))
    x = np.arange(N) * h
    A, L = build_ops(N, h)

    nR = int(np.rint((Ldom / 2.0 + 2.0 * np.pi) / h))
    xR = h * (nR + phi)
    x0 = xR - 2.0 * np.pi
    Uc = compacton(x, x0, Ldom)
    guess = Uc if Uguess is None else Uguess

    tic = time.time()
    Us, fp_res = solve_traveling_compacton(guess, A, L, dt, int(kappa))
    _, factors = one_cell_map(Us, A, L, dt, int(kappa), record=True)

    mR = nR % N
    qL = (x0 - 2.0 * np.pi) / h
    phiL = qL - np.floor(qL)
    mL = int(np.rint(qL)) % N
    win = np.arange(-40, 41)
    idxR = (mR + win) % N
    idxL = (mL + win) % N
    rng = np.random.default_rng(8101)
    seedR = np.zeros(N)
    seedL = np.zeros(N)
    seedR[idxR] = rng.normal(size=idxR.size)
    seedL[idxL] = rng.normal(size=idxL.size)

    ew, ev, Q, nit, _ = block_ritz(
        lambda Z: apply_P(Z, factors), N, [seedR, seedL], nb=nb, maxiter=maxiter
    )

    top = []
    for j in range(min(6, len(ew))):
        v = Q @ ev[:, j]
        rr = float(np.linalg.norm(apply_P(v, factors) - ew[j] * v) / (np.linalg.norm(v) + 1e-300))
        top.append((ew[j], rr, v))

    lam, rres, r = top[0]
    ener = np.abs(r) ** 2
    etot = float(ener.sum()) + 1e-300
    eR = float(ener[idxR].sum() / etot)
    eL = float(ener[idxL].sum() / etot)
    mode_type = "real flip" if abs(lam.imag) < 1e-7 and lam.real < 0 else "complex pair"

    row = {
        "h": h,
        "kappa": float(kappa),
        "dt": dt,
        "phi_right": phi,
        "phi_left": phiL,
        "x0": x0,
        "N": N,
        "Ldom": Ldom,
        "fixed_point_residual": fp_res,
        "lambda1_real": float(lam.real),
        "lambda1_imag": float(lam.imag),
        "rho_F": float(abs(lam)),
        "arg_lambda": float(np.angle(lam)),
        "mode_type": mode_type,
        "lambda1_residual": rres,
        "right_edge_energy_fraction": eR,
        "left_edge_energy_fraction": eL,
        "two_edge_energy_fraction": eR + eL,
        "ritz_iterations": nit,
        "runtime_s": time.time() - tic,
    }
    for j, (z, rr, _) in enumerate(top[1:5], start=2):
        row[f"lambda{j}_real"] = float(z.real)
        row[f"lambda{j}_imag"] = float(z.imag)
        row[f"lambda{j}_abs"] = float(abs(z))
        row[f"lambda{j}_residual"] = rr
    return row, Us


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(Path(__file__).resolve().parent))
    p.add_argument("--phi", type=float, default=0.25)
    p.add_argument("--Ldom", type=float, default=16.0)
    p.add_argument("--maxiter", type=int, default=500)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hs = [0.005, 0.01, 0.015, 0.02, 0.03]
    kappas = [5, 10, 15, 20, 30]
    rows = []
    for h in hs:
        prev = None
        print(f"=== h={h:g} ===", flush=True)
        for kappa in kappas:
            row, prev = analyze_case(h, kappa, args.phi, args.Ldom, Uguess=prev, maxiter=args.maxiter)
            rows.append(row)
            print(
                f"kappa={kappa:2d} rho={row['rho_F']:.9f} "
                f"lambda={row['lambda1_real']:+.8f}{row['lambda1_imag']:+.8f}i "
                f"fp={row['fixed_point_residual']:.1e} eig={row['lambda1_residual']:.1e} "
                f"{row['mode_type']} time={row['runtime_s']:.1f}s",
                flush=True,
            )
        pd.DataFrame([r for r in rows if r["h"] == h]).to_csv(out / f"fixed_phi_h{h:g}.csv", index=False)

    df = pd.DataFrame(rows)
    df["excess_percent"] = 100.0 * (df["rho_F"] - 1.0)
    df["log_growth_per_cell"] = np.log(df["rho_F"])
    df.to_csv(out / "fixed_phase_floquet_map.csv", index=False)
    meta = {
        "phi_right": args.phi,
        "Ldom": args.Ldom,
        "h_values": hs,
        "kappa_values": kappas,
        "description": "Full 5x5 Floquet map at fixed right-edge phase; kappa=h/(c dt), c=1.",
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
