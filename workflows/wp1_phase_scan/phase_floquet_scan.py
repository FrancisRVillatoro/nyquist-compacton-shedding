#!/usr/bin/env python3
import argparse
import json
import math
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


def fourier_shift_periodic(U: np.ndarray, shift_cells: float):
    n = U.size
    q = 2.0 * np.pi * np.fft.fftfreq(n)
    return np.fft.ifft(np.fft.fft(U) * np.exp(-1j * q * shift_cells)).real


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


def apply_PT(Z, factors):
    Y = np.asarray(Z)
    isvec = Y.ndim == 1
    if isvec:
        Y = Y[:, None]
    Y = np.roll(Y, 1, axis=0)
    iscomplex = np.iscomplexobj(Y)
    for lu, Jm in factors[::-1]:
        if iscomplex:
            tmp = lu.solve(Y.real, trans="T") + 1j * lu.solve(Y.imag, trans="T")
        else:
            tmp = lu.solve(Y, trans="T")
        Y = Jm.T @ tmp
    return Y[:, 0] if isvec else Y


def block_ritz(apply, N, seed_vectors, nb=10, maxiter=420, check_every=20, tol=1e-8):
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
            rv = apply(v) - ew[0] * v
            resid = float(np.linalg.norm(rv) / (np.linalg.norm(v) + 1e-300))
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


def highpass(z):
    return (2.0 * z - np.roll(z, 1) - np.roll(z, -1)) / 4.0


def analyze_phase(h, kappa, phi, Ldom, Uguess=None, nb=10, maxiter=420):
    dt = h / kappa
    N = int(round(Ldom / h))
    x = np.arange(N) * h
    A, L = build_ops(N, h)

    # Keep the right-edge lattice index fixed and vary only its subcell phase.
    nR = int(np.rint((Ldom / 2.0 + 2.0 * np.pi) / h))
    xR = h * (nR + phi)
    x0 = xR - 2.0 * np.pi
    Uc = compacton(x, x0, Ldom)

    if Uguess is None:
        guess = Uc
    else:
        guess = Uguess

    tic = time.time()
    Us, fp_res = solve_traveling_compacton(guess, A, L, dt, int(kappa))
    _, factors = one_cell_map(Us, A, L, dt, int(kappa), record=True)

    mR = nR % N
    xL = x0 - 2.0 * np.pi
    qL = xL / h
    nL = int(np.floor(qL))
    phiL = qL - nL
    mL = int(np.rint(qL)) % N

    win = np.arange(-40, 41)
    idxR = (mR + win) % N
    idxL = (mL + win) % N
    seedR = np.zeros(N)
    seedL = np.zeros(N)
    rng = np.random.default_rng(8101)
    seedR[idxR] = rng.normal(size=idxR.size)
    seedL[idxL] = rng.normal(size=idxL.size)

    right = block_ritz(
        lambda Z: apply_P(Z, factors),
        N,
        [seedR, seedL],
        nb=nb,
        maxiter=maxiter,
    )
    ew, ev, Q, nit, _ = right
    lam = ew[0]
    r = Q @ ev[:, 0]
    rres = float(np.linalg.norm(apply_P(r, factors) - lam * r) / np.linalg.norm(r))

    left = block_ritz(
        lambda Z: apply_PT(Z, factors),
        N,
        [seedR, seedL],
        nb=nb,
        maxiter=maxiter,
    )
    ewL, evL, QL, nitL, _ = left
    target = np.conj(lam)
    jj = int(np.argmin(np.abs(ewL - target)))
    lamL = ewL[jj]
    ell = QL @ evL[:, jj]
    lres = float(np.linalg.norm(apply_PT(ell, factors) - lamL * ell) / np.linalg.norm(ell))

    overlap = np.vdot(ell, r)
    ell = ell / np.conj(overlap)
    biorth = np.vdot(ell, r)

    # Coordinate normalized by the right-edge high-pass peak of the eigenvector.
    hp_r = highpass(r)
    hr = float(np.max(np.abs(hp_r[idxR])))
    rhat = r / hr
    lhat = hr * ell
    q0 = np.vdot(lhat, (Uc - Us) / h**2)

    ener = np.abs(r) ** 2
    etot = float(ener.sum()) + 1e-300
    eR = float(ener[idxR].sum() / etot)
    eL = float(ener[idxL].sum() / etot)

    # Edge-defect proxies.  The left-edge orientation changes signs, but the
    # norm is the relevant phase-strength diagnostic here.
    DN_R = (2.0 * phi - 1.0) / 180.0
    DN_L = (2.0 * phiL - 1.0) / 180.0
    DN_pair = math.sqrt(DN_R**2 + DN_L**2)

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
        "lambda_real": float(lam.real),
        "lambda_imag": float(lam.imag),
        "rho_F": float(abs(lam)),
        "arg_lambda": float(np.angle(lam)),
        "mode_type": mode_type,
        "right_eigen_residual": rres,
        "left_eigen_residual": lres,
        "left_ritz_lambda_real": float(lamL.real),
        "left_ritz_lambda_imag": float(lamL.imag),
        "biorthogonality_real": float(biorth.real),
        "biorthogonality_imag": float(biorth.imag),
        "right_edge_energy_fraction": eR,
        "left_edge_energy_fraction": eL,
        "two_edge_energy_fraction": eR + eL,
        "q0_real": float(q0.real),
        "q0_imag": float(q0.imag),
        "q0_abs": float(abs(q0)),
        "DN_right": DN_R,
        "DN_left": DN_L,
        "DN_pair_norm": DN_pair,
        "right_ritz_iterations": nit,
        "left_ritz_iterations": nitL,
        "runtime_s": time.time() - tic,
    }
    return row, Us


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / 'outputs'))
    parser.add_argument("--nphase", type=int, default=16)
    parser.add_argument("--Ldom", type=float, default=16.0)
    parser.add_argument("--cases", nargs="*", default=["0.01,5", "0.02,20", "0.01,20"])
    parser.add_argument("--maxiter", type=int, default=420)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    phis = np.arange(args.nphase, dtype=float) / args.nphase
    allrows = []

    for spec in args.cases:
        h, kappa = map(float, spec.split(","))
        case = f"h{h:g}_k{kappa:g}"
        print(f"=== {case} ===", flush=True)
        prevU = None
        prev_phi = None
        rows = []
        for ii, phi in enumerate(phis):
            # Fourier-shift the previous fixed point by the phase increment.
            guess = None
            if prevU is not None:
                dphi = phi - prev_phi
                guess = fourier_shift_periodic(prevU, dphi)
            row, Us = analyze_phase(
                h,
                kappa,
                float(phi),
                args.Ldom,
                Uguess=guess,
                maxiter=args.maxiter,
            )
            row["case"] = case
            rows.append(row)
            allrows.append(row)
            prevU, prev_phi = Us, phi
            print(
                f"phi={phi:.4f} rho={row['rho_F']:.8f} "
                f"lambda={row['lambda_real']:+.7f}{row['lambda_imag']:+.7f}i "
                f"q0={row['q0_abs']:.3e} fp={row['fixed_point_residual']:.1e} "
                f"rres={row['right_eigen_residual']:.1e} {row['mode_type']}",
                flush=True,
            )
        pd.DataFrame(rows).to_csv(out / f"phase_scan_{case}.csv", index=False)

    df = pd.DataFrame(allrows)
    df.to_csv(out / "phase_scan_all.csv", index=False)
    meta = {
        "nphase": args.nphase,
        "Ldom": args.Ldom,
        "cases": args.cases,
        "description": "Phase-resolved Floquet scan for numerical compactons; kappa=h/(c dt), c=1.",
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved {out / 'phase_scan_all.csv'}", flush=True)


if __name__ == "__main__":
    main()
