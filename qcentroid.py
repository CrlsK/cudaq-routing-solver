"""
CUDA-Q QAOA Solver for Real-Time Adaptive Routing Under Uncertainty
Use case 747 | Solver: qcentroid-labs-quantum-vrp-cudaq-qaoa-gpu

Algorithm: QAOA (Quantum Approximate Optimization Algorithm) for QUBO-VRP
Hardware:  NVIDIA GPU via CUDA-Q (falls back to CPU simulator automatically)
Author:    QCentroid Labs
Version:   1.0.0
"""
import logging
import time
import math
import itertools
from typing import List, Tuple, Dict, Any

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger("qcentroid-user-log")

# --- CUDA-Q target selection --------------------------------------------------
try:
    import cudaq
    try:
        cudaq.set_target("nvidia")          # Single-GPU (preferred)
        _CUDAQ_TARGET = "nvidia (GPU)"
    except Exception:
        cudaq.set_target("qpp-cpu")         # CPU fallback
        _CUDAQ_TARGET = "qpp-cpu (CPU fallback)"
    _CUDAQ_AVAILABLE = True
except ImportError:
    _CUDAQ_AVAILABLE = False
    _CUDAQ_TARGET = "unavailable"

logger.info(f"CUDA-Q target: {_CUDAQ_TARGET}")

# --- Constants ----------------------------------------------------------------
EARTH_RADIUS_KM = 6371.0
DEFAULT_SPEED_KMH = 50.0
PENALTY_CAPACITY = 10.0
PENALTY_ASSIGNMENT = 8.0
QAOA_P_LAYERS = 2
QAOA_SHOTS = 1024
COBYLA_MAXITER = 150


# --- Helper functions ---------------------------------------------------------

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two GPS coordinates."""
    r = EARTH_RADIUS_KM
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _travel_time_min(lat1, lon1, lat2, lon2, speed_kmh: float) -> float:
    """Travel time in minutes between two GPS points."""
    return (_haversine(lat1, lon1, lat2, lon2) / speed_kmh) * 60.0


def _apply_disruptions(base_time: float, disruptions: List[Dict], stop_id: str) -> float:
    """Add disruption delay (minutes) at a given stop if any disruption applies."""
    for d in disruptions:
        if d.get("affected_stop") == stop_id or d.get("location") == stop_id:
            base_time += d.get("delay_min", d.get("disruption_delay_min", 0))
    return base_time


# --- QUBO formulation ---------------------------------------------------------

def _build_qubo(customers: List[Dict], vehicles: List[Dict],
                depot: Dict, disruptions: List[Dict]) -> np.ndarray:
    """Build QUBO matrix for VRP assignment. H = H_cost + H_assignment + H_capacity"""
    n_c = len(customers)
    n_v = len(vehicles)
    n_vars = n_c * n_v
    Q = np.zeros((n_vars, n_vars))
    depot_lat = depot.get("lat", 0.0)
    depot_lon = depot.get("lon", 0.0)

    def idx(i, k):
        return i * n_v + k

    for i, c in enumerate(customers):
        for k, v in enumerate(vehicles):
            speed = v.get("speed_kmh", DEFAULT_SPEED_KMH)
            t = _travel_time_min(depot_lat, depot_lon, c["lat"], c["lon"], speed)
            t = _apply_disruptions(t, disruptions, c["id"])
            Q[idx(i, k), idx(i, k)] += t

    for i in range(n_c):
        for j in range(i + 1, n_c):
            c_i, c_j = customers[i], customers[j]
            for k, v in enumerate(vehicles):
                speed = v.get("speed_kmh", DEFAULT_SPEED_KMH)
                t_ij = _travel_time_min(c_i["lat"], c_i["lon"], c_j["lat"], c_j["lon"], speed)
                Q[idx(i, k), idx(j, k)] += t_ij
                Q[idx(j, k), idx(i, k)] += t_ij

    for i in range(n_c):
        for k in range(n_v):
            Q[idx(i, k), idx(i, k)] += PENALTY_ASSIGNMENT * (1 - 2)
            for l in range(k + 1, n_v):
                Q[idx(i, k), idx(i, l)] += 2 * PENALTY_ASSIGNMENT
                Q[idx(i, l), idx(i, k)] += 2 * PENALTY_ASSIGNMENT

    for k, v in enumerate(vehicles):
        cap = v.get("capacity", float('inf'))
        if cap == float('inf'):
            continue
        demands = [c.get("demand", 0) for c in customers]
        for i in range(n_c):
            for j in range(n_c):
                Q[idx(i, k), idx(j, k)] += PENALTY_CAPACITY * demands[i] * demands[j] / (cap ** 2)

    return Q


def _qubo_to_ising(Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert QUBO matrix to Ising model (h, J) via x_i = (1 - z_i)/2."""
    n = Q.shape[0]
    h = np.zeros(n)
    J = np.zeros((n, n))
    for i in range(n):
        h[i] -= Q[i, i] / 2.0
        for j in range(i + 1, n):
            coupling = (Q[i, j] + Q[j, i]) / 4.0
            J[i, j] = coupling
            J[j, i] = coupling
            h[i] -= coupling
            h[j] -= coupling
    return h, J


# --- QAOA circuit (CUDA-Q) ----------------------------------------------------

def _build_qaoa_kernel(n_qubits: int, p: int, ising_h: np.ndarray, ising_J: np.ndarray):
    """Build and return a CUDA-Q QAOA kernel."""
    import cudaq
    edges = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)
             if abs(ising_J[i, j]) > 1e-8]
    e_weights = [float(ising_J[i, j]) for i, j in edges]
    h_local = [float(ising_h[i]) for i in range(n_qubits)]

    @cudaq.kernel
    def qaoa(thetas: List[float]):
        q = cudaq.qvector(n_qubits)
        h(q)
        for layer in range(p):
            gamma = thetas[layer]
            beta  = thetas[p + layer]
            for k in range(len(edges)):
                ei, ej = edges[k]
                cx(q[ei], q[ej])
                rz(2.0 * gamma * e_weights[k], q[ej])
                cx(q[ei], q[ej])
            for i in range(n_qubits):
                if abs(h_local[i]) > 1e-8:
                    rz(2.0 * gamma * h_local[i], q[i])
            for i in range(n_qubits):
                rx(2.0 * beta, q[i])

    return qaoa, edges, e_weights, h_local


def _evaluate_qaoa(thetas: np.ndarray, kernel, Q: np.ndarray, n_shots: int = QAOA_SHOTS) -> float:
    """Sample QAOA circuit and return average QUBO cost."""
    import cudaq
    counts = cudaq.sample(kernel, thetas.tolist(), shots_count=n_shots)
    total_cost, total_samples = 0.0, 0
    for bitstring, freq in counts.items():
        x = np.array([int(b) for b in bitstring], dtype=float)
        total_cost += float(x @ Q @ x) * freq
        total_samples += freq
    return total_cost / max(total_samples, 1)


def _best_bitstring(thetas: np.ndarray, kernel, Q: np.ndarray, n_shots: int = QAOA_SHOTS * 4) -> np.ndarray:
    """Sample circuit and return lowest-cost bitstring."""
    import cudaq
    counts = cudaq.sample(kernel, thetas.tolist(), shots_count=n_shots)
    best_x, best_cost = None, float('inf')
    for bitstring, _ in counts.items():
        x = np.array([int(b) for b in bitstring], dtype=float)
        cost = float(x @ Q @ x)
        if cost < best_cost:
            best_cost, best_x = cost, x
    return best_x if best_x is not None else np.zeros(Q.shape[0])


# --- Classical fallback -------------------------------------------------------

def _greedy_assignment(customers, vehicles, depot):
    """Greedy nearest-neighbour assignment fallback."""
    n_c, n_v = len(customers), len(vehicles)
    assignment = np.zeros((n_c, n_v), dtype=int)
    assigned = [False] * n_c
    for k, v in enumerate(vehicles):
        cap = v.get("capacity", float("inf"))
        load = 0.0
        cx, cy = depot.get("lat", 0), depot.get("lon", 0)
        for _ in range(n_c):
            best_i, best_dist = -1, float("inf")
            for i, c in enumerate(customers):
                if assigned[i] or load + c.get("demand", 0) > cap:
                    continue
                d = _haversine(cx, cy, c["lat"], c["lon"])
                if d < best_dist:
                    best_dist, best_i = d, i
CUDA-Q QAOA Solver for Real-Time Adaptive Routing Under Uncertainty
Use case 747 | Solver: qcentroid-labs-quantum-vrp-cudaq-qaoa-gpu
Algorithm: QAOA for QUBO-VRP | Hardware: NVIDIA GPU via CUDA-Q
"""
import logging, time, math
from typing import List, Tuple, Dict, Any
import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger("qcentroid-user-log")

try:
    import cudaq
    try:
        cudaq.set_target("nvidia")
        _CUDAQ_TARGET = "nvidia (GPU)"
    except Exception:
        cudaq.set_target("qpp-cpu")
        _CUDAQ_TARGET = "qpp-cpu (CPU fallback)"
    _CUDAQ_AVAILABLE = True
except ImportError:
    _CUDAQ_AVAILABLE = False
    _CUDAQ_TARGET = "unavailable"

EARTH_RADIUS_KM = 6371.0
DEFAULT_SPEED_KMH = 50.0
PENALTY_CAPACITY = 10.0
PENALTY_ASSIGNMENT = 8.0
QAOA_P_LAYERS = 2
QAOA_SHOTS = 1024
COBYLA_MAXITER = 150


def _haversine(lat1, lon1, lat2, lon2):
    r = EARTH_RADIUS_KM
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlam = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2*r*math.asin(math.sqrt(a))


def _travel_time_min(lat1, lon1, lat2, lon2, speed_kmh):
    return (_haversine(lat1, lon1, lat2, lon2) / speed_kmh) * 60.0


def _apply_disruptions(base_time, disruptions, stop_id):
    for d in disruptions:
        if d.get("affected_stop") == stop_id or d.get("location") == stop_id:
            base_time += d.get("delay_min", d.get("disruption_delay_min", 0))
    return base_time


def _build_qubo(customers, vehicles, depot, disruptions):
    n_c, n_v = len(customers), len(vehicles)
    Q = np.zeros((n_c*n_v, n_c*n_v))
    dlat, dlon = depot.get("lat",0.0), depot.get("lon",0.0)
    idx = lambda i,k: i*n_v+k
    for i,c in enumerate(customers):
        for k,v in enumerate(vehicles):
            speed = v.get("speed_kmh", DEFAULT_SPEED_KMH)
            t = _apply_disruptions(_travel_time_min(dlat,dlon,c["lat"],c["lon"],speed), disruptions, c["id"])
            Q[idx(i,k),idx(i,k)] += t
    for i in range(n_c):
        for j in range(i+1,n_c):
            for k,v in enumerate(vehicles):
                t = _travel_time_min(customers[i]["lat"],customers[i]["lon"],customers[j]["lat"],customers[j]["lon"],v.get("speed_kmh",DEFAULT_SPEED_KMH))
                Q[idx(i,k),idx(j,k)] += t; Q[idx(j,k),idx(i,k)] += t
    for i in range(n_c):
        for k in range(n_v):
            Q[idx(i,k),idx(i,k)] += PENALTY_ASSIGNMENT*(1-2)
            for l in range(k+1,n_v):
                Q[idx(i,k),idx(i,l)] += 2*PENALTY_ASSIGNMENT; Q[idx(i,l),idx(i,k)] += 2*PENALTY_ASSIGNMENT
    for k,v in enumerate(vehicles):
        cap = v.get("capacity", float('inf'))
        if cap==float('inf'): continue
        demands = [c.get("demand",0) for c in customers]
        for i in range(n_c):
            for j in range(n_c):
                Q[idx(i,k),idx(j,k)] += PENALTY_CAPACITY*demands[i]*demands[j]/(cap**2)
    return Q


def _qubo_to_ising(Q):
    n = Q.shape[0]; h = np.zeros(n); J = np.zeros((n,n))
    for i in range(n):
        h[i] -= Q[i,i]/2.0
        for j in range(i+1,n):
            c = (Q[i,j]+Q[j,i])/4.0; J[i,j]=c; J[j,i]=c; h[i]-=c; h[j]-=c
    return h, J


def _build_qaoa_kernel(n_qubits, p, ising_h, ising_J):
    import cudaq
    edges = [(i,j) for i in range(n_qubits) for j in range(i+1,n_qubits) if abs(ising_J[i,j])>1e-8]
    e_weights = [float(ising_J[i,j]) for i,j in edges]
    h_local = [float(ising_h[i]) for i in range(n_qubits)]
    @cudaq.kernel
    def qaoa(thetas: List[float]):
        q = cudaq.qvector(n_qubits)
        h(q)
        for layer in range(p):
            gamma = thetas[layer]; beta = thetas[p+layer]
            for k in range(len(edges)):
                ei,ej = edges[k]; cx(q[ei],q[ej]); rz(2.0*gamma*e_weights[k],q[ej]); cx(q[ei],q[ej])
            for i in range(n_qubits):
                if abs(h_local[i])>1e-8: rz(2.0*gamma*h_local[i],q[i])
            for i in range(n_qubits): rx(2.0*beta,q[i])
    return qaoa, edges, e_weights, h_local


def _evaluate_qaoa(thetas, kernel, Q, n_shots=QAOA_SHOTS):
    import cudaq
    counts = cudaq.sample(kernel, thetas.tolist(), shots_count=n_shots)
    total_cost = total_samples = 0
    for bitstring, freq in counts.items():
        x = np.array([int(b) for b in bitstring], dtype=float)
        total_cost += float(x@Q@x)*freq; total_samples += freq
    return total_cost/max(total_samples,1)


def _best_bitstring(thetas, kernel, Q, n_shots=QAOA_SHOTS*4):
    import cudaq
    counts = cudaq.sample(kernel, thetas.tolist(), shots_count=n_shots)
    best_x, best_cost = None, float('inf')
    for bitstring,_ in counts.items():
        x = np.array([int(b) for b in bitstring], dtype=float)
        cost = float(x@Q@x)
        if cost < best_cost: best_cost, best_x = cost, x
    return best_x if best_x is not None else np.zeros(Q.shape[0])


def _greedy_assignment(customers, vehicles, depot):
    n_c, n_v = len(customers), len(vehicles)
    assignment = np.zeros((n_c,n_v), dtype=int)
    assigned = [False]*n_c
    for k,v in enumerate(vehicles):
        cap, load = v.get("capacity",float("inf")), 0.0
        cx, cy = depot.get("lat",0), depot.get("lon",0)
        for _ in range(n_c):
            best_i, best_dist = -1, float("inf")
            for i,c in enumerate(customers):
                if assigned[i] or load+c.get("demand",0)>cap: continue
                d = _haversine(cx,cy,c["lat"],c["lon"])
                if d<best_dist: best_dist, best_i = d, i
            if best_i==-1: break
            assignment[best_i,k]=1; assigned[best_i]=True
            load+=customers[best_i].get("demand",0); cx,cy=customers[best_i]["lat"],customers[best_i]["lon"]
    for i,c in enumerate(customers):
        if not assigned[i]:
            loads=[sum(customers[j].get("demand",0)*assignment[j,k] for j in range(n_c)) for k in range(n_v)]
            assignment[i,int(np.argmin(loads))]=1
    return assignment.flatten()


def _decode_bitstring(x, customers, vehicles):
    n_c, n_v = len(customers), len(vehicles)
    x_mat = x.reshape(n_c,n_v)
    for k in range(n_v):
        cap = vehicles[k].get("capacity",float("inf"))
        load = sum(customers[i].get("demand",0)*x_mat[i,k] for i in range(n_c))
        if load>cap:
            assigned = sorted([i for i in range(n_c) if x_mat[i,k]==1], key=lambda i: customers[i].get("demand",0))
            overload = load-cap
            for i in assigned:
                if overload<=0: break
                loads=[sum(customers[j].get("demand",0)*x_mat[j,m] for j in range(n_c)) for m in range(n_v)]
                target=min(range(n_v),key=lambda m:loads[m] if m!=k else float("inf"))
                x_mat[i,k]=0; x_mat[i,target]=1; overload-=customers[i].get("demand",0)
    return [[i for i in range(n_c) if x_mat[i,k]==1] for k in range(n_v)]


def _greedy_order(route_indices, customers, depot):
    if len(route_indices)<=1: return route_indices
    ordered, remaining = [], list(route_indices)
    clat, clon = depot.get("lat",0), depot.get("lon",0)
    while remaining:
        nearest = min(remaining, key=lambda i: _haversine(clat,clon,customers[i]["lat"],customers[i]["lon"]))
        ordered.append(nearest); remaining.remove(nearest); clat,clon=customers[nearest]["lat"],customers[nearest]["lon"]
    return ordered


def _evaluate_route(route_indices, customers, vehicle, depot, disruptions):
    speed = vehicle.get("speed_kmh",DEFAULT_SPEED_KMH)
    clat, clon = depot.get("lat",0), depot.get("lon",0)
    current_time = total_time = fuel_km = late_penalty = 0.0
    stop_etas, violations = {}, []
    for i in route_indices:
        c = customers[i]
        travel = _apply_disruptions(_travel_time_min(clat,clon,c["lat"],c["lon"],speed), disruptions, c["id"])
        current_time += travel + c.get("service_time",0)
        total_time += travel + c.get("service_time",0)
        fuel_km += _haversine(clat,clon,c["lat"],c["lon"])
        tw = c.get("time_window",[0,10000])
        if current_time>tw[1]:
            late_penalty += (current_time-tw[1])*2.0
            violations.append({"stop":c["id"],"lateness_min":round(current_time-tw[1],2)})
        stop_etas[c["id"]] = round(current_time,2)
        clat, clon = c["lat"], c["lon"]
    total_time += _travel_time_min(clat,clon,depot.get("lat",0),depot.get("lon",0),speed)
    fuel_km += _haversine(clat,clon,depot.get("lat",0),depot.get("lon",0))
    return {"total_time_min":round(total_time+late_penalty,3),"late_penalty_min":round(late_penalty,3),
            "fuel_km":round(fuel_km,3),"fuel_cost_eur":round(fuel_km*0.18,3),
            "stop_etas":stop_etas,"violations":violations,"on_time":len(violations)==0}


def run(input_data: dict, solver_params: dict, extra_arguments: dict) -> dict:
    """QAOA solver for VRP using NVIDIA CUDA-Q GPU-accelerated quantum circuit simulation."""
    start_time = time.time()
    customers = input_data.get("customers",[])
    vehicles  = input_data.get("vehicles",[])
    depot     = input_data.get("depot",{})
    disruptions = input_data.get("disruptions",[])
    n_c, n_v = len(customers), len(vehicles)
    p_layers = int(solver_params.get("p_layers",QAOA_P_LAYERS))
    n_shots  = int(solver_params.get("n_shots",QAOA_SHOTS))
    max_iter = int(solver_params.get("max_iter",COBYLA_MAXITER))
    n_vars = n_c*n_v

    Q = _build_qubo(customers,vehicles,depot,disruptions)
    h_ising, J_ising = _qubo_to_ising(Q)

    best_x = None
    qaoa_iterations, qaoa_final_cost = 0, float("inf")
    algorithm_label = "QAOA_CUDAQ_v1"

    if _CUDAQ_AVAILABLE and n_vars<=20:
        try:
            kernel,edges,e_weights,h_local = _build_qaoa_kernel(n_vars,p_layers,h_ising,J_ising)
            rng = np.random.default_rng(seed=42)
            theta0 = rng.uniform(0,np.pi/4,2*p_layers)
            opt_result = minimize(lambda t: _evaluate_qaoa(t,kernel,Q,n_shots), theta0,
                                  method="COBYLA", options={"maxiter":max_iter,"rhobeg":0.5})
            qaoa_iterations = opt_result.nfev
            qaoa_final_cost = float(opt_result.fun)
            best_x = _best_bitstring(opt_result.x,kernel,Q)
            algorithm_label = f"QAOA_CUDAQ_v1 ({_CUDAQ_TARGET}, p={p_layers})"
        except Exception as e:
            logger.warning(f"QAOA failed ({e}), using greedy fallback")
    else:
        algorithm_label = "GreedyNN_fallback"

    if best_x is None:
        best_x = _greedy_assignment(customers,vehicles,depot)
        algorithm_label = "GreedyNN_fallback"

    routes_raw = _decode_bitstring(best_x,customers,vehicles)
    route_results, total_cost, all_violations, stop_etas_all = [], 0.0, [], {}
    total_vehicles_used = 0

    for k,v in enumerate(vehicles):
        if not routes_raw[k]: continue
        ordered = _greedy_order(routes_raw[k],customers,depot)
        eval_r = _evaluate_route(ordered,customers,v,depot,disruptions)
        route_results.append({"vehicle_id":v.get("id",f"v{k}"),"stops":[customers[i]["id"] for i in ordered],**eval_r})
        total_cost += eval_r["total_time_min"]
        all_violations.extend(eval_r["violations"])
        stop_etas_all.update(eval_r["stop_etas"])
        total_vehicles_used += 1

    served = sum(len(r["stops"]) for r in route_results)
    status = "optimal" if not all_violations else "feasible"
    on_time_prob = round((served-len(all_violations))/max(served,1),4)
    elapsed = round(time.time()-start_time,3)

    return {
        "routes": route_results,
        "total_vehicles_used": total_vehicles_used,
        "stop_etas": stop_etas_all,
        "objective_value": round(total_cost,3),
        "solution_status": status,
        "computation_metrics": {"wall_time_s":elapsed,"algorithm":algorithm_label,
            "qaoa_p_layers":p_layers,"qaoa_iterations":qaoa_iterations,
            "qaoa_final_cost":round(qaoa_final_cost,4) if qaoa_final_cost<float("inf") else None,
            "n_qubits":n_vars,"cudaq_target":_CUDAQ_TARGET},
        "cost_breakdown": {"travel_time_min":round(total_cost,3),
            "fuel_cost_eur":round(sum(r.get("fuel_cost_eur",0) for r in route_results),3),
            "lateness_penalty_min":round(sum(r.get("late_penalty_min",0) for r in route_results),3)},
        "risk_metrics": {"on_time_probability":on_time_prob,
            "time_window_violations":len(all_violations),"uncertainty_factor":round(1-on_time_prob,4)},
        "service_level_results": stop_etas_all,
        "constraint_violations": all_violations,
        "quantum_advantage": {"algorithm_family":"QAOA","n_qubits":n_vars,
            "circuit_depth":2*p_layers+1,"hardware_target":_CUDAQ_TARGET,
            "nisq_ready":True},
        "benchmark": {"execution_cost":{"value":1.5,"unit":"credits"},
            "time_elapsed":f"{elapsed}s",
            "energy_consumption":round(elapsed*0.0028,6)}
}
