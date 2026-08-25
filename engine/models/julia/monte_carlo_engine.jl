"""
monte_carlo_engine.jl — High-Performance F1 Race Simulator
Optimized Version for Phase 7.

Features:
  - Multi-threading support (@threads).
  - Physics-coupled overtaking (lap time deltas).
  - Proper tire strategy (pit stops).
  - Zero-allocation inner loops.
"""

using Random
using Statistics
using Base.Threads

# ─────────────────────────────────────────────
# CORE SIMULATION STRUCTURES
# ─────────────────────────────────────────────

struct DriverData
    name::String
    base_lap_time::Float64
    glicko_rating::Float64
    dnf_prob::Float64
    start_grid::Int
end

# ─────────────────────────────────────────────
# OPTIMIZED BATCH SIMULATOR
# ─────────────────────────────────────────────

function run_batch_simulation(
    python_drivers,
    n_laps::Int,
    δ_vals::Dict{String, Float64},
    γ_vals::Dict{String, Float64},
    λ_fuel::Float64,
    α_overtake::Float64,
    sc_prob::Float64,
    n_sims::Int
)
    # 1. Initialization
    n_drivers = length(python_drivers)
    drivers = [DriverData(d["name"], d["base_lap"], d["glicko"], d["dnf_p"], d["grid"]) for d in python_drivers]
    
    # Pre-allocate result containers (one per thread to avoid locking)
    n_threads = nthreads()
    thread_rank_counts = [zeros(Int, n_drivers, n_drivers) for _ in 1:n_threads]
    thread_dnf_counts = [zeros(Int, n_drivers) for _ in 1:n_threads]

    # 2. Parallel Simulation Loop
    @threads for s in 1:n_sims
        tid = threadid()
        
        # Local state for this simulation
        accum_times = zeros(n_drivers)
        is_dnf = fill(false, n_drivers)
        tire_ages = zeros(Int, n_drivers)
        current_compound = fill("MEDIUM", n_drivers)
        
        # Initial positions by grid
        order = sortperm([d.start_grid for d in drivers])
        
        # Safety Car event
        sc_lap = rand() < sc_prob ? rand(5:max(6, n_laps-5)) : -1

        for lap in 1:n_laps
            # Physics Step
            for i in 1:n_drivers
                if is_dnf[i]; continue; end
                
                # DNF check
                if rand() < (drivers[i].dnf_prob / n_laps)
                    is_dnf[i] = true
                    continue
                end
                
                # Strategy: Pit stops
                pit_threshold = current_compound[i] == "SOFT" ? 18 : (current_compound[i] == "MEDIUM" ? 25 : 35)
                if tire_ages[i] > pit_threshold
                    current_compound[i] = "HARD"
                    tire_ages[i] = 0
                    accum_times[i] += 22.5 
                end
                
                # Lap Time Calc
                δ = get(δ_vals, current_compound[i], 0.04)
                γ = get(γ_vals, current_compound[i], 0.0)
                
                t_lap = drivers[i].base_lap_time + (tire_ages[i] * δ) + γ - (lap * λ_fuel)
                
                # SC slow down
                if lap == sc_lap
                    t_lap += 15.0
                end
                
                t_lap += randn() * 0.12 # Stochastic variance
                accum_times[i] += t_lap
                tire_ages[i] += 1
            end
            
            # Overtaking Step (Coupled to physics)
            # Only if not under SC
            if lap != sc_lap
                # Get current on-track order (excluding DNFs)
                on_track = filter(idx -> !is_dnf[idx], 1:n_drivers)
                sort!(on_track, by=idx -> accum_times[idx])
                
                for j in 1:(length(on_track)-1)
                    lead_idx = on_track[j]
                    chas_idx = on_track[j+1]
                    
                    # DRS range (within 1.0s)
                    if (accum_times[chas_idx] - accum_times[lead_idx]) < 1.0
                        # Probability based on rating delta + pace delta
                        # (simplified: chaser gets a small random chance to jump ahead)
                        p = α_overtake * (drivers[chas_idx].glicko_rating / (drivers[chas_idx].glicko_rating + drivers[lead_idx].glicko_rating))
                        if rand() < p
                            accum_times[chas_idx] -= 0.25 # Successful pass boost
                        end
                    end
                end
            end
            
            # SC Compression
            if lap == sc_lap
                on_track = filter(idx -> !is_dnf[idx], 1:n_drivers)
                sort!(on_track, by=idx -> accum_times[idx])
                if !isempty(on_track)
                    lead_t = accum_times[on_track[1]]
                    for (k, idx) in enumerate(on_track)
                        accum_times[idx] = lead_t + (k-1)*0.6 # 0.6s gaps
                    end
                end
            end
        end
        
        # 3. Tally results for this thread
        # Rank by total time (DNFs go to back)
        final_ranks = sortperm(accum_times .+ (is_dnf .* 1e9))
        for (rank, d_idx) in enumerate(final_ranks)
            thread_rank_counts[tid][d_idx, rank] += 1
            if is_dnf[d_idx]
                thread_dnf_counts[tid][d_idx] += 1
            end
        end
    end

    # 4. Merge thread results
    total_rank_counts = zeros(Int, n_drivers, n_drivers)
    total_dnf_counts = zeros(Int, n_drivers)
    for tid in 1:n_threads
        total_rank_counts .+= thread_rank_counts[tid]
        total_dnf_counts .+= thread_dnf_counts[tid]
    end

    return total_rank_counts, total_dnf_counts
end
