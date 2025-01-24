using Printf, Test

mean(x) = sum(x)/length(x)

function log_division(a, b)
    if b == 0
        return a == 0 ? NaN : Inf
    end
    return log(a / b)
end

"""
    grad_test(J, x0, Δx, dJdx; ΔJ=nothing, maxiter=6, h0=5e-2, stol=1e-1, hfactor=8e-1)

Test the gradient using Taylor series convergence.

Compute a series of residuals using zeroth order and first order Taylor approximations.
Each perturbed computation J(x₀ + hᵢ Δx) is compared to J(x₀) and J(x₀) + hᵢ ΔJ,
where hᵢ = hfactor * hᵢ₋₁ and ΔJ = dJdx ⋅ Δx by default. If the computation of J and
dJdx is correct, J is sufficiently smooth, and h is sufficiently small, the zeroth
order approximation should converge linearly and the first order approximation should
converge quadratically.

It can be difficult to obtain the correct convergence, so we average the
convergence factors across maxiter values of h and test that the convergence
factor is more than correct convergence factor minus the stol parameter.

# Mathematical basis

For J sufficiently smooth, the value of a point at a small perturbation from x₀ can be
computed using the Taylor series expansion.
```math
J(x₀ + h Δx) = J(x₀) + h (dJ/dx) Δx + h² (d²J/dx²) : (Δx Δxᵀ) + O(h³)
```

If d²J/dx² is non-zero and h is small enough, the value of the first-order Taylor
approximation differs from the true value by approximately a constant proportional to h².
```math
err(h) = |J(x₀ + h Δx) - J(x₀) - h (dJ/dx) Δx| ≈ h²c
```

If we consider the error for two different values of h, the unknown constant can be eliminated,
and we can determine the convergence rate.
```math
err(h₁) / err(h₂) ≈ (h₁ / h₂)^rate
log(err(h₁) / err(h₂)) ≈ rate * log(h₁ / h₂)
log(err(h₁) / err(h₂))/log(h₁ / h₂) ≈ rate
```
First-order convergence has rate 1, and second order convergence has rate 2.
So if h₁ is divided by a factor α, then the ratio of the errors should be divided by a factor α².
"""
function grad_test(J, x0, Δx, dJdx; ΔJ=nothing, maxiter=6, h0=5e-2, stol=1e-1, hfactor=8e-1, unittest=:test)
    if !xor(isnothing(dJdx), isnothing(ΔJ))
        error("Must specify either dJdx or ΔJ")
    end
    if isnothing(ΔJ)
        ΔJ = dot(dJdx, Δx)
    end
    J0 = J(x0)
    h = h0

    log_factor = log(hfactor)
    expected_f1 = 1e0 / hfactor
    expected_f2 = 1e0 / hfactor ^2e0

    err1 = zeros(Float64, maxiter)
    err2 = zeros(Float64, maxiter)
    Js = zeros(Float64, maxiter)
    @printf("%11s, %12s | %11s, %11s | %11s, %11s | %12s, %12s | %12s %12s %12s \n", "h", "ΔJ", "e1", "e2", "factor1", "factor2", "rate1", "rate2", "Finite-diff", "T1 approx", "T2 approx")
    @printf("%11s, % 12.5e | %11s, %11s | %11.5e, %11.5e | % 12.5e, % 12.5e | \n", "", ΔJ, "0", "0", expected_f1, expected_f2, 1, 2)
    @printf("%11s, %12s | %11s, %11s | %11s, %11s | % 12s, % 12s \n", "___________", "___________", "___________", "___________", "___________", "___________", "____________", "____________")
    all_info = []
    for j=1:maxiter
        Jh = J(x0 + h*Δx)
        Js[j] = Jh
        err1[j] = norm(Jh - J0, 1)
        err2[j] = norm(Jh - J0 - h*ΔJ, 1)
        j == 1 ? prev = 1 : prev = j - 1

        dJ_est = (Jh - J0) / h
        α = expected_f2
        dJ_est1 = (α*Js[prev] - Jh + (1-α)*J0) / (h * (α/hfactor - 1))
        α = -expected_f2
        dJ_est2 = (α*Js[prev] - Jh + (1-α)*J0) / (h * (α/hfactor - 1))

        rate1 = log_division(err1[j], err1[prev]) / log_factor
        rate2 = log_division(err2[j], err2[prev]) / log_factor

        info = (h, h*norm(ΔJ, 1), err1[j], err2[j], err1[prev]/err1[j], err2[prev]/err2[j], rate1, rate2, dJ_est, dJ_est1, dJ_est2)
        push!(all_info, info)
        @printf("%11.5e, % 12.5e | %11.5e, %11.5e | %11.5e, %11.5e | % 12.5e, % 12.5e | % 12.5e, % 12.5e, % 12.5e \n", info...)
        h = h * hfactor
    end

    println()
    @printf("%11s, %12s | %11s, %11s | %11s, %11s | %12s, %12s | %12s %12s %12s \n", "h", "ΔJ", "e1", "e2", "factor1", "factor2", "rate1", "rate2", "Finite-diff", "T1 approx", "T2 approx")
    @printf("%11s, % 12.5e | %11s, %11s | %11.5e, %11.5e | % 12.5e, % 12.5e | \n", "", ΔJ, "0", "0", expected_f1, expected_f2, 1, 2)
    @printf("%11s, %12s | %11s, %11s | %11s, %11s | % 12s, % 12s \n", "___________", "___________", "___________", "___________", "___________", "___________", "____________", "____________")
    for j=1:maxiter
        info = all_info[j]
        @printf("%11.5e, % 12.5e | %11.5e, %11.5e | %11.5e, %11.5e | % 12.5e, % 12.5e | % 12.5e, % 12.5e, % 12.5e \n", info...)
        h = h * hfactor
    end

    factor1 = err1[1:end-1]./err1[2:end]
    factor2 = err2[1:end-1]./err2[2:end]

    rate1 = log_division.(err1[2:end], err1[1:end-1]) / log_factor
    rate2 = log_division.(err2[2:end], err2[1:end-1]) / log_factor

    mean_factor1 = mean(factor1)
    mean_factor2 = mean(factor2)
    @printf("%11s  %12s   %11s  %11s | %11.5e, %11.5e | %12.5e, %12.5e | \n", "", "", "", "mean", mean_factor1, mean_factor2, mean(rate1), mean(rate2))
    @printf("%11s  %12s   %11s  %11s | %11.5e, %11.5e | %12.5e, %12.5e | \n", "", "", "", "min", minimum(factor1), minimum(factor2), minimum(rate1), minimum(rate2))
    println()

    @test mean_factor1 ≥ expected_f1 - stol
    if unittest == :skip
        @test mean_factor2 ≥ expected_f2 - stol skip=true
    elseif unittest == :broken
        @test mean_factor2 ≥ expected_f2 - stol broken=true
    else
        @test mean_factor2 ≥ expected_f2 - stol
    end
end

