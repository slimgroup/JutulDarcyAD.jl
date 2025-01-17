## A 64×64 2D example for end-to-end inversion of a tortuous channel

using DrWatson
@quickactivate "JutulDarcyRules"

using JutulDarcyRules
using LinearAlgebra
using PythonPlot
using Flux
using LineSearches
using JLD2
using JUDI
using Random

Random.seed!(2023)

sim_name = "end-to-end-inv"
exp_name = "tortuous"

# patchy saturation model
include("utils.jl")

mkpath(datadir())
mkpath(plotsdir())

## grid size
JLD2.@load datadir("K.jld2") K K0;
K = Float64.(K * md);
K0 = Float64.(K0 * md);
n = size(K)
d = (15.0, 15.0)

ϕ = 0.25
model0 = jutulModel((n[1], 1, n[2]), (d[1], 10.0, d[2]), ϕ, K1to3(K0))
model = jutulModel((n[1], 1, n[2]), (d[1], 10.0, d[2]), ϕ, K1to3(K))

## simulation time steppings
tstep = 40 * ones(51)
tot_time = sum(tstep)

# observation vintages
nv = 11
survey_indices = Int.(round.(range(1, stop=length(tstep), length=nv)))

## injection & production
inj_loc = (3, 1, 32) .* (d[1], 10.0, d[2])
prod_loc = (62, 1, 32) .* (d[1], 10.0, d[2])
irate = 5e-3
f = jutulForce(irate, [inj_loc, prod_loc])

## set up modeling operator
S = jutulModeling(model, tstep)

## simulation
mesh = CartesianMesh(model)
T(x) = log.(KtoTrans(mesh, K1to3(exp.(x))))

logK0 = log.(K0)
logK = log.(K)
logK_init = deepcopy(logK0)

@time state0 = S(T(logK0), f)
@time state = S(T(logK), f)

### observed states
O(state::AbstractVector) = Float32.(permutedims(reshape(state[1:length(tstep)*prod(n)], n[1], n[end], length(tstep)), [3,1,2])[survey_indices,:,:])
sw_true = O(state)

# set up rock physics
vp = 3500 * ones(Float32,n)     # p-wave
phi = 0.25f0 * ones(Float32,n)  # porosity
rho = 2200 * ones(Float32,n)    # density
R(c::AbstractArray{Float32,3}) = Patchy(c,vp,rho,phi)[1]
vps = R(sw_true)   # time-varying vp

## upsampling
upsample = 2
u(x::Vector{Matrix{Float32}}) = [repeat(x[i], inner=(upsample,upsample)) for i = 1:nv]
vpups = u(vps)

##### Wave equation
nw = n.*upsample
dw = (15f0/upsample, 15f0/upsample)        # discretization for wave equation
o = (0f0, 0f0)          # origin

nsrc = 32       # num of sources
nrec = 960      # num of receivers

models = [Model(nw, dw, o, (1f3 ./ vpups[i]).^2f0; nb = 80) for i = 1:nv]   # wave model

timeS = timeR = 750f0               # recording time
dtS = dtR = 1f0                     # recording time sampling rate
ntS = Int(floor(timeS/dtS))+1       # time samples
ntR = Int(floor(timeR/dtR))+1       # source time samples

# source locations -- half at the left hand side of the model, half on top
xsrc = convertToCell(vcat(range(dw[1],stop=dw[1],length=Int(nsrc/2)),range(dw[1],stop=(nw[1]-1)*dw[1],length=Int(nsrc/2))))
ysrc = convertToCell(range(0f0,stop=0f0,length=nsrc))
zsrc = convertToCell(vcat(range(dw[2],stop=(nw[2]-1)*dw[2],length=Int(nsrc/2)),range(10f0,stop=10f0,length=Int(nsrc/2))))

# receiver locations -- half at the right hand side of the model, half on top
xrec = vcat(range((nw[1]-1)*dw[1],stop=(nw[1]-1)*dw[1], length=Int(nrec/2)),range(dw[1],stop=(nw[1]-1)*dw[1],length=Int(nrec/2)))
yrec = 0f0
zrec = vcat(range(dw[2],stop=(nw[2]-1)*dw[2],length=Int(nrec/2)),range(10f0,stop=10f0,length=Int(nrec/2)))

# set up src/rec geometry
srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtS, t=timeS)
recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)

# set up source
f0 = 0.05f0     # kHz
wavelet = ricker_wavelet(timeS, dtS, f0)
q = judiVector(srcGeometry, wavelet)

# set up simulation operators
Fs = [judiModeling(models[i], srcGeometry, recGeometry) for i = 1:nv] # acoustic wave equation solver

## wave physics
function F(v::Vector{Matrix{Float32}})
    m = [vec(1f3./v[i]).^2f0 for i = 1:nv]
    return [Fs[i](m[i], q) for i = 1:nv]
end

# Define seismic data directory
mkpath(datadir("seismic-data"))
misc_dict = @strdict nsrc nrec upsample

### generate/load data
if ~isfile(datadir("seismic-data", savename(misc_dict, "jld2"; digits=6)))
    println("generating data")
    global d_obs = [Fs[i]*q for i = 1:nv]
    seismic_dict = @strdict nsrc nrec upsample d_obs q srcGeometry recGeometry model
    @tagsave(
        datadir("seismic-data", savename(seismic_dict, "jld2"; digits=6)),
        seismic_dict;
        safe=true
    )
else
    println("loading data")
    JLD2.@load datadir("seismic-data", savename(misc_dict, "jld2"; digits=6)) d_obs
    global d_obs = d_obs
end

## add noise
noise_ = deepcopy(d_obs)
for i = 1:nv
    for j = 1:nsrc
        noise_[i].data[j] = randn(Float32, ntR, nrec)
    end
end
snr = 10f0
noise_ = noise_/norm(noise_) *  norm(d_obs) * 10f0^(-snr/20f0)
d_obs = d_obs + noise_

ls = BackTracking(order=3, iterations=10)

lower, upper = 1.1*minimum(logK), 0.9*maximum(logK)
box_logK(x::AbstractArray{T}) where T = max.(min.(x,T(upper)),T(lower))

# Main loop
niterations = 100
fhistory = zeros(niterations)

### add box to co2 and velocity
box_co2(x::AbstractArray{T}) where T = max.(min.(x,T(1)),T(0))
box_v(x::AbstractMatrix{T}) where T = max.(min.(x,T(3501)),T(3200))
box_v(x::AbstractVector) = [box_v(x[i]) for i = 1:length(x)]
y_init = box_co2(O(S(T(logK_init), f)))

# objective function for inversion
fval = Inf32

function obj(logK)
    c = box_co2(O(S(T(logK), f))); v = R(c); v_up = box_v(u(v)); dpred = F(v_up);
    global fval = .5f0 * norm(dpred-d_obs)^2f0
    @show fval
    return fval
end

for j=1:niterations

    global fval = fval
    Base.flush(Base.stdout)
    ## AD by Flux
    @time g = gradient(()->obj(logK0), Flux.params(logK0))[logK0]
    fhistory[j] = fval
    p = -g
    
    println("Inversion iteration no: ",j,"; function value: ", fhistory[j])

    # linesearch
    function f_(α)
        misfit = obj(box_logK(logK0 .+ α .* p))
        @show α, misfit
        return misfit
    end

    step, fval = ls(f_, 3e-3, fhistory[j], dot(g, p))

    # Update model and bound projection
    global logK0 = box_logK(logK0 .+ step .* p)

    ### plotting
    y_predict = box_co2(O(S(T(logK0), f)))

    ### save intermediate results
    save_dict = @strdict j snr logK0 step niterations nv nsrc nrec survey_indices fhistory
    @tagsave(
        joinpath(datadir(sim_name, exp_name), savename(save_dict, "jld2"; digits=6)),
        save_dict;
        safe=true
    )

    ## save figure
    fig_name = @strdict j snr niterations nv nsrc nrec survey_indices

    ## compute true and plot
    SNR = -2f1 * log10(norm(K-exp.(logK0))/norm(K))
    fig = figure(figsize=(20,12));
    subplot(2,2,1);
    imshow(exp.(logK0)'./md,vmin=20,vmax=120);title("inversion by NN, $(j) iter");colorbar();
    subplot(2,2,2);
    imshow(K'./md,vmin=20,vmax=120);title("GT permeability");colorbar();
    subplot(2,2,3);
    imshow(exp.(logK_init)'./md,vmin=20,vmax=120);title("initial permeability");colorbar();
    subplot(2,2,4);
    imshow(abs.(K'-exp.(logK0)')./md,vmin=20,vmax=120);title("error, SNR=$SNR");colorbar();
    suptitle("End-to-end Inversion at iter $j, seismic data snr=$snr")
    tight_layout()
    safesave(joinpath(plotsdir(sim_name, exp_name), savename(fig_name; digits=6)*"_K.png"), fig);
    close(fig)

    ## loss
    fig = figure(figsize=(20,12));
    plot(fhistory[1:j]);title("loss=$(fhistory[j])");
    suptitle("Learned Coupled Inversion at iter $j, seismic data snr=$snr")
    tight_layout()
    safesave(joinpath(plotsdir(sim_name, exp_name), savename(fig_name; digits=6)*"_loss.png"), fig);
    close(fig)

    ## data fitting
    fig = figure(figsize=(20,12));
    for i = 1:5
        subplot(4,5,i);
        imshow(y_init[2*i,:,:]', vmin=0, vmax=1);
        title("initial prediction at snapshot $(survey_indices[2*i])")
        subplot(4,5,i+5);
        imshow(sw_true[2*i,:,:]', vmin=0, vmax=1);
        title("true at snapshot $(survey_indices[2*i])")
        subplot(4,5,i+10);
        imshow(y_predict[2*i,:,:]', vmin=0, vmax=1);
        title("predict at snapshot $(survey_indices[2*i])")
        subplot(4,5,i+15);
        imshow(5*abs.(sw_true[2*i,:,:]'-y_predict[2*i,:,:]'), vmin=0, vmax=1);
        title("5X diff at snapshot $(survey_indices[2*i])")
    end
    suptitle("Learned Coupled Inversion at iter $j, seismic data snr=$snr")
    tight_layout()
    safesave(joinpath(plotsdir(sim_name, exp_name), savename(fig_name; digits=6)*"_saturation.png"), fig);
    close(fig)

end
