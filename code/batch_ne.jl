#=

Adapted from code at the following documentation link: https://diffeqflux.sciml.ai/stable/examples/minibatch/

Neural ODE batch training for the Northeast Region. 
    
=#

using DifferentialEquations, Flux, Optim, DiffEqFlux, Plots, CSV, DataFrames, DelimitedFiles, Sundials, Statistics
using IterTools: ncycle 

make_Q = readdlm("tuples/yearly_tuples_prop_NJ.csv", ',', Float64)
Q = ifelse.(make_Q .> 0, 1, make_Q)
NE = ["CT", "MA", "ME", "NH", "NJ", "NY", "PA", "RI", "VT"]

tstart = 0.0
tend = 20.0
sampling = 1

model_params_ne = [sqrt(0.3), sqrt(0.9), sqrt(0.19), sqrt(0.5), sqrt(0.01159)]

# take square root of all the numbers and then square them in the model
alpha_sr = sqrt(0.15)
gamma_sr = sqrt(0.00744)
delta_sr = sqrt(0.1)
sigma_sr = sqrt(0.9)

function model(du, u, p, t)
    S, P, I, A, R, D = u

    N = S + P + I + A + R + D
    
    phi_sr, epsilon_sr, beta_sr, zeta_sr, mu_sr = p

    du[1] = -(alpha_sr^2)*S + (epsilon_sr^2)*P + (delta_sr^2)*R
    du[2] = (alpha_sr^2)*S - ((epsilon_sr^2) + (gamma_sr^2) + (beta_sr^2))*P
    du[3] = (beta_sr^2)*P - (phi_sr^2)*I
    du[4] = (gamma_sr^2)*P + (sigma_sr^2)*R + (phi_sr^2)*I - (zeta_sr^2)*A - (mu_sr^2)*A
    du[5] = (zeta_sr^2)*A - ((delta_sr^2) + (sigma_sr^2))*R
    du[6] = (mu_sr^2)*A

end

function predict_adjoint(batch)
    prob=ODEProblem(model, batch[:, 1],(tstart,tend), model_params_ne)
    sol = Array(concrete_solve(prob, Tsit5(), batch[:, 1], model_params_ne, saveat=tstart:sampling:tend, 
            abstol=1e-9,reltol =1e-9, sensealg = ForwardDiffSensitivity()))
    sol
end

function loss_adjoint(batch)
    pred = predict_adjoint(batch)
    loss = sum(abs2, Q.*(batch' - pred'))#, pred
    loss
end

global cat_ne = Array{Float64}(undef, 0, 6)
for i in 1:length(NE)
    data = readdlm("tuples/yearly_tuples_prop_" * NE[i] * ".csv", ',', Float64)
    global cat_ne = vcat(cat_ne, data)   
end

train_loader_ne = Flux.Data.DataLoader(cat_ne', batchsize = 21)

for x in train_loader_ne
    @assert size(x) == (6, 21)
    x = x'
end

const losses_ne=[]
cb() = begin
    ls = []
    for x in train_loader_ne
        l = loss_adjoint(x)[1]
        push!(ls, l)
    end
    push!(losses_ne, mean(ls))
end

ps_ne = Flux.params(model_params_ne)
opt=ADAM(0.001)

Flux.train!(loss_adjoint, ps_ne, ncycle(train_loader_ne, 10000), opt, cb = cb)
println("Fitted parameters:")
println("$((model_params_ne.^2))")

avg_losses_ne = []
for i in 1:length(losses_ne)
    if i%length(NE) == 0
        push!(avg_losses_ne, losses_ne[i])
    end
end

ls = plot(avg_losses_ne, xlabel = "Iterations", ylabel = "Loss", yaxis = :log, legend = false)
savefig(ls, "figs/loss_ne.pdf")
display(ls)

pops = DataFrame(CSV.File("pops.csv"))
function plotFit(param, u0, data, title)
    
    tspan=(tstart,20)
    sol_fit=solve(ODEProblem(model, u0, tspan, param), Tsit5(), saveat=tstart:sampling:tend)
    for i in 1:21
        sol_fit[:, i] = sol_fit[:, i]*pops[i, title]
    end
    ndata = pops[!, title].*data
    
    tgrid=tstart:sampling:tend
    pl=plot(sol_fit, vars = [2 3 4 5 6], lw=2, legend=:outertopright, label = ["P" "I" "A" "R" "D"])
    scatter!(pl,tgrid, ndata[:,2], color=:blue, label = "P")
    scatter!(pl,tgrid, ndata[:,3], color=:orange, label = "I")
    scatter!(pl,tgrid, ndata[:,4], color=:green, label = "A")
    scatter!(pl,tgrid, ndata[:,5], color=:pink, label = "R")
    scatter!(pl,tgrid, ndata[:,6], color=:brown, label = "D")
    xlabel!(pl,"Time")
    ylabel!(pl,"Population")
    # title!(pl,title)
    savefig(pl, "figs/fitdyn_" * title * "_ne.pdf")
    display(pl)
#     return(Array(sol_fit))
end

for i in 1:length(NE)
    dat = readdlm("tuples/yearly_tuples_prop_" * NE[i] * ".csv", ',', Float64)
    plotFit(model_params_ne, dat[1,:], dat, NE[i])
end