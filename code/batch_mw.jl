#=

Adapted from code at the following documentation link: https://diffeqflux.sciml.ai/stable/examples/minibatch/

Neural ODE batch training for the Midwest Region. 
    
=#

using DifferentialEquations, Flux, Optim, DiffEqFlux, Plots, CSV, DataFrames, DelimitedFiles, Sundials, Statistics
using IterTools: ncycle 

make_Q = readdlm("tuples/yearly_tuples_prop_NJ.csv", ',', Float64)
Q = ifelse.(make_Q .> 0, 1, make_Q)
MW = ["IA", "IL", "IN", "KS", "MI", "MN", "MO", "NE", "OH", "WI"]

tstart = 0.0
tend = 20.0
sampling = 1

model_params_mw = [sqrt(0.3), sqrt(0.9), sqrt(0.19), sqrt(0.5), sqrt(0.01159)]

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
    prob=ODEProblem(model, batch[:, 1],(tstart,tend), model_params_mw)
    sol = Array(concrete_solve(prob, Tsit5(), batch[:, 1], model_params_mw, saveat=tstart:sampling:tend, 
            abstol=1e-9,reltol=1e-9, sensealg = ForwardDiffSensitivity()))
    sol
end

function loss_adjoint(batch)
    pred = predict_adjoint(batch)
    loss = sum(abs2, Q.*(batch' - pred'))#, pred
    loss
end

global cat_mw = Array{Float64}(undef, 0, 6)
for i in 1:length(MW)
    data = readdlm("tuples/yearly_tuples_prop_" * MW[i] * ".csv", ',', Float64)
    global cat_mw = vcat(cat_mw, data)   
end

train_loader_mw = Flux.Data.DataLoader(cat_mw', batchsize = 21)

for x in train_loader_mw
    @assert size(x) == (6, 21)
    x = x'
end

const losses_mw=[]
cb() = begin
    ls = []
    for x in train_loader_mw
        l = loss_adjoint(x)[1]
        push!(ls, l)
    end
    push!(losses_mw, mean(ls))
end

ps_mw = Flux.params(model_params_mw)
opt=ADAM(0.001)

Flux.train!(loss_adjoint, ps_mw, ncycle(train_loader_mw, 10000), opt, cb = cb)
println("Fitted parameters:")
println("$((model_params_mw.^2))")

avg_losses_mw = []
for i in 1:length(losses_mw)
    if i%length(MW) == 0
        push!(avg_losses_mw, losses_mw[i])
    end
end

ls = plot(avg_losses_mw, xlabel = "Iterations", ylabel = "Loss", yaxis = :log, legend = false)
savefig(ls, "figs/loss_mw.pdf")
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
    savefig(pl, "figs/fitdyn_" * title * "_mw.pdf")
    display(pl)
#     return(Array(sol_fit))
end

for i in 1:length(MW)
    dat = readdlm("tuples/yearly_tuples_prop_" * MW[i] * ".csv", ',', Float64)
    plotFit(model_params_mw, dat[1,:], dat, MW[i])
end